import argparse
from geotiff import GeoTiff
import numpy as np
import os
from patchify_tiff import patchify
from PIL import Image
from pyproj import Proj, transform
import shutil
from tag_geotiff import create_geotiff_metadata
import tempfile
import time
import warnings


def transform_to_wgs84(inputs, output_dir):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=Image.DecompressionBombWarning)

        os.makedirs(output_dir, exist_ok=True)
        geotiffs = []
        output_paths = []
        
        anonymous_idx = 0
        for input in inputs:
            if isinstance(input, GeoTiff):
                geotiffs.append(('%05i' % anonymous_idx, input))
                anonymous_idx += 1
            elif isinstance(input, str):
                path = None

                if os.path.isdir(input):
                    for root, dirs, files in os.walk(input):
                        for filename in files:
                            if not any(filename.lower().endswith(ext) for ext in ['.tif', '.tiff']):
                                continue
                            path = os.path.join(root, filename)
                            geotiffs.append((os.path.splitext(filename)[0], GeoTiff(path)))
                elif os.path.isfile(input):
                    geotiffs.append((os.path.splitext(os.path.basename(input))[0], GeoTiff(input)))
        
        for pair in geotiffs:
            name = pair[0]
            gt = pair[1]
            bbox = [*gt.tif_bBox[0], *gt.tif_bBox[1]]
            proj = Proj(f'EPSG:{gt.crs_code}')
            out_proj = Proj("+init=EPSG:4326")  # WGS84
            # (0,1): nw; (0,3): sw; (2,1): ne; (2,3): se
            pts = [transform(proj, out_proj, x, y) for x, y in [(bbox[i], bbox[j]) for i, j in [(0,1), (0,3), (2,1), (2,3)]]]
            print(f'Coords for "{name}": orig: {bbox} -> WGS84: {pts}')
            
            # get GCS (WGS84) outer bounding box from non-axis-aligned GCS rectangle
            
            # we want our images to be both aligned with the
            # GCS axes (one row/col ^= one lat/lon coord, resp.) and tileable
            # that means we need to "collapse" the images into an arbitrary x and an arbitrary y
            # direction (e.g. always take the minimum of all west coords, but also the minimum of all east coords,
            # "collapsing" the x dimension to the west/left)
            # instead of always taking the outer coords
            # (e.g. minimum over all x for west, maximum over all x for east, etc.)

            # these coords shall lie at the boundaries of the transformed image
            gcs_west_outer = min(pts[0][0], pts[1][0])
            gcs_east_outer = min(pts[2][0], pts[3][0])
            
            gcs_south_outer = max(pts[1][1], pts[3][1])
            gcs_north_outer = max(pts[0][1], pts[2][1])

            # matrix for mapping GCS coords to pixel in destination image
            # we will map gcs_west_outer to x @ 0, gcs_east_outer to x @ gt.width,
            # gcs_north_outer to y @ 0, gcs_south_outer to y @ gt.height

            gcs_to_pixel_b = np.array([0, 0,
                                       0, gt.tif_shape[0]-1,
                                       gt.tif_shape[1]-1, 0])
            
            gcs_to_pixel_A = np.array([[gcs_west_outer, gcs_north_outer, 1, 0, 0, 0],
                                       [0, 0, 0, gcs_west_outer, gcs_north_outer, 1],
                                       [gcs_west_outer, gcs_south_outer, 1, 0, 0, 0],
                                       [0, 0, 0, gcs_west_outer, gcs_south_outer, 1],
                                       [gcs_east_outer, gcs_north_outer, 1, 0, 0, 0],
                                       [0, 0, 0, gcs_east_outer, gcs_north_outer, 1]])

            gcs_to_pixel_M_vec = np.linalg.solve(gcs_to_pixel_A, gcs_to_pixel_b)
            gcs_to_pixel_M = np.array([[gcs_to_pixel_M_vec[0], gcs_to_pixel_M_vec[1], gcs_to_pixel_M_vec[2]],
                                    [gcs_to_pixel_M_vec[3], gcs_to_pixel_M_vec[4], gcs_to_pixel_M_vec[5]],
                                    [0, 0, 1]])

            # transform outer GCS coordinates back to original CRS
            orig_crs_se = transform(out_proj, proj, gcs_east_outer, gcs_south_outer)
            orig_crs_ne = transform(out_proj, proj, gcs_east_outer, gcs_north_outer)
            orig_crs_sw = transform(out_proj, proj, gcs_west_outer, gcs_south_outer)
            orig_crs_nw = transform(out_proj, proj, gcs_west_outer, gcs_north_outer)

            backtransformed_pts = [orig_crs_se, orig_crs_ne, orig_crs_sw, orig_crs_nw]

            # get original CRS outer bounding box from transformed coordinates
            orig_crs_west = min(pt[0] for pt in backtransformed_pts)
            orig_crs_east = max(pt[0] for pt in backtransformed_pts)
            orig_crs_south = min(pt[1] for pt in backtransformed_pts)
            orig_crs_north = max(pt[1] for pt in backtransformed_pts)

            # expected format: (LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)
            reproj_bbox = [orig_crs_west, orig_crs_south, orig_crs_east, orig_crs_north]
            
            # construct intermediate image from bbox; this function will automatically add content from all GeoTIFFs in
            # the input directory to fill the provided bbox as much as possible

            tmp_dir = tempfile.mkdtemp()
            result_paths = patchify(inputs, output_dir=tmp_dir, patch_width_px=0, patch_height_px=0, output_format='tiff',
                                    create_tags=True, keep_fractional=True, keep_blanks=True, bboxes=[reproj_bbox])

            # transform the resulting image such that 

            with Image.open(result_paths[0]) as img:
                # we need to project (0, 0) to the *reprojected coords*!
                new_gt = GeoTiff(result_paths[0])

                # matrix for translating original CRS to GCS coordinates

                crs_to_gcs_b = np.array([pts[0][0],
                                         pts[0][1],
                                         pts[1][0],
                                         pts[1][1],
                                         pts[2][0],
                                         pts[2][1]])
                crs_to_gcs_A = np.array([[bbox[0], bbox[1], 1, 0, 0, 0],
                                         [0, 0, 0, bbox[0], bbox[1], 1],
                                         [bbox[0], bbox[3], 1, 0, 0, 0],
                                         [0, 0, 0, bbox[0], bbox[3], 1],
                                         [bbox[2], bbox[1], 1, 0, 0, 0],
                                         [0, 0, 0, bbox[2], bbox[1], 1]])
                crs_to_gcs_M_vec = np.linalg.solve(crs_to_gcs_A, crs_to_gcs_b)
                crs_to_gcs_M = np.array([[crs_to_gcs_M_vec[0], crs_to_gcs_M_vec[1], crs_to_gcs_M_vec[2]],
                                        [crs_to_gcs_M_vec[3], crs_to_gcs_M_vec[4], crs_to_gcs_M_vec[5]],
                                        [0, 0, 1]])

                # matrix for translating pixels in the intermediate image to coordinates in the original CRS

                pixel_to_crs_b = np.array([new_gt.tif_bBox[0][0],
                                           new_gt.tif_bBox[0][1],
                                           new_gt.tif_bBox[1][0],
                                           new_gt.tif_bBox[1][1],
                                           new_gt.tif_bBox[0][0],
                                           new_gt.tif_bBox[1][1]])
                pixel_to_crs_A = np.array([[0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1],
                                           [new_gt.tif_shape[1]-1, new_gt.tif_shape[0]-1, 1, 0, 0, 0],
                                           [0, 0, 0, gt.tif_shape[1]-1, new_gt.tif_shape[0]-1, 1],
                                           [0, new_gt.tif_shape[0]-1, 1, 0, 0, 0],
                                           [0, 0, 0, 0, new_gt.tif_shape[0]-1, 1]])
                pixel_to_crs_M_vec = np.linalg.solve(pixel_to_crs_A, pixel_to_crs_b)
                pixel_to_crs_M = np.array([[pixel_to_crs_M_vec[0], pixel_to_crs_M_vec[1], pixel_to_crs_M_vec[2]],
                                          [pixel_to_crs_M_vec[3], pixel_to_crs_M_vec[4], pixel_to_crs_M_vec[5]],
                                          [0, 0, 1]])

                # final affine transformation matrix

                # note: "gcs_to_pixel_M @ crs_to_gcs_M @ pixel_to_crs_M" maps coords in the *source* (intermediate)
                # image to coordinates in the *destination* image, yet PIL expects a matrix mapping from coords in
                # the destination image to coords in the source image
                # hence, we need to invert this expression

                M = np.linalg.inv(gcs_to_pixel_M @ crs_to_gcs_M @ pixel_to_crs_M)

                # matrix must be provided to "transform" in vectorized form
                transformed_img = img.transform(gt.tif_shape[:2], Image.Transform.AFFINE,
                                                (M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2]))

                tl_coord = np.array([gcs_west_outer, gcs_north_outer, 1.0])
                br_coord = np.array([gcs_east_outer, gcs_south_outer, 1.0])

                tl_px = gcs_to_pixel_M @ tl_coord
                br_px = gcs_to_pixel_M @ br_coord
                tl_delta = np.linalg.inv(gcs_to_pixel_M) @ np.array([1.0, 1.0, 1.0]) - np.array([gcs_west_outer, gcs_north_outer, 1.0])
                tiff_meta = create_geotiff_metadata((0, 0), (gcs_west_outer, gcs_north_outer), (abs(tl_delta[0]), abs(tl_delta[1])),
                                                    crs_code=4326, crs_geo_citation=None, crs_gt_citation=None)

                cropped_img = transformed_img.crop((tl_px[0], tl_px[1] + 1, br_px[0], br_px[1] + 1))

                def get_fn(counter):
                    nonlocal name
                    ctr = '' if counter <= 1 else f'_{counter}'
                    fn = f'{name}{ctr}.tiff'
                    return fn

                counter = 0
                while True:
                    counter += 1
                    fn = get_fn(counter)
                    output_path = os.path.join(output_dir, fn)
                    if not os.path.isfile(output_path):
                        break

                cropped_img.save(output_path, tiffinfo=tiff_meta)
                output_paths.append(output_path)
            
            shutil.rmtree(tmp_dir)
    
        return output_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', help='Input director(y|ies)', type=str, action='append', default=None)
    parser.add_argument('-o', '--output-dir', help='Output directory', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'supremap_wgs84_transformation_{int(time.time())}'

    transform_to_wgs84(args.input_dir, args.output_dir)
