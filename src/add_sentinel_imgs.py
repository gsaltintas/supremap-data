# run this script after the segmentation PNGs and the coordinate JSONs are in the "swisstopo" folder

import argparse
from geotiff import GeoTiff
import json
import os
from patchify_tiff import patchify
from pathlib import Path
from PIL import Image
from pyproj import Proj
import re
import sentinel
import shutil
from tag_geotiff import create_geotiff_metadata
import tempfile
from transform_to_wgs84 import transform_to_wgs84
import xml.etree.ElementTree as ET


class SentinelArguments(object):
    bbox = None
    rows = 5
    max_rows = 5
    max_cloud_coverage_pct = 2
    date_range = None
    order_by_position = None
    force_metadata_regeneration = True
    desc = None
    crs_code = None
    input_dir = Path('./swisstopo').resolve().absolute()
    save_dir = Path('./sentinel').resolve().absolute()
    uncropped_geotiff_dir = Path('./uncropped_geotiffs').resolve().absolute()
    processing_level = 'Level-2A'


def postprocess_sentinel_data(args):
    # coord_list should be [LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH]

    for saved_fpath in args.save_dir.iterdir():
      saved_fpath_str = str(saved_fpath)
      if saved_fpath_str.lower().endswith('.zip'):
        try:
          sentinel.unzip_sentinel(saved_fpath_str)
          print(f'Unzipped {saved_fpath_str}')
        except:
          pass
        # delete either way
        os.unlink(saved_fpath_str)
    
    os.makedirs(args.uncropped_geotiff_dir, exist_ok=True)
    
    for path_0 in args.save_dir.iterdir():
      if not path_0.is_dir():
        continue

      path_components = path_0.name.split('_')
      mgrs_code = next(filter(lambda s: s.startswith('T'), path_components), None)
      if mgrs_code is None:
        continue
      mgrs_code = mgrs_code[1:]

      for path_1 in path_0.joinpath('GRANULE').iterdir():
        if not path_1.is_dir():
          continue

        metadata_json_path = os.path.join(path_1, 'metadata.json')

        # extract important metadata into JSON

        metadata_xml_path = os.path.join(path_1, 'MTD_TL.xml')
        if not os.path.isfile(metadata_xml_path):
          continue
        
        if args.force_metadata_regeneration or not os.path.isfile(metadata_json_path):
          tree = ET.parse(metadata_xml_path)
          root = tree.getroot()
          cloudiness = float(root.find('.//CLOUDY_PIXEL_PERCENTAGE').text)
          timestamp = root.find('.//SENSING_TIME').text
          hor_cs_name = root.find('.//HORIZONTAL_CS_NAME').text
          hor_cs_code = root.find('.//HORIZONTAL_CS_CODE').text
          
          geoposition_x_tl = float(root.find('.//Geoposition/ULX').text)
          geoposition_y_tl = float(root.find('.//Geoposition/ULY').text)

          geoposition_xdim = float(root.find('.//Geoposition/XDIM').text)
          geoposition_ydim = float(root.find('.//Geoposition/YDIM').text)

          nrows = float(root.find('.//NROWS').text)
          ncols = float(root.find('.//NCOLS').text)

          geoposition_x_br = geoposition_x_tl + geoposition_xdim * ncols
          geoposition_y_br = geoposition_y_tl + geoposition_ydim * nrows

          proj = Proj(hor_cs_code)
          lon_tl, lat_tl = proj(geoposition_x_tl, geoposition_y_tl, inverse=True)
          lon_br, lat_br = proj(geoposition_x_br, geoposition_y_br, inverse=True)

          pixel_scale_x = abs(geoposition_xdim)
          pixel_scale_y = abs(geoposition_ydim)

          meta = {'version': 1,
                  'cloudiness': cloudiness,
                  'timestamp': timestamp,
                  'hor_cs_name': hor_cs_name,
                  'hor_cs_code': hor_cs_code,
                  'mgrs_code': mgrs_code,
                  'nrows': nrows,
                  'ncols': ncols,
                  'pixel_scale_x': pixel_scale_x,
                  'pixel_scale_y': pixel_scale_y,
                  'geopos_orig_x_top_left': geoposition_x_tl,
                  'geopos_orig_y_top_left': geoposition_y_tl,
                  'geopos_orig_x_bottom_right': geoposition_x_br,
                  'geopos_orig_y_bottom_right': geoposition_y_br,
                  'geopos_proj_lon_top_left': lon_tl,
                  'geopos_proj_lat_top_left': lat_tl,
                  'geopos_proj_lon_bottom_right': lon_br,
                  'geopos_proj_lat_bottom_right': lat_br
                 }
          
          with open(metadata_json_path, 'w') as f:
            json.dump(meta, f)
          print(f'Metadata extracted from "{metadata_xml_path}" and written to "{metadata_json_path}"')
        elif os.path.isfile(metadata_json_path):
            with open(metadata_json_path, 'r') as f:
                meta = json.load(f)
            nrows = meta['nrows']
            ncols = meta['ncols']
            geoposition_x_tl = meta['geopos_orig_x_top_left']
            geoposition_y_tl = meta['geopos_orig_y_top_left']
            geoposition_x_br = meta['geopos_orig_x_bottom_right']
            geoposition_y_br = meta['geopos_orig_y_bottom_right']
            lon_tl = meta['geopos_proj_lon_top_left']
            lat_tl = meta['geopos_proj_lat_top_left']
            lon_br = meta['geopos_proj_lon_bottom_right']
            lat_br = meta['geopos_proj_lat_bottom_right']
            pixel_scale_x = meta['pixel_scale_x']
            pixel_scale_y = meta['pixel_scale_y']
            hor_cs_name = meta['hor_cs_name']
            hor_cs_code = meta['hor_cs_code']

            proj = Proj(hor_cs_code)

        img_data_dir = os.path.join(path_1, 'IMG_DATA')
        if not os.path.isdir(img_data_dir):
          continue

        for root, dirs, files in os.walk(path_1):
          for fn in files:
            fn_l = fn.lower()
            if not fn_l.endswith('.jp2') or any(['_10m' not in fn_l, '_tci' not in fn_l]):
              continue
      
            old_img_path = os.path.join(root, fn)
            new_fn = fn.replace('.jp2', '.tiff').replace('.JP2', '.TIFF')
            new_img_path = f'{args.uncropped_geotiff_dir}/{new_fn}'
            if not os.path.isfile(new_img_path):
              print(f'Converting: {old_img_path} to {new_img_path}... ', end=None)
              pixel_x_y = (0.0, 0.0)
              pixel_lon_lat = (geoposition_x_tl, geoposition_y_tl)
              pixel_scale_x_y = (pixel_scale_x, pixel_scale_y)
              crs_code = int(hor_cs_code.split(':')[-1].strip())
              crs_geo_citation = hor_cs_name.split('/')[0].strip()
              crs_gt_citation = hor_cs_name
              print(f'{pixel_x_y=} {pixel_lon_lat=} {pixel_scale_x_y=} {crs_code=} {crs_geo_citation=} {crs_gt_citation=}')
              metadata = create_geotiff_metadata(pixel_x_y, pixel_lon_lat, pixel_scale_x_y, crs_code, crs_geo_citation,
                                                 crs_gt_citation)
              with Image.open(old_img_path) as jp2_img:
                jp2_img.save(new_img_path, tiffinfo=metadata)
              print('OK!')


def download_sentinel_data(args):
    coord_dir = Path(args.input_dir)
    for entry in coord_dir.iterdir():
        if entry.name.endswith('.tif') and all(s not in entry.name for s in ['_segmentation', '_lowres']):
            print(f'***** {entry.name} *****')
            image_id = re.search('swissimage-dop10_([0-9]+)_[0-9]+-[0-9]+', entry.name)[0]
            json_file = f'{image_id}.json'
            with open(coord_dir.joinpath(json_file), 'r') as f:
              coord_list = json.load(f)
            # format: [LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH]
            input_bbox = [coord_list[0][0], coord_list[1][1], coord_list[3][0], coord_list[0][1]]
            
            print(f'{coord_list=}')
            print(f'{input_bbox=}')

            sentinel_args.bbox = input_bbox
            url = sentinel.get_url(sentinel_args)
            downloaded = sentinel.download_zips(url, sentinel_args)
            print(f'Downloaded into: {downloaded}')
            postprocess_sentinel_data(sentinel_args)

            gt = GeoTiff(str(entry))

            # create temporary image of upscaled region
            margin = 0.01  # in lon/lat

            proj = Proj(args.crs_code)
            extended_bbox_wgs = [elem + m for elem, m in zip(input_bbox, [-margin, -margin, margin, margin])]
            
            extended_bbox_sw = proj.transform(extended_bbox_wgs[0], extended_bbox_wgs[1])
            extended_bbox_ne = proj.transform(extended_bbox_wgs[2], extended_bbox_wgs[3])
            extended_bbox = (extended_bbox_sw, extended_bbox_ne)

            tmp_dir = tempfile.mkdtemp()
            # need the bump here for alignment
            upsampled_img_path = patchify([str(args.uncropped_geotiff_dir)], tmp_dir, bboxes=[extended_bbox],
                                          bbox_patch_sizes=[(20, 20, True)],
                                          patch_width_px=0, patch_height_px=0, output_format='tiff',
                                          create_tags=True, keep_fractional=True, keep_blanks=True, add_bump=True)[0]

            transformed_img_path = transform_to_wgs84(inputs=[upsampled_img_path], output_dir=tmp_dir)[0]

            proj_bbox_sw = proj.transform(input_bbox[0], input_bbox[1])
            proj_bbox_ne = proj.transform(input_bbox[2], input_bbox[3])
            proj_bbox = (proj_bbox_sw, proj_bbox_ne)

            print(f'{proj_bbox=}')
            patchify(inputs=[transformed_img_path], output_dir=args.input_dir,
                     output_naming_scheme='prefix_only', output_format='tiff',
                     output_naming_prefix=os.path.splitext(entry.name)[0] + '_lowres',
                     patch_width_px=0, patch_height_px=0, create_tags=True, keep_fractional=True, keep_blanks=True,
                     bboxes=[input_bbox], bbox_patch_sizes=[(gt.tif_shape[1], gt.tif_shape[0])], add_bump=True)

            shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='Path to directory with coordinate JSONs to process.', type=str,
                        default='swisstopo')
    parser.add_argument('--crs-code', help='CRS code used in the area covered by the TIFFs to process.', type=str,
                        default='EPSG:32632')  # TODO: universalize
    args = parser.parse_args()
    sentinel_args = SentinelArguments()
    sentinel_args.input_dir = args.input_dir
    sentinel_args.crs_code = args.crs_code
    download_sentinel_data(sentinel_args)
