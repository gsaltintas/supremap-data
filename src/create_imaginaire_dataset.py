import argparse
import cairosvg
from geojson import Feature, GeoJSON, Polygon
from geotiff import GeoTiff
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import os
import osmnx as ox
import io
from pathlib import Path
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2
import random
from shapely import Polygon as PolygonShapely
import shutil
import time


from add_sentinel_imgs import add_sentinel_imgs
from osmCategories import *
from patchify_tiff import patchify
import swisstopo_helpers as sw
from transform_to_wgs84 import transform_to_wgs84


class SwisstopoDownloadArguments(object):
  date_range = None
  bbox = None
  resolution = 0.1
  save_dir = None
  max_rows = 0

class AddSentinelImgsArguments(object):
    input_dir = None
    aligned_save_dir = None

# def create_sample(point, zoom_level, blurred_path, image_path, instance_map_path, seg_map_path):


def create_imaginaire_dataset(output_dir, points, zoom_level, sqrt_num_patches_per_point, patch_size,
                              train_fraction=0.8, shuffle=True,
                              street_widths={'footway': 1.5, 'steps': 1.5, 'pedestrian': 1.5, 'service': 1.5,
                                             'path': 1.5, 'track': 1.5, 'motorway': 6, 'default': 5}):
    for point_idx, point in enumerate(points):
        if isinstance(point, str):
            points[point_idx] = eval(point)
    
    join = os.path.join
    for split in ['train', 'val']:
        for dir in ['blurred', 'images', 'instance_maps', 'seg_maps']:
            os.makedirs(join(output_dir, split, dir), exist_ok=True)

    unaligned_images_tiff_dir = join(output_dir, 'creation', 'whole', 'unaligned_images')
    aligned_images_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_images')
    aligned_low_res_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_low_res')
    aligned_instance_maps_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_instance_maps')
    aligned_seg_maps_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_seg_maps')
    

    aligned_images_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_images')
    aligned_low_res_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_low_res')
    aligned_instance_maps_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_instance_maps')
    aligned_seg_maps_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_seg_maps')

    for dir_path in {unaligned_images_tiff_dir, aligned_images_tiff_dir, aligned_low_res_tiff_dir,
                     aligned_instance_maps_tiff_dir, aligned_seg_maps_tiff_dir}:
        os.makedirs(dir_path, exist_ok=True)

    bboxes = []
    # download required TIFFs
    for point in points:
        bbox = ox.utils_geo.bbox_from_point((point[1], point[0]), dist=zoom_level)
        bboxes.append(bbox)

        sw_args = SwisstopoDownloadArguments()
        sw_args.save_dir = Path(unaligned_images_tiff_dir)
        sw_args.bbox = [bbox[3], bbox[1], bbox[2], bbox[0]]  # [LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH]
        url = sw.get_url(sw_args)
        sw.download_tifs(url, sw_args)

    # transform to WGS-84
    transform_to_wgs84(inputs=[unaligned_images_tiff_dir], output_dir=aligned_images_tiff_dir)
    
    # add sentinel data

    sentinel_args = AddSentinelImgsArguments()
    sentinel_args.input_dir = aligned_images_tiff_dir
    sentinel_args.aligned_save_dir = aligned_low_res_tiff_dir
    add_sentinel_imgs(sentinel_args)

    # create segmentation and instance maps
    for entry in Path(aligned_images_tiff_dir).iterdir():
        if not os.path.splitext(entry.name.lower())[-1] in {'.tif', '.tiff'}:
            continue
            
        gt = GeoTiff(str(entry))
        tif_bbox = gt.tif_bBox

        # [LON_EAST, LON_WEST, LAT_SOUTH, LAT_NORTH]
        # need nw, sw, se, ne, nw in lon-lat
        geo_coords = ((tif_bbox[0][0], tif_bbox[0][1]), (tif_bbox[0][0], tif_bbox[1][1]),
                      (tif_bbox[1][0], tif_bbox[1][1]), (tif_bbox[1][0], tif_bbox[0][1]),
                      (tif_bbox[0][0], tif_bbox[0][1]))

        poly = PolygonShapely(geo_coords)
        print(poly)
        
        plt.ioff()

        seg_map_background_color = 'white'
        seg_map_resolution = 10000

        # function to create lots of colors
        def getColormap(currOffset, currAmount):
            return ListedColormap([f'#{color:06x}' for color in range(currOffset, currOffset + currAmount)])

        def plotRoads(plotSeg, plotInst, offset, tags, color, street_widths):
            roads = ox.graph_from_polygon(
                poly, retain_all=True, truncate_by_edge=True)
            geo = ox.utils_graph.graph_to_gdfs(roads, nodes=False)

            for index, (type, size) in enumerate(street_widths.items()):
                typeGeo = geo.loc[geo['highway'] == type]
                amount = 2 * typeGeo.shape[0]
                typeGeo.plot(ax=plotSeg[index][1], color=color, linewidth=size)
                typeGeo.plot(ax=plotInst[index][1], cmap=getColormap(
                    offset, amount), linewidth=size)
                offset = offset + amount

            typeGeo = geo.loc[~geo['highway'].isin(street_widths.keys())]
            size = street_widths['default']
            amount = 2 * typeGeo.shape[0]
            typeGeo.plot(ax=plotSeg[-1][1], color=color, linewidth=size)
            typeGeo.plot(ax=plotInst[-1][1], cmap=getColormap(
                offset, amount), linewidth=size)
            offset = offset + amount

            return offset

        def plotGeometry(axisSeg, axisInst, offset, tags, color):
            geo = ox.geometries_from_polygon(
                poly, tags=tags)
            amount = 2 * geo.shape[0]
            geo.plot(ax=axisSeg, color=color)
            geo.plot(ax=axisInst, cmap=getColormap(offset, amount))
            return offset + amount

        # arrays to save layers of osm images
        imagesSeg = []
        imagesInst = []

        # offset for colormap creation
        offset = 0

        for index, (tag, color) in enumerate([(WaterTags, 'blue'), (FieldTags, 'green'), (BeachTags, 'brown'),
                                              ('roads', 'red'), (BuildingTags, 'black')]):
            figSeg, axSeg = plt.subplots(figsize=(10, 10))
            figInst, axInst = plt.subplots(figsize=(10, 10))

            plotSeg = [(figSeg, axSeg)]
            plotInst = [(figInst, axInst)]

            # draw content
            if (tag == 'roads'):
                # create an additional plot for each street width option
                for _ in street_widths:
                    plotSeg.append(plt.subplots(figsize=(10, 10)))
                    plotInst.append(plt.subplots(figsize=(10, 10)))

                print(f'Fetching and plotting layer {index} with roads')
                offset = plotRoads(plotSeg, plotInst, offset,
                                tag, color, street_widths)

            else:
                print(f'Fetching and plotting layer {index} with {tag.keys()}')
                offset = plotGeometry(axSeg, axInst, offset, tag, color)
                figSeg = []

            # convert ax to svg to png
            for plots, outputArr in [(plotSeg, imagesSeg), (plotInst, imagesInst)]:
                for fig, ax in plots:
                    # crop
                    bbox = [tif_bbox[1][1],tif_bbox[0][1], tif_bbox[0][0], tif_bbox[1][0]]
                    ax.set_xlim((bbox[2], bbox[3]))
                    ax.set_ylim((bbox[0], bbox[1]))
                    ax.set_axis_off()

                    fig.tight_layout(pad=-0.08)

                    # convert plot to svg, then save svg without anti-aliasing
                    imgdata = io.StringIO()
                    fig.savefig(imgdata, format='svg', transparent=True)
                    plt.close(fig)
                    imgdata.seek(0)  # rewind the data

                    svg_dta = imgdata.read()  # convert to string

                    # add no antialiasing to svg
                    svg_dta = svg_dta.replace(
                        "version=\"1.1\">", "shape-rendering=\"crispEdges\" version=\"1.1\">")

                    # save png to buffer, load with PIL and append to images
                    buffer = io.BytesIO()
                    cairosvg.svg2png(bytestring=svg_dta,
                                    write_to=buffer, output_height=seg_map_resolution)

                    buffer.seek(0)
                    outputArr.append(Image.open(buffer))

        # for saving with GeoTIFF information (same as in original)
        
        def tag_bg():    
            background.tag_v2 = ImageFileDirectory_v2()
            with Image.open(str(entry)) as reference_img:
                for to_copy in {1024, 1025, 2016, *range(2048, 2062), *range(3072, 3096), *range(4096, 5000),
                                33550, 33922, 34735, 34736, 34737}:
                    if to_copy in reference_img.tag_v2:
                        background.tag_v2[to_copy] = reference_img.tag_v2[to_copy]
            
                # add missing ModelTiepoint
                if 33922 not in background.tag_v2:
                    background.tag_v2[33922] = (0.0, 0.0, 0.0)
                
                # adapt ModelPixelScaleTag
                if background.width != reference_img.width or background.height != reference_img.height:
                    pst = background.tag_v2[33550]
                    background.tag_v2[33550] = (pst[0] * (reference_img.width / background.width),
                                                pst[1] * (reference_img.height / background.height),
                                                pst[2])
        
        # stack images and save to PNG file
        background = Image.new(
            mode='RGBA', size=(seg_map_resolution, seg_map_resolution), color=seg_map_background_color)
        for image in imagesSeg:
            background = Image.alpha_composite(background, image)

        tag_bg()
        background.save(join(aligned_seg_maps_tiff_dir, entry.name), tiffinfo=background.tag_v2)

        background = Image.new(
            mode='RGBA', size=(seg_map_resolution, seg_map_resolution), color=seg_map_background_color)
        for image in imagesInst:
            background = Image.alpha_composite(background, image)

        tag_bg()
        background.save(join(aligned_instance_maps_tiff_dir, entry.name), tiffinfo=background.tag_v2)

    # patchify the resulting directories

    subtype_dirs_sizes = [(aligned_images_tiff_dir, aligned_images_patch_dir, 10000),
                          (aligned_low_res_tiff_dir, aligned_low_res_patch_dir, 10000),
                          (aligned_instance_maps_tiff_dir, aligned_instance_maps_patch_dir, 10000),
                          (aligned_seg_maps_tiff_dir, aligned_seg_maps_patch_dir, 10000)]

    for subtype_input_dir, subtype_output_dir, subtype_initial_image_size in subtype_dirs_sizes:
        patchify([subtype_input_dir], output_dir=subtype_output_dir,
                  patch_width_px=subtype_initial_image_size // sqrt_num_patches_per_point,
                  patch_height_px=subtype_initial_image_size // sqrt_num_patches_per_point,
                  output_format='png',
                  create_tags=False,
                  keep_fractional=True,
                  keep_blanks=False,
                  output_naming_scheme='patch_idx',
                  regular_patch_size=(patch_size, patch_size))
    
    copy_list = []

    for image_path in Path(aligned_images_patch_dir).iterdir():
        if not image_path.name.endswith('.png'):
            continue
        
        copy_dict = {'blurred': join(aligned_low_res_patch_dir, image_path.name),
                     'images': str(image_path),
                     'instance_maps': join(aligned_instance_maps_patch_dir, image_path.name),
                     'seg_maps': join(aligned_seg_maps_patch_dir, image_path.name)}
        if any(map(lambda p: not os.path.isfile(p), copy_dict.values())):
            continue
        
        copy_list.append(copy_dict)
        
    if shuffle:
        random.shuffle(copy_list)

    num_samples = len(copy_list)
    num_train_samples = int(train_fraction * num_samples)
    
    for sample_idx, sample_copy_data in enumerate(copy_list):
        is_train = sample_idx < num_train_samples
        split_dir = 'train' if is_train else 'val'
        for subtype, path in sample_copy_data.items():
            shutil.copy(path, join(output_dir, split_dir, subtype, f'{sample_idx:08}.png'))

    shutil.rmtree(join(output_dir, 'creation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', help='Directory to create dataset in.', type=str)
    parser.add_argument('-z', '--zoom-level', help='Zoom level to use.', type=int)
    parser.add_argument('-p', '--point', help='Bounding box, in WGS84 coords and format (LON, LAT). '\
                        'Possible to specify multiple.', type=str, action='append')
    parser.add_argument('-t', '--train-fraction', help='Fraction of points to use for training from totality of all '\
                                                       'training + validation samples', default=0.8, type=float)
    parser.add_argument('-s', '--shuffle', help='Whether to shuffle the points when creating the splits.',
                        default=True, type=bool)
    parser.add_argument('-n', '--sqrt-num-patches-per-point',
                        help='Square root n of number of patches to generate per point. Final number of patches will'\
                             'be <= p * n^2, where n is this parameter and p is the number of points.',
                        default=3, type=int)
    parser.add_argument('-S', '--patch-size', help='Final patch side size, in pixels', type=int, default=768)

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f'supremap_imaginaire_dataset_{int(time.time())}'

    create_imaginaire_dataset(args.output_dir, args.point, args.zoom_level, args.sqrt_num_patches_per_point,
                              args.patch_size, args.train_fraction, args.shuffle)