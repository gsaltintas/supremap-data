import argparse
import io
import os
import random
import shutil
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geojson import Feature, GeoJSON, Polygon
from geotiff import GeoTiff
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from PIL import Image, ImageColor
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from shapely import Polygon as PolygonShapely

import swisstopo_helpers as sw
from add_sentinel_imgs import add_sentinel_imgs
from osmCategories import *
from patchify_tiff import patchify
from transform_to_wgs84 import transform_to_wgs84


class SwisstopoDownloadArguments(object): 
    id = None
    date_range = None
    bbox = None
    resolution = 0.1
    save_dir = None
    max_rows = 0
    id = None


class AddSentinelImgsArguments(object):
    input_dir = None
    aligned_save_dir = None


def create_seg_inst_maps(data):
    entry_path_str, aligned_seg_maps_tiff_dir, aligned_instance_maps_tiff_dir, create_bg_instances, street_widths = data
    entry = Path(entry_path_str)
    
    join = os.path.join

    if not os.path.splitext(entry.name.lower())[-1] in {'.tif', '.tiff'}:
        return
    
    seg_map_output_path = join(aligned_seg_maps_tiff_dir, entry.name)
    instance_map_output_path = join(aligned_instance_maps_tiff_dir, entry.name)

    if (os.path.isfile(seg_map_output_path) and
        os.path.isfile(instance_map_output_path)):
        return
    
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

    seg_map_background_color = (255, 255, 255, 255)
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
    background.save(seg_map_output_path, tiffinfo=background.tag_v2)

    background = Image.new(
        mode='RGBA', size=(seg_map_resolution, seg_map_resolution), color=seg_map_background_color)
    for image in imagesInst:
        background = Image.alpha_composite(background, image)

    if create_bg_instances:
        # going through the png and coloring all connected black pixels
        img = np.array(background)
        img2 = (img == seg_map_background_color).all(2)
        seg_map_background_color = np.array(seg_map_background_color, dtype=np.uint8)

        def color_neighbourhood(x, y, newColor):
            nonlocal img, img2, seg_map_resolution, seg_map_background_color
            goals = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            while len(goals) > 0:
                xn, yn = goals.pop(0)
                # print(f'{xn}, {yn}')
                if (xn < 0 or xn >= seg_map_resolution or yn < 0 or yn >= seg_map_resolution):
                    continue
                if (img2[xn, yn] == True):
                    goals.extend(
                        [(xn-1, yn), (xn+1, yn), (xn, yn-1), (xn, yn+1)])
                    img2[xn, yn] = False
                    img[xn, yn] = newColor

        for x in range(seg_map_resolution):
            y = 0
            while (y < seg_map_resolution):
                # find true values and skip forward to it
                trues = img2[x].nonzero()[0]
                if (len(trues) > 0):
                    y = trues[0]
                else:
                    y = seg_map_resolution
                    continue

                # if its background colored, paint it
                if (img2[x, y] == True):
                    newColor = np.array(ImageColor.getcolor(f'#{offset:06x}', 'RGBA'), dtype=np.uint8)
                    offset += 2
                    img[x, y] = newColor
                    img2[x, y] = False
                    color_neighbourhood(x, y, newColor)

                y += 1
        
        background = Image.fromarray(img)

    tag_bg()
    background.save(instance_map_output_path, tiffinfo=background.tag_v2)


def patchify_mp(data): 
    entry_str, subtype_name, subtype_input_dir, subtype_output_dir, subtype_initial_image_size,\
    sqrt_num_patches_per_point, patch_size, geojson_dir = data

    patchify([entry_str], output_dir=subtype_output_dir,
             patch_width_px=subtype_initial_image_size // sqrt_num_patches_per_point,
             patch_height_px=subtype_initial_image_size // sqrt_num_patches_per_point,
             output_format='png',
             create_tags=False,
             keep_fractional=True,
             keep_blanks=False,
             output_naming_scheme='patch_idx',
             resizing_interpolation_method='nearest_neighbor' if '_map' in subtype_name else 'bicubic',
             output_naming_prefix=os.path.splitext(os.path.basename(entry_str))[0].replace('_lowres', ''),
             regular_patch_size=(patch_size, patch_size),
             geojson_dir=geojson_dir)


def create_imaginaire_dataset(output_dir, points, zoom_level, sqrt_num_patches_per_point, patch_size,
                              train_fraction=0.8, shuffle=True,
                              street_widths={'footway': 1.5, 'steps': 1.5, 'pedestrian': 1.5, 'service': 1.5,
                                             'path': 1.5, 'track': 1.5, 'motorway': 6, 'default': 5},
                              unaligned_images_tiff_dir=None, aligned_images_tiff_dir=None,
                              aligned_instance_maps_tiff_dir=None, aligned_seg_maps_tiff_dir=None,
                              aligned_low_res_tiff_dir=None, create_bg_instances=True, num_jobs=None,
                              download_or_transform_images=True, geojson_dir=None, random_seed=1):
    if num_jobs is None:
        num_jobs = max(1, (cpu_count() * 3) // 4)

    for point_idx, point in enumerate(points or []):
        if isinstance(point, str):
            points[point_idx] = eval(point)
    
    join = os.path.join
    for split in ['train', 'val']:
        for dir in ['blurred', 'images', 'instance_maps', 'seg_maps', 'geojsons']:
            os.makedirs(join(output_dir, split, dir), exist_ok=True)

    if unaligned_images_tiff_dir in {None, ''}:
        unaligned_images_tiff_dir = join(output_dir, 'creation', 'whole', 'unaligned_images')
    
    if aligned_images_tiff_dir in {None, ''}:
        aligned_images_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_images')
    
    if aligned_low_res_tiff_dir in {None, ''}:
        aligned_low_res_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_low_res')

    if aligned_instance_maps_tiff_dir in {None, ''}:
        aligned_instance_maps_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_instance_maps')

    if aligned_seg_maps_tiff_dir in {None, ''}:
        aligned_seg_maps_tiff_dir = join(output_dir, 'creation', 'whole', 'wgs84_seg_maps')
        
    if geojson_dir in {None, ''}:
        geojson_dir = join(output_dir, 'creation', 'whole', 'wgs_84_geojson')
        
    
    aligned_images_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_images')
    aligned_low_res_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_low_res')
    aligned_instance_maps_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_instance_maps')
    aligned_seg_maps_patch_dir = join(output_dir, 'creation', 'patchified', 'wgs84_seg_maps')

    for dir_path in {unaligned_images_tiff_dir, aligned_images_tiff_dir, aligned_low_res_tiff_dir,
                     aligned_instance_maps_tiff_dir, aligned_seg_maps_tiff_dir, geojson_dir}:
        os.makedirs(dir_path, exist_ok=True)

    bboxes = []
    # download required TIFFs
    for point in points:
        bbox = ox.utils_geo.bbox_from_point((point[1], point[0]), dist=zoom_level)
        bboxes.append(bbox)

        if download_or_transform_images:
            sw_args = SwisstopoDownloadArguments()
            sw_args.save_dir = Path(unaligned_images_tiff_dir)
            sw_args.bbox = [bbox[3], bbox[1], bbox[2], bbox[0]]  # [LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH]
            url = sw.get_url(sw_args)
            sw.download_tifs(url, sw_args)

    if download_or_transform_images:
        # transform to WGS-84
        transform_to_wgs84(inputs=[unaligned_images_tiff_dir], output_dir=aligned_images_tiff_dir)
    
    # determine relevant TIFFs

    relevant_tiff_names = []
    for entry in Path(aligned_images_tiff_dir).iterdir():
        enl = entry.name.lower()
        if not enl.endswith('.tif') and not enl.endswith('.tiff'):
            continue
        
        gt = GeoTiff(str(entry))
        tb = gt.tif_bBox
        relevant = False
        for bbox in bboxes:
            if any([tb[0][0] <= bbox[3] < tb[1][0], tb[0][0] < bbox[2] <= tb[1][0],
                    tb[1][1] < bbox[0] <= tb[0][1], tb[1][1] <= bbox[1] < tb[0][1],
                    bbox[3] <= tb[0][0] < bbox[2], bbox[3] < tb[1][0] <= bbox[2],
                    bbox[1] < tb[0][1] <= bbox[0], bbox[1] <= tb[1][1] < bbox[0]]):
                relevant = True
                break
        if relevant:
            relevant_tiff_names.append(entry.name)

    # add sentinel data

    sentinel_args = AddSentinelImgsArguments()
    sentinel_args.input_dir = aligned_images_tiff_dir
    sentinel_args.aligned_save_dir = aligned_low_res_tiff_dir
    add_sentinel_imgs(sentinel_args)

    # create instance/seg maps

    create_seg_inst_maps_data = [(path, aligned_seg_maps_tiff_dir, aligned_instance_maps_tiff_dir,
                                create_bg_instances, street_widths)
                                for path in Path(aligned_images_tiff_dir).iterdir()
                                if path.name in relevant_tiff_names]

    if num_jobs == 1:
        for data_entry in create_seg_inst_maps_data:
            create_seg_inst_maps(data_entry)
    else:    
        with Pool(processes=num_jobs) as pool:
            pool.map(create_seg_inst_maps, create_seg_inst_maps_data)

    # patchify the resulting directories

    subtype_dirs_sizes = [('images', aligned_images_tiff_dir, aligned_images_patch_dir, 10000),
                          ('low_res', aligned_low_res_tiff_dir, aligned_low_res_patch_dir, 10000),
                          ('instance_maps', aligned_instance_maps_tiff_dir, aligned_instance_maps_patch_dir, 10000),
                          ('seg_maps', aligned_seg_maps_tiff_dir, aligned_seg_maps_patch_dir, 10000)]

    # "patchify_mp_data" expects: entry, subtype_name, subtype_input_dir, subtype_output_dir,
    # subtype_initial_image_size, sqrt_num_patches_per_point, patch_size, geojson_dir
    patchify_mp_data = [(str(entry), sds[0], sds[1], sds[2], sds[3], sqrt_num_patches_per_point, patch_size, geojson_dir)
                         for sds in subtype_dirs_sizes
                         for entry in Path(sds[1]).iterdir() if entry.name.replace('_lowres', '') in relevant_tiff_names]

    if num_jobs == 1:
        for data in patchify_mp_data:
            patchify_mp(data)
    else:
        with Pool(processes=num_jobs) as pool:
            pool.map(patchify_mp, patchify_mp_data)
    
    copy_list = []
    for image_path in Path(aligned_images_patch_dir).iterdir():
        if not image_path.name.endswith('.png'):
            continue
        
        copy_dict = {'blurred': join(aligned_low_res_patch_dir, image_path.name),
                     'images': str(image_path),
                     'instance_maps': join(aligned_instance_maps_patch_dir, image_path.name),
                     'seg_maps': join(aligned_seg_maps_patch_dir, image_path.name),
                     'geojsons': join(geojson_dir, image_path.with_suffix('.geojson').name)}
        if any(map(lambda p: not os.path.isfile(p), copy_dict.values())):
            continue
        
        copy_list.append(copy_dict)
        
    if shuffle:
        random.Random(random_seed).shuffle(copy_list)

    num_samples = len(copy_list)
    num_train_samples = int(train_fraction * num_samples)
    
    for sample_idx, sample_copy_data in enumerate(copy_list):
        is_train = sample_idx < num_train_samples
        split_dir = 'train' if is_train else 'val'
        for subtype, path in sample_copy_data.items():
            shutil.copy(path, join(output_dir, split_dir, subtype, f'{sample_idx:08}{Path(path).suffix}'))
    
    for split_dir in {'train', 'val'}:
        for subtype in {'instance_maps', 'seg_maps'}:
            shutil.copytree(join(output_dir, split_dir, subtype), join(output_dir, split_dir, subtype + '_visual'))

    # turn segmentation & instance maps into 8-bit PNGs
    for sample_idx, sample_copy_data in enumerate(copy_list):
        is_train = sample_idx < num_train_samples
        split_dir = 'train' if is_train else 'val'
        for subtype, path in sample_copy_data.items():
            if subtype not in {'instance_maps', 'seg_maps'}:
                continue

            path = join(output_dir, split_dir, subtype, f'{sample_idx:08}{Path(path).suffix}')
            if not os.path.isfile(path):
                continue

            with Image.open(path) as img:
                img_array = np.array(img)
                new_array = np.zeros(img_array.shape[:2], dtype=np.uint8)
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

                if subtype == 'seg_maps':
                    new_array[np.logical_and(np.logical_and(r == 255, g == 255), b == 255)] = 0  # background: #FFFFFF
                    new_array[np.logical_and(np.logical_and(r == 0, g == 0), b == 0)] = 1  # building: #000000
                    new_array[np.logical_and(np.logical_and(r == 255, g == 0), b == 0)] = 2  # road: #FF0000
                    new_array[np.logical_and(np.logical_and(r == 0, g == 128), b == 0)] = 3  # green: #008000
                    new_array[np.logical_and(np.logical_and(r == 0, g == 0), b == 255)] = 4  # water: #0000FF
                    new_array[np.logical_and(np.logical_and(r == 165, g == 42), b == 42)] = 5  # beach: #0000FF
                elif subtype == 'instance_maps':
                    color_ints = 256 * 256 * r + 256 * g + b
                    unique = np.unique(color_ints)
                    for unique_idx, unique_val in enumerate(unique):
                        if unique_val > 0:
                            new_array[color_ints == unique_val] = min(255, unique_idx)

                with Image.fromarray(new_array, mode='L') as new_img:
                    new_img.save(path)

    try:
        shutil.rmtree(join(output_dir, 'creation'))
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', help='Directory to create dataset in.', type=str)
    parser.add_argument('-z', '--zoom-level', help='Zoom level to use.', type=int)
    parser.add_argument('-p', '--point', help='Bounding box, in WGS84 coords and format (LON, LAT). '\
                        'Possible to specify multiple.', type=str, action='append', default=[])
    parser.add_argument('-c', '--csv', help='Path to CSV file with "x_center" and "y_center" columns. '\
                        'Possible to specify multiple.', type=str, action='append', default=[])
    parser.add_argument('-t', '--train-fraction', help='Fraction of points to use for training from totality of all '\
                                                       'training + validation samples', default=0.8, type=float)
    parser.add_argument('-s', '--shuffle', help='Whether to shuffle the points when creating the splits.',
                        default=True, type=bool)
    parser.add_argument('-n', '--sqrt-num-patches-per-point',
                        help='Square root n of number of patches to generate per point. Final number of patches will '\
                             'be <= p * n^2, where n is this parameter and p is the number of points.',
                        default=10, type=int)
    parser.add_argument('-S', '--patch-size', help='Final patch side size, in pixels.', type=int, default=256)
    parser.add_argument('--create-bg-instances', help='Whether to generate separate instances for each background patch.'\
                        ' Doing so helps style encoders provide information to the generator about the appearance of '\
                        'the background patches.', type=bool, default=True)
    parser.add_argument('--unaligned-images-tiff-dir',
                        help='Directory into which to download high-res images. Omit to create a new one.',
                        type=str, default='/data/swisstopo_custom_2056/')
    parser.add_argument('--aligned-images-tiff-dir',
                        help='Directory into which to store WGS84-aligned high-res images. Omit to create a new one.',
                        type=str, default='/data/swisstopo_custom_wgs84/')
    parser.add_argument('--aligned-low-res-tiff-dir',
                        help='Directory into which to store WGS84-aligned Sentinel images. Omit to create a new one.',
                        type=str, default='/data/sentinel_wgs84/')
    parser.add_argument('--aligned-instance-maps-tiff-dir',
                        help='Directory into which to store WGS84-aligned instance maps. Omit to create a new one.',
                        type=str, default='/data/instance_maps_wgs84/')
    parser.add_argument('--aligned-seg-maps-tiff-dir',
                        help='Directory into which to store WGS84-aligned segmentation maps. Omit to create a new one.',
                        type=str, default='/data/seg_maps_wgs84/')
    parser.add_argument('--download-or-transform-images', help='Whether to download/transform any Swisstopo tiles.',
                        type=bool, default=False)
    parser.add_argument('-j', '--num-jobs', help='Number of jobs to use. Omit to use floor(0.75 * core_count).',
                        type=int, default=None)
    parser.add_argument('-R', '--random-seed', help='Seed to use when creating the train/val splits randomly.',
                        type=int, default=1)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--geojson-dir', help='Path to the geojsons.', type=str, default=None)
        
    args = parser.parse_args()
    random.seed(args.seed)
    
    if args.output_dir is None:
        args.output_dir = f'supremap_imaginaire_dataset_{int(time.time())}'

    for csv_path in args.csv or []:
        with open(csv_path) as f:
            csv_str = f.read()
            csv_lines = list(filter(str.__len__, csv_str.splitlines()))
            if len(csv_lines) == 0:
                continue
            columns = list(map(str.strip, csv_lines[0].split(',')))
            try:
                x_center_idx = columns.index('x_center')
                y_center_idx = columns.index('y_center')
            except ValueError:
                continue 
            
            for csv_line in csv_lines[1:]:
                cells = list(map(str.strip, csv_line.split(',')))
                try:
                    args.point.append((float(cells[x_center_idx]), float(cells[y_center_idx])))
                except:
                    pass

    create_imaginaire_dataset(args.output_dir, args.point, args.zoom_level, args.sqrt_num_patches_per_point,
                              args.patch_size, args.train_fraction, args.shuffle,
                              unaligned_images_tiff_dir=args.unaligned_images_tiff_dir,
                              aligned_images_tiff_dir=args.aligned_images_tiff_dir,
                              aligned_low_res_tiff_dir=args.aligned_low_res_tiff_dir,
                              aligned_instance_maps_tiff_dir=args.aligned_instance_maps_tiff_dir,
                              aligned_seg_maps_tiff_dir=args.aligned_seg_maps_tiff_dir,
                              create_bg_instances=args.create_bg_instances,
                              download_or_transform_images=args.download_or_transform_images,
                              num_jobs=args.num_jobs, geojson_dir=args.geojson_dir,
                              random_seed=args.random_seed)
