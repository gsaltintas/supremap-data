import argparse
from geotiff import GeoTiff
import itertools
import numpy as np
import os
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from sortedcontainers import SortedDict, SortedList
import time

from utils import custom_get_int_box


def patchify(inputs, output_dir, patch_width_px, patch_height_px, output_format, create_tags, keep_fractional,
             keep_blanks, bboxes=None, regular_patch_size=None, bbox_patch_sizes=None,
             resizing_interpolation_method='bicubic', output_naming_scheme='patch_idx', output_naming_prefix='',
             add_bump=False):
    # "inputs" can be list of dirs, or GeoTiffs
    # should return list of GeoTiffs

    if resizing_interpolation_method not in ['', None]:
        filters = {'nearest': Image.NEAREST, 'nearest_neighbor': Image.NEAREST,
                   'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
        resizing_interpolation_filter = filters[resizing_interpolation_method]
    else:
        resizing_interpolation_filter = None

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    num_channels = -1

    # We assume all images use the same coordinate system, due to the nontrivial conversion between coordinate reference
    # systems.
    # We load all GeoTiffs. While doing so, we determine which one has the leftmost left corner, and which one has the
    # topmost top corner. Then, we determine the locations of each image, and start overlaying images sequentially
    # (for the *first* version).

    leftmost_geotiff = None
    topmost_geotiff = None
    first_crs_code = None
    num_channels = -1

    # construct 2 "event" multimaps: x (for x starts & ends)
    #                                y (for y starts & ends)
    # note that we need events in the same multimap to be ordered, and we need to support having the same key in the
    # multimap more than once (the latter requirement makes SortedDict ununsable for our purpose, yet we can work with
    # lists since we only do a full scan)
    # then go through whole image in patches

    first_geotiff_path = None

    geotiffs = []
    
    def process_geotiff(geotiff):
        nonlocal leftmost_geotiff, topmost_geotiff, num_channels
        bbox = geotiff.tif_bBox
        if leftmost_geotiff is None or bbox[0][0] < leftmost_geotiff.tif_bBox[0][0]:
            leftmost_geotiff = geotiff

        if topmost_geotiff is None or bbox[0][1] > topmost_geotiff.tif_bBox[0][1]:
            topmost_geotiff = geotiff

        if len(geotiff.tif_shape) != 3:
            return
        
        if num_channels == -1:
            num_channels = geotiff.tif_shape[-1]
        elif num_channels != geotiff.tif_shape[-1]:
            raise ValueError(f'Received TIFF images with different number of channels '\
                             f'({num_channels} vs. {geotiff.tif_shape[-1]})!')

        geotiffs.append(geotiff)

    for input in inputs:
        if isinstance(input, GeoTiff):
            process_geotiff(input)
        elif isinstance(input, str):
            gt = None
            path = None

            def prelim_process_gt(path, gt):
                nonlocal first_geotiff_path, first_crs_code
                if gt is not None:
                    if first_geotiff_path is None:
                        first_geotiff_path = path

                    gt._as_crs = gt._crs_code
                    
                    if first_crs_code is None:
                        first_crs_code = gt.crs_code
                    elif first_crs_code != gt.crs_code:
                        raise NotImplementedError('Transform TIFFs to one single CRS first before patchifying!')
                    
                    process_geotiff(gt)

            if os.path.isdir(input):
                for root, dirs, files in os.walk(input):
                    for filename in files:
                        if not any(filename.lower().endswith(ext) for ext in ['.tif', '.tiff']):
                            continue
                        path = os.path.join(root, filename)
                        prelim_process_gt(path, GeoTiff(path))
            elif os.path.isfile(input):
                prelim_process_gt(input, GeoTiff(input))
    
    if first_geotiff_path is None and create_tags:
        raise NotImplementedError('Must supply at least one path to a GeoTiff.')

    x_dict_pxs = []
    y_dict_pxs = []
    
    for gt in geotiffs:
        bbox = gt.tif_bBox

        leftmost_bbox = custom_get_int_box(leftmost_geotiff, bbox, add_bump=add_bump)
        topmost_bbox = custom_get_int_box(topmost_geotiff, bbox, add_bump=add_bump)

        x_start_px = leftmost_bbox[0][0]
        y_start_px = topmost_bbox[0][1]
        
        x_end_px = leftmost_bbox[1][0]
        y_end_px = topmost_bbox[1][1]

        x_dict_pxs.append((x_start_px, ('start', gt)))
        y_dict_pxs.append((y_start_px, ('start', gt)))
        
        x_dict_pxs.append((x_end_px, ('end', gt)))
        y_dict_pxs.append((y_end_px, ('end', gt)))

    # note that the first/last entries of these lists will always be "start"/"end" events
    
    x_dict_pxs.sort(key=lambda el: el[0])
    y_dict_pxs.sort(key=lambda el: el[0])

    x_keys = list(map(lambda el: el[0], x_dict_pxs))
    y_keys = list(map(lambda el: el[0], y_dict_pxs))

    num_x_keys = len(y_dict_pxs)

    patch_y_xs = SortedDict()

    if min(patch_width_px, patch_height_px) > 0:
        for patch_y_idx in itertools.count():
            patch_y_start = patch_y_idx * patch_height_px
            patch_y_end = (patch_y_idx + 1) * patch_height_px

            if patch_y_start >= y_keys[-1] or (not keep_fractional and patch_y_end >= y_keys[-1]):
                break
            
            patch_y_xs[(patch_y_start, patch_y_end)] = SortedList()

            x_keys_pos = 0
            for patch_x_idx in itertools.count():
                patch_x_start = patch_x_idx * patch_width_px
                patch_x_end = (patch_x_idx + 1) * patch_width_px
                
                if patch_x_start >= x_keys[-1] or (not keep_fractional and patch_x_end >= x_keys[-1]):
                    break
                
                patch_y_xs[(patch_y_start, patch_y_end)].add((patch_x_start, patch_x_end))

    if bboxes not in [None, []]:
        for bbox_idx, bbox in enumerate(bboxes):
            if isinstance(bbox, str):
                bbox = eval(bbox)
            if len(bbox) == 2:
                bbox = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
            
            # bbox provided in format: (LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)
            # get_int_box expects: (LON_WEST, LAT_NORTH, LON_EAST, LAT_SOUTH)
            gt_bbox = ((bbox[0], bbox[3]), (bbox[2], bbox[1]))
            int_box_x = custom_get_int_box(leftmost_geotiff, gt_bbox, add_bump=add_bump)
            int_box_y = custom_get_int_box(topmost_geotiff, gt_bbox, add_bump=add_bump)
            
            patch_x_start = int_box_x[0][0]
            patch_x_end = int_box_x[1][0]
            patch_y_start = int_box_y[0][1]
            patch_y_end = int_box_y[1][1]

            if bbox_patch_sizes is not None and bbox_idx < len(bbox_patch_sizes):
                bbox_patch_size = bbox_patch_sizes[bbox_idx]
                if isinstance(bbox_patch_size, str):
                    bbox_patch_size = eval(bbox_patch_size)
            else:
                bbox_patch_size = None

            y_bbox = (patch_y_start, patch_y_end, (bbox[3], bbox[1]))
            patch_y_xs[y_bbox] = SortedList()
            if bbox_patch_size is not None:
                patch_y_xs[y_bbox].add((patch_x_start, patch_x_end, (bbox[0], bbox[2]), bbox_patch_size))
            else:
                patch_y_xs[y_bbox].add((patch_x_start, patch_x_end, (bbox[0], bbox[2])))

    for patch_y_idx, patch_y_coords in enumerate(patch_y_xs.keys()):
        patch_y_start, patch_y_end, *extra_data_y = patch_y_coords
        
        extra_data_y += max(0, 1 - len(extra_data_y)) * [None]
        orig_y_coords = extra_data_y[0]

        active_geotiffs = set()
        geotiffs_to_disable = []

        x_keys_pos = 0
        for patch_x_idx, patch_x_coords in enumerate(patch_y_xs[patch_y_coords]):
            patch_x_start, patch_x_end, *extra_data_x = patch_x_coords
            
            extra_data_x += max(0, 2 - len(extra_data_x)) * [None]
            orig_x_coords, patch_size = extra_data_x
            patch_size = patch_size or regular_patch_size

            if len(extra_data_x) >= 1:
                orig_x_coords = extra_data_x[0]
            else:
                orig_x_coords = None

            for geotiff_to_disable in geotiffs_to_disable:
                active_geotiffs.remove(geotiff_to_disable)
            
            geotiffs_to_disable.clear()

            # keep a list of currently active TIFFs that need to be disabled after the current patch

            if orig_x_coords is not None and len(orig_x_coords) == 2:
                patch_x_start_coords = min(orig_x_coords[0], orig_x_coords[1])
                patch_x_end_coords = max(orig_x_coords[0], orig_x_coords[1])
            else:
                coords_0 = leftmost_geotiff.get_coords(patch_x_start, patch_y_start)[0]
                coords_1 = leftmost_geotiff.get_coords(patch_x_end, patch_y_end)[0]
                patch_x_start_coords = min(coords_0, coords_1)
                patch_x_end_coords = max(coords_0, coords_1)
            
            if orig_y_coords is not None and len(orig_y_coords) == 2:
                patch_y_start_coords = max(orig_y_coords[0], orig_y_coords[1])
                patch_y_end_coords = min(orig_y_coords[0], orig_y_coords[1])
            else:
                coords_0 = topmost_geotiff.get_coords(patch_x_start, patch_y_start)[1]
                coords_1 = topmost_geotiff.get_coords(patch_x_end, patch_y_end)[1]
                patch_y_start_coords = max(coords_0, coords_1)
                patch_y_end_coords = min(coords_0, coords_1)

            while x_keys_pos < num_x_keys and x_keys[x_keys_pos] < patch_x_end:
                event = x_dict_pxs[x_keys_pos][1]
                event_type = event[0]
                event_gt = event[1]
                if event_type == 'start':
                    # check if this GT intersects with the active segment in the Y dimension
                    event_gt_top = max(event_gt.tif_bBox[0][1], event_gt.tif_bBox[1][1])
                    event_gt_btm = min(event_gt.tif_bBox[0][1], event_gt.tif_bBox[1][1])

                    # use < instead of <= for upper limits
                    if ((event_gt_btm            <= patch_y_start_coords < event_gt_top)
                        or (event_gt_btm         <= patch_y_end_coords   < event_gt_top)
                        or (patch_y_end_coords   <= event_gt_top         < patch_y_start_coords)
                        or (patch_y_end_coords   <= event_gt_btm         < patch_y_start_coords)):
                        active_geotiffs.add(event_gt)
                elif event_type == 'end' and event_gt in active_geotiffs and x_keys[x_keys_pos] < patch_x_start:
                    # last check is crucial: there could be other (bbox) patches that overlap with this one!
                    geotiffs_to_disable.append(event_gt)
                x_keys_pos += 1

            # + 1 not needed: end coords are exclusive
            patch_arr = np.zeros((patch_y_end - patch_y_start, patch_x_end - patch_x_start, num_channels))

            # paint this patch: create numpy array from zarrs of patches
            for gt in active_geotiffs:
                y_px_box = custom_get_int_box(topmost_geotiff, gt.tif_bBox, add_bump=add_bump)
                x_px_box = custom_get_int_box(leftmost_geotiff, gt.tif_bBox, add_bump=add_bump)

                px_box = ((x_px_box[0][0], y_px_box[0][1]), (x_px_box[1][0], y_px_box[1][1]))

                # determine intersection with current patch, then backproject to this geotiff's coord system
                px_box_inter = ((max(px_box[0][0], patch_x_start), max(px_box[0][1], patch_y_start)),
                                (min(px_box[1][0], patch_x_end), min(px_box[1][1], patch_y_end)))
                if px_box_inter[0][0] >= px_box_inter[1][0] or px_box_inter[0][1] >= px_box_inter[1][1]:
                    # no intersection
                    continue
                
                # always subtract the start, even from the end
                px_box_inter_backproj_global =\
                ((px_box_inter[0][0] - x_px_box[0][0], px_box_inter[1][0] - x_px_box[0][0]),
                 (px_box_inter[0][1] - y_px_box[0][1], px_box_inter[1][1] - y_px_box[0][1]))
                
                px_box_inter_backproj_local =\
                [((px_box_inter[0][0] - patch_x_start),
                  (px_box_inter[1][0] - patch_x_start)),
                 ((px_box_inter[0][1] - patch_y_start),
                  (px_box_inter[1][1] - patch_y_start))]

                # note: here, we use (y, x) instead of (x, y)

                pbibg = px_box_inter_backproj_global
                pbibl = px_box_inter_backproj_local
                patch_arr[pbibl[1][0]:pbibl[1][1], pbibl[0][0]:pbibl[0][1]] =\
                    gt.read()[pbibg[1][0]:pbibg[1][1], pbibg[0][0]:pbibg[0][1]]

            # create new file (for now, store each band as separate png)

            full_coord_box = ((patch_x_start_coords, patch_y_start_coords),
                              (patch_x_end_coords, patch_y_end_coords))

            os.makedirs(output_dir, exist_ok=True)
            
            output_naming_prefix = output_naming_prefix.strip()

            def get_fn(counter):
                nonlocal output_naming_prefix
                ctr = '' if counter <= 1 else f'_{counter}'
                if output_naming_scheme == 'prefix_only':
                    fn = f'{output_naming_prefix}{ctr}.{output_format}'
                else:
                    if output_naming_prefix not in {None, ''} and not output_naming_prefix[-1] == '_':
                        output_naming_prefix += '_'
                        
                    if output_naming_scheme == 'patch_pxs':
                        fn = f'{output_naming_prefix}y{patch_y_start}px_x{patch_x_start}px{ctr}.{output_format}'
                    elif output_naming_scheme == 'patch_idx':  # patch_idx
                        fn = f'{output_naming_prefix}y{"%05i" % patch_y_idx}_x{"%05i" % patch_x_idx}{ctr}.{output_format}'
                    elif output_naming_scheme == 'patch_coords':
                        fn = f'{output_naming_prefix}y{patch_y_start_coords}_x{patch_x_start_coords}{ctr}.{output_format}'
                    else:
                        raise NotImplementedError(f'Unknown naming scheme: "{output_naming_scheme}"')
                return fn

            counter = 0
            while True:
                counter += 1
                fn = get_fn(counter)
                output_path = os.path.join(output_dir, fn)
                if not os.path.isfile(output_path):
                    break

            print(f'*** Patch (y_idx: {patch_y_idx}, x_idx: {patch_x_idx} // ' +
                f'y: {patch_y_start}->{patch_y_end}, x: {patch_x_start}->{patch_x_end}) ***')
            print(f'Full coord box: {full_coord_box}')

            if len(active_geotiffs) == 0 and not keep_blanks:
                print('No GeoTIFF overlaps found; skipping blank patch.')
            else:
                img = Image.fromarray(patch_arr.astype(np.uint8))
                orig_img_width, orig_img_height = (img.width, img.height)
                    
                if patch_size is not None:
                    if len(patch_size) == 3 and patch_size[2]:
                        pas = patch_arr.shape
                        patch_size = (round(pas[1] * patch_size[0]), round(pas[0] * patch_size[1]))
                    img = img.resize(size=patch_size, resample=resizing_interpolation_filter)

                if create_tags:
                    img.tag_v2 = ImageFileDirectory_v2()
                    with Image.open(first_geotiff_path) as first_geotiff:
                        for to_copy in {1024, 1025, 2016, *range(2048, 2062), *range(3072, 3096), *range(4096, 5000),
                                        33550, 33922, 34735, 34736, 34737}:
                            if to_copy in first_geotiff.tag_v2:
                                img.tag_v2[to_copy] = first_geotiff.tag_v2[to_copy]
                    
                    # change ModelTiepointTag; i, j, k, x, y, z
                    img.tag_v2[33922] = (0.0, 0.0, 0.0, patch_x_start_coords, patch_y_start_coords, 0.0)
                    
                    if patch_size is not None:  # else assume ModelPixelScaleTag stays the same
                        pst = img.tag_v2[33550]
                        img.tag_v2[33550] = (pst[0] * (orig_img_width / patch_size[0]),
                                             pst[1] * (orig_img_height / patch_size[1]),
                                             pst[2])
                    
                    img.save(output_path, tiffinfo=img.tag_v2)
                else:
                    img.save(output_path)
            output_paths.append(output_path)
    
    return output_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', help='Input director(y|ies)', type=str, action='append', default=None)
    parser.add_argument('-o', '--output-dir', help='Output directory', type=str, default=None)
    parser.add_argument('-W', '--patch-width', help='Width of output patches (in px)', type=int, default=0)
    parser.add_argument('-H', '--patch-height', help='Height of output patches (in px)', type=int, default=0)
    parser.add_argument('-f', '--output-format', help='Format of the output', type=str, choices=['png', 'tiff'])
    parser.add_argument('-T', '--skip-tagging', help='Whether to skip tagging of created TIFF patches.',
                        action='store_true')
    parser.add_argument('-F', '--skip-fractional', help='Discard patches that are not covered by the input images '\
                        'completely', action='store_true')
    parser.add_argument('-n', '--output-naming-scheme', help='Naming scheme to use for output files ("patch_idx": '\
                        'horizontal/vertical index of patch; "patch_pxs": position of top-left corner, in pixels; '\
                        '"patch_coords": coordinates of top-left corner, in the CRS used in the input TIFFs).',
                        choices=['patch_idx', 'patch_pxs', 'patch_coords'], default='patch_coords')
    parser.add_argument('-p', '--output-naming-prefix', help='Prefix to use for output files.', type=str,
                        default='output')
    parser.add_argument('-b', '--bbox', help='Bounding box(es) to cut instead of using a uniform patchification '\
                                             '(format: (LON_WEST, LAT_SOUTH, LON_EAST, LAT_NORTH)).',
                        type=str, action='append', default=None)
    parser.add_argument('-B', '--skip-blanks', help='Whether to skip patches for which no overlap with an input image '\
                        'exists.', action='store_true')
    parser.add_argument('--regular-patch-size', help='(width, height) to resize regular patches to. Omit to preserve '\
                                                     'original size. Use (scale_x, scale_y, True) to specify rescaling '\
                                                     'ratios rather than absolute sizes.', type=str, default=None)
    parser.add_argument('--bbox-patch-sizes', help='(width, height) to resize bbox patches to. Supply one pair per '\
                                                   'bbox. Omit to preserve original size. Use (scale_x, scale_y, True) '\
                                                   'to specify rescaling ratios rather than absolute sizes.',
                        type=str, action='append', default=None)
    parser.add_argument('--resizing-interpolation-method', help='Interpolation method to use when resizing patches. '\
                                                                'Choose between: bilinear, bicubic, nearest_neighbor.',
                        type=str, choices=['nearest', 'nearest_neighbor', 'bilinear', 'bicubic'], default='bicubic')

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f'supremap_patchification_{int(time.time())}'

    patchify(args.input_dir, args.output_dir, args.patch_width, args.patch_height, args.output_format,
             not args.skip_tagging, not args.skip_fractional, not args.skip_blanks, args.bbox, args.regular_patch_size,
             args.bbox_patch_sizes, args.resizing_interpolation_method, args.output_naming_scheme,
             args.output_naming_prefix)
