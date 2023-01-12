import argparse
import os
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2


def transfer_geotiff_tag(input_path, reference_path, output_path=None):
    if output_path in {None, ''}:
        ipl =  input_path.lower()
        output_path = (input_path if ipl.endswith('.tiff') or ipl.endswith('.tif')
                       else os.path.splitext(input_path)[0] + '.tiff')

    with Image.open(input_path) as input_img:
        input_img.tag_v2 = ImageFileDirectory_v2()
        with Image.open(reference_path) as reference_img:
            for to_copy in {1024, 1025, 2016, *range(2048, 2062), *range(3072, 3096), *range(4096, 5000),
                            33550, 33922, 34735, 34736, 34737}:
                if to_copy in reference_img.tag_v2:
                    input_img.tag_v2[to_copy] = reference_img.tag_v2[to_copy]
                        
            # add missing ModelTiepoint
            if 33922 not in input_img.tag_v2:
                input_img.tag_v2[33922] = (0.0, 0.0, 0.0)
            
            # adapt ModelPixelScaleTag
            if input_img.width != reference_img.width or input_img.height != reference_img.height:
                pst = input_img.tag_v2[33550]
                input_img.tag_v2[33550] = (pst[0] * (reference_img.width / input_img.width),
                                           pst[1] * (reference_img.height / input_img.height),
                                           pst[2])
            
        input_img.save(output_path, tiffinfo=input_img.tag_v2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', help='File to tag.', type=str)
    parser.add_argument('--reference-path', '-r', help='File with GeoTIFF data.', type=str)
    parser.add_argument('--output-path', '-o', help='Path under which to store output GeoTIFF. If omitted, will use '\
                                                    'the path in "input-file".', type=str, default=None)


    args = parser.parse_args()
    transfer_geotiff_tag(args.input_path, args.reference_path, args.output_path)
