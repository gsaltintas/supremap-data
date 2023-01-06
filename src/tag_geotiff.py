import argparse
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2


ModelPixelScaleTag = 33550
ModelTiepointTag = 33922
GeoKeyDirectoryTag = 34735
GeoDoubleParamsTag = 34736
GeoAsciiParamsTag = 34737

GTModelTypeGeoKey = 1024
ModelTypeProjected = 1

GTRasterTypeGeoKey = 1025
RasterPixelIsArea = 1

GTCitationGeoKey = 1026
GeogCitationGeoKey = 2049

GeogAngularUnitsGeoKey = 2054
GeogAngularUnitsAngularDegree = 9102

ProjectedCSTypeGeoKey = 3072


def create_geotiff_metadata(pixel_x_y, pixel_coords_x_y, pixel_scale_x_y, crs_code, crs_geo_citation, crs_gt_citation):
    def p(obj):
        if isinstance(obj, list):
            ret = []
            for el in obj:
                ret.append(p(el))
            return ret
        else:
            return obj

    ifd = ImageFileDirectory_v2()
    geokey_dir = p([1, 1, 2, 0])
    num_keys = 0
    
    geokey_dir += p([GTModelTypeGeoKey, 0, 1, ModelTypeProjected])
    num_keys += 1

    geokey_dir += p([GTRasterTypeGeoKey, 0, 1, RasterPixelIsArea])
    num_keys += 1

    ascii_data = ''
    ascii_data_len = 0

    # Remember to add keys in lexicographical order!

    if crs_gt_citation not in [None, '']:
        geokey_dir += p([GTCitationGeoKey, GeoAsciiParamsTag, len(crs_gt_citation), ascii_data_len])
        num_keys += 1
        ascii_data_len += len(crs_gt_citation) + 1
        ascii_data += crs_gt_citation + '|'

    if crs_geo_citation not in [None, '']:
        geokey_dir += p([GeogCitationGeoKey, GeoAsciiParamsTag, len(crs_geo_citation), ascii_data_len])
        num_keys += 1
        ascii_data_len += len(crs_geo_citation) + 1
        ascii_data += crs_geo_citation + '|'
    
    geokey_dir += p([GeogAngularUnitsGeoKey, 0, 1, GeogAngularUnitsAngularDegree])
    num_keys += 1

    geokey_dir += p([ProjectedCSTypeGeoKey, 0, 1, crs_code])
    num_keys += 1

    geokey_dir[3] = num_keys

    ifd[GeoKeyDirectoryTag] = tuple(geokey_dir)
    ifd[GeoAsciiParamsTag] = ascii_data

    ifd[ModelPixelScaleTag] = (float(pixel_scale_x_y[0]), float(pixel_scale_x_y[1]), 0.0)
    ifd[ModelTiepointTag] = (float(pixel_x_y[0]), float(pixel_x_y[1]), 0.0,
                             float(pixel_coords_x_y[0]), float(pixel_coords_x_y[1]), 0.0)

    return ifd


if __name__ == '__main__':
    # For testing purposes:

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', help='File to tag.', type=str)
    parser.add_argument('--pixel-px-x', '-x', help='X position of reference pixel.', type=float, default=0.0)
    parser.add_argument('--pixel-px-y', '-y', help='Y position of reference pixel.', type=float, default=0.0)
    parser.add_argument('--pixel-scale-x', '-v', help='Pixel scale in X direction.', type=float)
    parser.add_argument('--pixel-scale-y', '-w', help='Pixel scale in Y direction.', type=float)
    parser.add_argument('--crs-code', '-c', help='CRS code to use for GeoTIFF.', type=int)
    parser.add_argument('--crs-geo-citation', '-g', help='CRS geo citation.', type=str, default=None)
    parser.add_argument('--crs-gt-citation', '-t', help='CRS GT citation.', type=str, default=None)
    parser.add_argument('--pixel-coord-x', '-X', help='X coordinate (e.g. longitude) of location at reference pixel.',
                        type=float)
    parser.add_argument('--pixel-coord-y', '-Y', help='Y coordinate (e.g. latitude) of location at reference pixel.',
                        type=float)
    parser.add_argument('--output-path', '-p', help='Path under which to store output GeoTIFF.',
                        type=str)

    args = parser.parse_args()

    with Image.open(args.input_file) as img:
        info = create_geotiff_metadata(pixel_x_y=(args.pixel_px_x, args.pixel_px_y),
                                       pixel_coords_x_y=(args.pixel_coord_x, args.pixel_coord_y),
                                       pixel_scale_x_y=(args.pixel_scale_x, args.pixel_scale_y), crs_code=args.crs_code,
                                       crs_geo_citation=args.crs_geo_citation, crs_gt_citation=args.crs_gt_citation)
        img.save(args.output_path, tiffinfo=info)
