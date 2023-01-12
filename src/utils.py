import geotiff
import json


class RoundingFloat(float):
    # set floating number precision for geojson dumps
    __repr__ = staticmethod(lambda x: format(x, '.10f'))

json.encoder.c_make_encoder = None
if hasattr(json.encoder, 'FLOAT_REPR'):
    # Python 2
    json.encoder.FLOAT_REPR = RoundingFloat.__repr__
else:
    # Python 3
    json.encoder.float = RoundingFloat


def dummy_func():
    json.encoder


def custom_get_int_box(gt_obj, bBox, outer_points=None, add_bump=False):
    # adapted copy of the geotiff.GeoTiff.custom_get_int_box function,
    # extended to allow manual toggling of the "bump" that gets added

    # all 4 corners of the box
    left_top = bBox[0]
    right_bottom = bBox[1]
    left_bottom = (bBox[0][0], bBox[1][1])
    right_top = (bBox[1][0], bBox[0][1])

    left_top_c = gt_obj._convert_coords(gt_obj.as_crs, gt_obj.crs_code, left_top)
    right_bottom_c = gt_obj._convert_coords(gt_obj.as_crs, gt_obj.crs_code, right_bottom)
    left_bottom_c = gt_obj._convert_coords(gt_obj.as_crs, gt_obj.crs_code, left_bottom)
    right_top_c = gt_obj._convert_coords(gt_obj.as_crs, gt_obj.crs_code, right_top)

    all_x = [left_top_c[0], left_bottom_c[0], right_bottom_c[0], right_top_c[0]]
    all_y = [left_top_c[1], left_bottom_c[1], right_bottom_c[1], right_top_c[1]]

    # get the outer ints based on the maxs and mins
    x_min = min(all_x)
    y_min = min(all_y)
    x_max = max(all_x)
    y_max = max(all_y)

    # convert to int
    i_bump = int(not gt_obj.tif_bBox[0][0] == x_min and add_bump)
    i_min = gt_obj._get_x_int(x_min) + i_bump
    
    j_bump = int(not gt_obj.tif_bBox[0][1] == y_max and add_bump)
    j_min = gt_obj._get_y_int(y_max) + j_bump
    i_max = gt_obj._get_x_int(x_max)
    j_max = gt_obj._get_y_int(y_min)

    if outer_points:
        i_min_out: int = i_min - int(outer_points)
        j_min_out: int = j_min - int(outer_points)
        i_max_out: int = i_max + int(outer_points)
        j_max_out: int = j_max + int(outer_points)
        height = gt_obj.tif_shape[0]
        width = gt_obj.tif_shape[1]
        if i_min_out < 0 or j_min_out < 0 or i_max_out > width or j_max_out > height:
            raise geotiff.BoundaryNotInTifError("area_box is too close to the TIF edge; cannot get the outer points")
        return ((i_min_out, j_min_out), (i_max_out, j_max_out))

    return ((i_min, j_min), (i_max, j_max))
