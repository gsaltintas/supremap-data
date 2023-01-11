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