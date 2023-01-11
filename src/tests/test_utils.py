def test_json_precision():
    import json

    from utils import dummy_func

    x = 0.12
    res = json.dumps(x)
    assert res == f'{x:.10f}'
