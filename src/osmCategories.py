FieldTags = {
    'landuse': ['allotments', 'brownfield', 'cemetery', 'greenhouse_horticulture', 'farmland', 'flowerbew',
                'forest', 'grass', 'landfill,' 'meadow', 'orchard', 'plant_nursery', 'recreation_ground', 'village_green', 'vineyard'],
}

WaterTags = {
    'natural': ['water'],
    # including waterways breaks the layering with streets :(
    # # 'waterway': ['basin', 'canal', 'ditch', 'drain', 'fairway', 'fish_pass', 'lagoon', 'lake', 'lock', 'moat', 'oxbow', 'pond', 'reflecting_pool', 'reservoir', 'river', 'riverbank', 'stream', 'stream_pool', 'tidal_channel'],
    # 'landuse': ['aquaculture', 'salt_pond']
}

BeachTags = {
    'natural': ['beach', 'cape', 'hot_spring', 'reef'],
}

BuildingTags = {
    'building': True
}

RailwayTags = {
    'railway': ['disused', 'funicular', 'light_rail', 'miniature', 'monorail']
}
