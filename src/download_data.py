import argparse
import json
from pathlib import Path

import sentinel as se
import swisstopo_helpers as sw

"""_summary_
More information about the API can be found at https://www.geo.admin.ch/de/geo-dienstleistungen/geodienste/downloadienste/stac-api.html
for longitude, latitudes of specific cities see https://www.mapsofworld.com/lat_long/switzerland-lat-long.html
zurich: 
"""

parser=argparse.ArgumentParser(description='Swisstopo Data Donwload Engine')

parser.add_argument('-l', '--list', help='List all possible collection Ids', action='store_true')
parser.add_argument('--swisstopo', help='Download from Swisstopo', action='store_true')
parser.add_argument('--sentinel', help='Download from sentinel', action='store_true')
parser.add_argument('-b', '--bbox', help='Search by bounding box, enter in the following format: \"[LONGITUDE_WEST, LAT_SOUTH, LONG_EAST, LAT_NORTH]\"', type=json.loads, default=[5,46,10,48])
parser.add_argument('-d', '--date_range', help='Specify a date range in the following format: \"[START]\" or \"[START, END\", eg: \"[\\"2018-01-01\\", \\"2018-12-31\\"]\"', default=["2022-01-01", "2022-12-05"], type=json.loads)
parser.add_argument('--save_dir', default='../out', help='Target directory',  type=lambda x: Path(x).resolve().absolute())
parser.add_argument('-r', '--resolution', help='Ground sampling distance, available: 0.1 or 2', type=float, default=0.1)
parser.add_argument('--rows', help='Rows to query for in the Sentinel API, max 100', type=int, default=20)
parser.add_argument('--order_by_position', help='Sorts the results based on their position, by default this is turned off and sorting is done absed on the ingestion date', action='store_true')
parser.add_argument('--desc', help='Sorts the results in the descending order if provided', action='store_true')
    
args = parser.parse_args()

if args.swisstopo:
    if args.list:
        sw.list_collections()
    url = sw.get_url(args)
    sw.download_tifs(url, args)

if args.sentinel:
    url = se.get_url(args)
    se.download_zips(url, args)
