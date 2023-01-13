import datetime
import json
from pathlib import Path

import geopandas as gp
import pandas as pd
import requests
from geojson import Feature, GeoJSON, Polygon
from tabulate import tabulate

from utils import dummy_func

dummy_func()

VERSION = '0.9'
BASE_URL = f'https://data.geo.admin.ch/api/stac/v{VERSION}'
SWISSTOPO_URL = f'{BASE_URL}/collections/ch.swisstopo.swissimage-dop10'
def format_date_str(date_str: str):
    # must be in ISO8601 format, see https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/FullTextSearch?redirectedfrom=SciHubUserGuide.3FullTextSearch
    return datetime.datetime.fromisoformat(date_str).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def get_next_response(response, key='next'):

    for d in response['links']:
        if d['rel'] == key:
            return d['href']
    return None

def send_request(url):
    """ 
    sends a get request to utl and checks its status, 
        raises HTTPError if the request is unsuccessful
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def list_collections():

    response = send_request(f'{BASE_URL}/collections')
    collections = pd.DataFrame(response['collections'])
    while get_next_response(response):
        response = send_request(get_next_response(response))
        collections_ = pd.DataFrame(response['collections'])
        collections=collections.append(collections_)
    print(tabulate(collections[['id', 'title']].reset_index(drop=True)))
    del response

def get_url(args) -> str:
    """ constructs query url based on passed arguments """
    url = SWISSTOPO_URL
    items = False
    if args.id:
        # each tile seems to be appearing only once per year, so might be okay
        items = True
        url += f'/items/{args.id}'
        return url
    if args.bbox:
        bbox = args.bbox
        if len(bbox) != 4:
            raise ValueError(f'Bounding box argument must have exactly 4 elements in the following format: \"[LONGITUDE_WEST, LAT_SOUTH, LONG_EAST, LAT_NORTH]\".')
        # https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissimage-dop10/items?coordinates=8.55,47.37  
        # todo check if the bbounding box is in switzerland
        items = True
        url += f'/items?bbox={",".join(map(str, bbox))}' 
        # daterange: datetime=2018-01-01/2018-12-31
    if args.date_range:
        dates = args.date_range
        url += '/items?' if not items else '&'
        url += f'datetime={format_date_str(dates[0])}'
        if len(dates) == 2:
            url += f'/{format_date_str(dates[1])}'
    return url


def download_tifs(url: str, args):
    """ downloads tifs based on the search specified by the url  into the args.save_dir"""
    response = send_request(url)
    if response['type'] == 'FeatureCollection':
        features = pd.DataFrame(response['features'])
    elif response['type'] == 'Feature':
        # handle single item cases
        features = pd.DataFrame([response])
    else:
        raise Exception(f'Unkown type: {response["type"]}')
    if len(features) == 0:
        print('No items found')
        return
    features.drop(['collection', 'type', 'stac_version'], axis=1, inplace=True)
    # returned object dict_keys(['id', 'collection', 'type', 'stac_version', 'geometry', 'bbox', 'properties', 'links', 'assets'])

    args.save_dir.mkdir(exist_ok=True, parents=True)

    #  # save geometry information per download
    # for index, geom in enumerate(features["geometry"]):
    #     listString = json.dumps(geom["coordinates"][0])
        
    #     local_name = args.save_dir.joinpath(f"{features['id'][index]}.json")
    #     jsonFile = open(local_name, "w")
    #     jsonFile.write(listString)
    #     jsonFile.close()


    # features = 
    features['tif'] = features['id'].map(lambda x: f'{x}_{args.resolution}_2056.tif')
    features['link'] = features.apply(lambda x: x['assets'][x['tif']]['href'], axis=1)
    print(features.loc[0, 'link'])

    def download_file(row):
        """Parses dataframe row, downloads files to the save_dir

        Args:
            row (_type_): _description_
        """
        # dt = datetime.datetime.strptime(row['properties']['datetime'], '%y-%m-%d')
        dt = row['properties']['datetime'][:10]
        bbox = '-'.join([f'{b:.2f}' for b in row['bbox']])
        link = row['link']
        local_name = args.save_dir.joinpath(f"{row['id']}_{bbox}_{dt}.tif")
        assets = row['assets'][row['tif']]

        if local_name.exists():
            print(f'Found file {local_name}, skipping download.')
        else:
            r = requests.get(link)
            f = open(local_name, 'wb')
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
            f.close()

        # save file's geojson data
        geo_local_name = args.save_dir.joinpath(f"{row['id']}.geojson")
        if geo_local_name.exists():
            print(f'Found geojson file {geo_local_name}, skipping write.')
        else:
            mask_name = args.save_dir.joinpath(f"{row['id']}_segmentation.png").as_posix()
            geo = Feature(id = row['id'], geometry = row['geometry'], properties={'datetime':row['properties']['datetime'], 'proj:epsg':assets['proj:epsg'], 'checksum:multihash':assets['checksum:multihash'], 'eo:gsd': assets['eo:gsd'], 'tif': local_name.as_posix(), 'mask': mask_name})
            print(row['geometry'])
            jsonFile = open(geo_local_name, "w")
            
            json.dump(geo, jsonFile, indent=4)
            jsonFile.close()

    features = features.truncate(after=max(0, args.max_rows-1))

    features.apply(download_file, axis=1)

    args.max_rows = max(0, args.max_rows - len(features))
        
    while args.max_rows > 0  and get_next_response(response):
        url = get_next_response(response)
        print('next page')
        download_tifs(url, args)

def download_csvs(args):
    # sample link in the csv (exported from https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html#technische_details): 
    # https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_2021_2629-1167/swissimage-dop10_2021_2629-1167_0.1_2056.tif
    for path in args.csv_paths:
        path = Path(path)
        # target: https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissimage-dop10/items/swissimage-dop10_2021_2629-1167
        links = pd.read_csv(path, header=None)
        links['id'] = links[0].apply(lambda s: s.split('/')[4])
        # brute force for now
        def download_one(row):
            args.id = row['id']
            url = get_url(args)
            print(url)
            download_tifs(url, args)
        links.apply(download_one, axis=1)