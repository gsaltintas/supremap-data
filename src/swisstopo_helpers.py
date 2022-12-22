import datetime
import json

import pandas as pd
import requests
from tabulate import tabulate

VERSION = '0.9'
BASE_URL = f'https://data.geo.admin.ch/api/stac/v{VERSION}'
SWISSTOPO_URL = f'{BASE_URL}/collections/ch.swisstopo.swissimage-dop10'

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
    print(tabulate(collections[['id', 'title']]))
    del response

def get_url(args) -> str:
    """ constructs query url based on passed arguments """
    url = SWISSTOPO_URL
    items = False
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
        url += f'datetime={dates[0]}'
        if len(dates) == 2:
            url += f'/{dates[1]}'
    return url

def download_tifs(url: str, args):
    """ downloads tifs based on the search specified by the url  into the args.save_dir"""
    response = send_request(url)
    features = pd.DataFrame(response['features'])
    if len(features) == 0:
        exit('No items found')
    features.drop(['collection', 'type', 'stac_version'], axis=1, inplace=True)
    # returned object dict_keys(['id', 'collection', 'type', 'stac_version', 'geometry', 'bbox', 'properties', 'links', 'assets'])
    features['tif'] = features['id'].map(lambda x: f'{x}_{args.resolution}_2056.tif')
    features['link'] = features.apply(lambda x: x['assets'][x['tif']]['href'], axis=1)
    print(features.loc[0, 'link'])
    args.save_dir.mkdir(exist_ok=True, parents=True)

    def download_file(row):
        """Parses dataframe row, downloads files to the save_dir

        Args:
            row (_type_): _description_
        """
        # dt = datetime.datetime.strptime(row['properties']['datetime'], '%y-%m-%d')
        dt = row['properties']['datetime'][:10]
        bbox = '-'.join([f'{b:.2f}' for b in row['bbox']])
        link = row['link']
        local_name = args.save_dir.joinpath(f"{row['id']}_{bbox}_{dt}")

        r = requests.get(link)
        f = open(local_name, 'wb')
        for chunk in r.iter_content(chunk_size=512 * 1024):
            if chunk:
                f.write(chunk)
        f.close()

    features.apply(download_file, axis=1)
        
    with open('downloaded.csv', 'a') as f:
        features['link'].to_csv('downloaded.csv', index=False, header=False, mode='a')
    if response['links'][-1]['rel'] == 'next':
        url  = response['links'][-1]['href']
        print('next page')
        download_tifs(url, args)
