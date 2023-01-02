import datetime
import json
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import requests
import xmltodict
from sys import exit
from tabulate import tabulate

VERSION = '0.9'
BASE_URL = f'https://apihub.copernicus.eu/apihub'
import os

pwd = os.environ['DHUS_PASSWORD'] 
user = os.environ['DHUS_USER'] 

def send_request(url, a, b):
    """ 
    sends a get request to utl and checks its status, 
        raises HTTPError if the request is unsuccessful
    """
    session = requests.Session()
    session.auth = user, pwd
    response = session.get(url)
    response.raise_for_status()
    decoded_response = response.content.decode('utf-8')
    response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
    return response_json

def get_url(args, start_row:int = 0) -> str:
    """ constructs query url based on passed arguments """
    #  https://apihub.copernicus.eu/apihub/search?q= platformname:Sentinel-2 AND beginPosition:[2022-01-01T06:00:00.000Z TO 2022-01-05T06:00:00.000Z] AND ( footprint:"Intersects(POLYGON((5.0000000000000 46.0000000000000,10.0000000000000 46.0000000000000,10.0000000000000 48.0000000000000,5.0000000000000 48.0000000000000,5.0000000000000 46.0000000000000 )))")&rows=25&start=0
    # url = f'{BASE_URL}/search?q= platformname:Sentinel-2 '
    url = f'{BASE_URL}/search?start={start_row}&rows={args.rows}&q= platformname:Sentinel-2 '
    items = False
    if args.bbox:
        bbox = args.bbox
        if len(bbox) != 4:
            raise ValueError(f'Bounding box argument must have exactly 4 elements in the following format: \"[LONGITUDE_WEST, LAT_SOUTH, LONG_EAST, LAT_NORTH]\".')

        items = True
        bbox_polygon = f'{bbox[2]} {bbox[1]}, {bbox[0]} {bbox[1]}, {bbox[0]} {bbox[3]}, {bbox[2]} {bbox[3]}, {bbox[2]} {bbox[1]}'

        url += f'AND ( footprint:"Intersects(POLYGON(( {bbox_polygon} )))" ) '
        # daterange: datetime=2018-01-01/2018-12-31
    if args.date_range:
        dates = args.date_range
        url += '/items?' if not items else '&'
        url += f' AND beginPosition: [{datetime.datetime.fromisoformat(dates[0]).isoformat()} '
        # url += f' AND ingestiondate: [{datetime.datetime.fromisoformat(dates[0]).isoformat()} '
        if len(dates) == 2:
            url += f' TO {datetime.datetime.fromisoformat(dates[1]).isoformat()}] '
        else:
            url += ' NOW] '
    if args.order_by_position:
        url += '&orderby=beginposition'
    else:
        url += '&orderby=ingestiondate'
    url += ' desc' if args.desc else ' asc'
    return url

def download_zips(url: str, args, start_row:int =0):
    """ downloads tifs based on the search specified by the url  into args.save_dir"""
    response_json = send_request(url, user, pwd)
    print(url)
    if 'entry' not in response_json['feed']:
        exit(f'No {"more" if start_row > 0 else ""} items found')
        
    # opensearch:itemsPerPage
    # title = response_json['feed']['entry'][0]['link']
    features = pd.DataFrame(response_json['feed']['entry'])
    # returned columns: Index(['title', 'link', 'id', 'summary', 'ondemand', 'date', 'int', 'double', 'str'],
    if len(features) == 0:
        exit('No items found')
    features.drop(['ondemand', 'int'], axis=1, inplace=True)
    features['links'] = features.apply(lambda x: x['link'][0]['@href'], axis=1)
    features['alternative_link'] = features.apply(lambda x: x['link'][1]['@href'], axis=1)
    features['icon_link'] = features.apply(lambda x: x['link'][2]['@href'], axis=1)
    args.save_dir.mkdir(exist_ok=True, parents=True)

    features['bbox'] = features['str'].apply(lambda x: x[1]['#text'])
    features['bbox'] = features['bbox'].apply(lambda x: x.split('gml:coordinates>')[1][:-2].split(','))
    def download_file(row):
        """
        Parses dataframe row, downloads files to the save_dir
        """
        # dt = datetime.datetime.strptime(row['properties']['datetime'], '%y-%m-%d')
        # dt = row['properties']['datetime'][:10]
        # bbox = '-'.join([f'{b:.2f}' for b in row['bbox']])
        link = row['links']
        # local_name = args.save_dir.joinpath(f"{row['id']}_{bbox}_{dt}")
        # local_name = args.save_dir.joinpath(f"{row['id']}").with_suffix('.zip')
        local_name = args.save_dir.joinpath(f"{row['title']}").with_suffix('.zip')
        try:
            session = requests.Session()
            session.auth = user, pwd
            r = session.get(link, stream=True)
            print(row['link'])
            r.raise_for_status()
            print(r.status_code)
            f = open(local_name, 'wb')
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
            f.close()
            unzip_sentinel(path_to_zip=local_name, out_path=args.save_dir)
        except  requests.exceptions.RequestException as e:
            warnings.warn(f'Problem with current request: {e}')
        except Exception as e:
            warnings.warn(f'Problem: {e}')

    features.apply(download_file, axis=1)
        
    with open('sentinel_downloaded.csv', 'a') as f:
        features['link'].to_csv(args.save_dir.joinpath('sentinel_downloaded.csv'), index=False, header=False, mode='a')
    print(f'Done with the range {start_row}-{start_row+args.rows-1}. ')
    url = get_url(args, start_row=start_row+args.rows)
    download_zips(url, args, start_row=start_row+args.rows)

def unzip_sentinel(path_to_zip, out_path=None):
    zip_ref = zipfile.ZipFile(path_to_zip, 'r')
    if out_path is None:
        out_path = Path(path_to_zip)
    zip_ref.extractall(out_path)
    zip_ref.close()