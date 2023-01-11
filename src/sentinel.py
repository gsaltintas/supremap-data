import datetime
import json
import time
import warnings
import zipfile
from pathlib import Path
from sys import exit

import numpy as np
import pandas as pd
import requests
import xmltodict
from geojson import Feature, GeoJSON, Polygon
from shapely import wkt
from tabulate import tabulate

from utils import dummy_func

dummy_func()

VERSION = '0.9'
BASE_URL = f'https://apihub.copernicus.eu/apihub'
# BASE_URL = 'https://scihub.copernicus.eu/dhus'
# BASE_URL = 'https://scihub.copernicus.eu/dhus/odata/v1'
import os

pwd = os.environ['DHUS_PASSWORD'] 
user = os.environ['DHUS_USER'] 
def format_datestr_for_sentinel(date_str: str):
    # must be in ISO8601 format, see https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/FullTextSearch?redirectedfrom=SciHubUserGuide.3FullTextSearch
    return datetime.datetime.fromisoformat(date_str).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def send_request(url, user: str = None, pwd:str = None):
    """ 
    sends a get request to utl and checks its status, 
        raises HTTPError if the request is unsuccessful
    """
    session = requests.Session()
    if user is not None and pwd is not None:
        session.auth = user, pwd
    response = session.get(url)
    response.raise_for_status()
    decoded_response = response.content.decode('utf-8')
    response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
    return response_json


def get_url(args, start_row:int = 0) -> str:
    """ constructs query url based on passed arguments """
    # url = f'{BASE_URL}/search?q= platformname:Sentinel-2 '
    # 'cloudcoverpercentage:["0" TO "30"]'
    # https://apihub.copernicus.eu/apihub/search?start=60&rows=20&q= platformname:Sentinel-2 AND processinglevel:Level-2A AND ( footprint:"Intersects(POLYGON(( 8.55180522978693 47.3661168914808, 8.535352210741475 47.3661168914808, 8.535352210741475 47.378, 8.55180522978693 47.378, 8.55180522978693 47.3661168914808 )))" ) & AND beginPosition: [2022-01-01T00:00:00  TO 2022-12-05T00:00:00]  AND (cloudcoverpercentage: [0 TO 2]) &orderby=ingestiondate asc

    # correct full url
    # https://scihub.copernicus.eu/dhus/search?start=0&rows=20&q= platformname:Sentinel-2 AND beginposition: [2022-01-01T00:00:00.000Z  TO 2022-12-05T00:00:00.000Z] AND cloudcoverpercentage:[0 TO 2] AND processinglevel:Level-2A AND  footprint:"Intersects(POLYGON(( 8.55180522978693 47.3661168914808, 8.535352210741475 47.3661168914808, 8.535352210741475 47.378, 8.55180522978693 47.378, 8.55180522978693 47.3661168914808 )))"   &orderby=ingestiondate asc
    url = f'{BASE_URL}/search?start={start_row}&rows={args.rows}' +\
          f'&q=platformname:Sentinel-2 AND (cloudcoverpercentage:[0 TO {args.max_cloud_coverage_pct}])'
    if args.processing_level not in {None, ''}:
        url += f' AND processinglevel:"{args.processing_level}"'

    if args.date_range:
        dates = args.date_range
        # date must be of ISO8601 format
        url += f' AND beginposition: [{format_datestr_for_sentinel(dates[0])} '
        # url += f' AND ingestiondate: [{format_datestr_for_sentinel(dates[0])} '
        if len(dates) == 2:
            url += f' TO {format_datestr_for_sentinel(dates[1])}] '
        else:
            url += ' TO NOW] '
    if args.bbox:
        bbox = args.bbox
        if len(bbox) != 4:
            raise ValueError(f'Bounding box argument must have exactly 4 elements in the following format: \"[LONGITUDE_WEST, LAT_SOUTH, LONG_EAST, LAT_NORTH]\".')

        bbox_polygon = f'{bbox[2]} {bbox[1]}, {bbox[0]} {bbox[1]}, {bbox[0]} {bbox[3]}, {bbox[2]} {bbox[3]}, {bbox[2]} {bbox[1]}'

        url += f'AND ( footprint:"Intersects(POLYGON(( {bbox_polygon} )))" ) '
    if args.order_by_position:
        url += '&orderby=beginposition'
    else:
        url += '&orderby=ingestiondate'
    url += ' desc' if args.desc else ' asc'
    return url

def extract_key_from_collection(x, key):
    """ parses the xml lists/dicts from sentinel response to get the corresponding key """
    if isinstance(x, dict):
        if x.get('@name') == key:
            return x['#text']
    elif isinstance(x, list):
        for field_dct in x:
            if field_dct.get('@name') == key:
                return field_dct['#text']
    return np.nan

def download_zips(url: str, args, start_row:int =0):
    """ downloads tifs based on the search specified by the url  into args.save_dir"""
    response_json = send_request(url, user, pwd)
    print(url)
    if 'entry' not in response_json['feed']:
        print(f'No{" more" if start_row > 0 else ""} items found')
        return []

    # opensearch:itemsPerPage
    # title = response_json['feed']['entry'][0]['link']
    features = pd.DataFrame(response_json['feed']['entry'])
    # returned columns: Index(['title', 'link', 'id', 'summary', 'ondemand', 'date', 'int', 'double', 'str'],
    if len(features) == 0:
        print('No items found')
        return []
    
    features.drop(['ondemand', 'int'], axis=1, inplace=True)
    features['links'] = features.apply(lambda x: x['link'][0]['@href'], axis=1)
    features['alternative_link'] = features.apply(lambda x: x['link'][1]['@href'], axis=1)
    features['icon_link'] = features.apply(lambda x: x['link'][2]['@href'], axis=1)
    args.save_dir.mkdir(exist_ok=True, parents=True)

    
    # features['bbox'] = features['str'].apply(lambda x: extract_key_from_collection(x, 'gmlfootprint')).dropna()#lambda x: x[1]['#text'])
    features['bbox'] = features['str'].apply(lambda x: extract_key_from_collection(x, 'footprint')).dropna()#lambda x: x[1]['#text'])
    # features['bbox'] = features['bbox'].apply(lambda x: x.split('gml:coordinates>')[1][:-2].split(','))
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
        unzipped_local_name = local_name.with_suffix('.SAFE')
        if local_name.exists() or local_name.with_suffix('.SAFE').exists():
            print(f'{local_name.stem} exists, skipping download. ')
            # return
        else:
            try:
                session = requests.Session()
                session.auth = user, pwd
                r = session.get(link, stream=True)
                print(f'Attempting download for {row["link"]}')
                r.raise_for_status()
                print(r.status_code)
                f = open(local_name, 'wb')
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        f.write(chunk)
                f.close()
            except  requests.exceptions.RequestException as e:
                warnings.warn(f'Problem with current request: {e}')
            except Exception as e:
                warnings.warn(f'Problem: {e}')

        if not local_name.with_suffix('.SAFE').exists():
            unzip_sentinel(path_to_zip=local_name, out_path=args.save_dir)
        if local_name.exists():
            local_name.unlink()

        # save file's geojson data
        geo_local_name = args.save_dir.joinpath(f"{row['title']}.geojson")
        if geo_local_name.exists():
            print(f'Found geojson file {geo_local_name}, skipping write.')
        else:
            cloudcoverage = extract_key_from_collection(row['double'], 'cloudcoverpercentage')
            mask_name = args.save_dir.joinpath(f"{row['title']}_segmentation.png").as_posix()
            geom = wkt.loads(row['bbox']) # parse the string polygon into shapely.polygon
            # todo: handle nans later if they are found necessary
            geo = Feature(id = row['title'], geometry = geom, properties={'datetime': extract_key_from_collection(row['date'], 'beginposition'), 'proj:epsg':np.nan, 'checksum:multihash':np.nan, 'eo:gsd': np.nan, 'tif': np.nan, 'mask': mask_name,
            'cloudcoverpercentage': cloudcoverage,})
            jsonFile = open(geo_local_name, "w")
            
            json.dump(geo, jsonFile, indent=4)
            jsonFile.close()
        return unzipped_local_name

    features = features.truncate(after=max(0, args.max_rows-1))
    downloaded = list(map(str, features.apply(download_file, axis=1)))
        
    args.max_rows = max(0, args.max_rows - len(features))

    with open('sentinel_downloaded.csv', 'a') as f:
        features['link'].to_csv(args.save_dir.joinpath('sentinel_downloaded.csv'), index=False, header=False, mode='a')
    print(f'Done with the range {start_row}-{start_row+args.rows-1}. ')
    if args.max_rows > 0:
        url = get_url(args, start_row=start_row+args.rows)
        return [*downloaded, *download_zips(url, args, start_row=start_row+args.rows)]
    else:
        return downloaded


def unzip_sentinel(path_to_zip, out_path=None):
    try:
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        if out_path is None:
            out_path = Path(path_to_zip)
        zip_ref.extractall(out_path)
        zip_ref.close()
    except FileNotFoundError:
        warnings.warn(f'Zip file not found {path_to_zip}.')
    except zipfile.BadZipFile:
        warnings.warn(f'Zip file bad format: {path_to_zip}')
    except Exception as e:
        print(f'Unknown error: {e}.')