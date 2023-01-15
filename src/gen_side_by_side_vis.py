import argparse
import itertools
import json
import math
import os
from pathlib import Path
from PIL import Image
import shutil


page_digits = 4

style = """
html, body {
    margin: 0;
    font-family: sans-serif;
}
body {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.link-bar a, .link-bar a:visited {
    padding: 5px;
    text-decoration: none;
    border: solid 1px #000;
    color: #000;
}
.link-bar a.disabled {
    opacity: 0.5;
}
td {
    border: solid 1px #000;
    border-collapse: collapse;
    text-align: center;
}
th {
    font-weight: bold;
}
h2 {
    margin-top: 20px;
}
h3 {
    margin-top: 5px;
    font-weight: 200;
}
.container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.link-bar {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    margin-top: 20px;
    margin-bottom: 20px;
}
.link-bar a:first-child {
    margin-right: 20px;
}
.table-header {
    font-weight: bold;
}
.coords {
    font-size: small;
    line-height: 150%;
}
"""

def gen_page_html(page_idx, num_pages, num_per_page, column_names, img_size, img_list, has_geojsons):
    # img_list is supposed to contain *all* images, not just the ones for this page!
    img_list_rows = img_list[page_idx * num_per_page:(page_idx+1) * num_per_page]
    if len(img_list_rows) == 0:
        return None
    geo_column = '<td>📌 Coords (WGS-84)</td>' if has_geojsons else ''
    size_str = f' width="{img_size}px" height="{img_size}px"'

    odd_even_dict = {0: 'even', 1: 'odd'}
    rows = list(map(lambda row_pair:
                    (f'<tr class="{odd_even_dict[row_pair[0] % 2]}-row">'+
                    ''.join(map(lambda column_pair:
                                f'<td><a href="{row_pair[1][1][column_pair[1]]}" target="_blank"><img src="{row_pair[1][1][column_pair[1]]}"{size_str} /></a></td>'
                                if column_pair[1] != '$coords'
                                else (f'<td class="coords">'
                                      f'<a href="https://www.google.com/maps/@{row_pair[1][1][column_pair[1]][1][1]},'
                                      f'{row_pair[1][1][column_pair[1]][0][0]},195m/data=!3m1!1e3" target="_blank">'
                                      f'{str(row_pair[1][1][column_pair[1]]).replace(", ", ",<br>")}</td>'
                                        if has_geojsons else ''),
                    enumerate(row_pair[1][1].keys()))) +
                    '</tr>'),
                    enumerate(img_list_rows)))

    return f"""
    <html>
    <head>
        <meta charset="UTF-8" />
        <title>SupReMap Visualization - Page 1</title>
        <link rel="stylesheet" href="style.css" />
    </head>
    <body>
        <div class="container">
            <h2>🛰️ SupReMap Visualization</h2>
            <h3>📖 Page {page_idx + 1} / {num_pages}</h3>
            <table cellpadding="5px" cellspacing="2px">
                <tr class="table-header">{''.join(f'<td>{col}</td>' for col in column_names)}{geo_column}</tr>
                {''.join(map(lambda row: ' ' * 20 + row, rows))}
            </table>
            <div class="link-bar">
                <a href="page-{f'%0{page_digits}i' % max(1, page_idx)}.html"{' class="disabled"' if page_idx == 0 else ""}>⬅️ Previous page</a>
                <a href="page-{f'%0{page_digits}i' % min(num_pages, page_idx + 2)}.html"{' class="disabled"' if page_idx == num_pages-1 else ""}>➡️ Following page</a>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='Path to input directory from which to get images. '\
                                            'Possible to supply multiple.', type=str, action='append')
    parser.add_argument('--column-name', help='Column names to use. '\
                                              'Possible to supply multiple.', type=str, action='append')
    parser.add_argument('--output-dir', help='Directory to store outputs to.', type=str)
    parser.add_argument('--num-per-page', help='Number of images to show per page.', type=int, default=10)
    parser.add_argument('--geojson-dir', help='Directory with GeoJSONs.', type=str, default=None)
    parser.add_argument('--image-size', help='Size of an image.', type=int, default=None)
    parser.add_argument('--max-num-rows', help='Maximum number of rows.', type=int, default=1e10)

    args = parser.parse_args()
    img_corrs = {}
    suffixes = {'.bmp', '.jpg', '.png', '.tif', '.tiff'}
    suffixes = {*suffixes, *map(str.upper, suffixes)}

    for input_dir_idx, input_dir in enumerate(args.input_dir):
        subtype_output_dir = os.path.join('imgs', str(input_dir_idx))
        os.makedirs(os.path.join(args.output_dir, subtype_output_dir), exist_ok=True)
        for entry in Path(input_dir).iterdir():
            if entry.suffix not in suffixes:
                continue
            
            fn = os.path.splitext(entry.name)[0]
            if fn not in img_corrs:
                img_corrs[fn] = {}
            
            # copy file
            file_output_path = os.path.join(args.output_dir, subtype_output_dir, entry.name)
            if not os.path.isfile(file_output_path):
                shutil.copy(str(entry.absolute()), file_output_path)
            img_corrs[fn][input_dir] = os.path.join(subtype_output_dir, entry.name)

    has_geojsons = args.geojson_dir not in {'', None}
    if has_geojsons:
        for entry in Path(args.geojson_dir).iterdir():
            if not entry.suffix.lower() in {'.json', '.geojson'}:
                continue

            fn = os.path.splitext(entry.name)[0]
            if not fn in img_corrs:
                continue

            with open(str(entry), 'r') as f:
                geo_data = json.load(f)
                if 'geometry' not in geo_data or 'coordinates' not in geo_data['geometry']:
                    continue
                img_corrs[fn]['$coords'] = geo_data['geometry']['coordinates']

    with open(os.path.join(args.output_dir, 'style.css'), 'w') as f:
        f.write(style)
    
    with open(os.path.join(args.output_dir, 'index.html'), 'w') as f:
        f.write(f'<html><head><meta http-equiv="refresh" content="0; url=\'page-{f"%0{page_digits}i" % 1}.html\'" />'
                f'</head><body /></html>')

    num_imgs = 0
    img_list = list(img_corrs.items())[:int(args.max_num_rows)]
    num_pages = int(math.ceil(len(img_list) / args.num_per_page))
    for page_idx in itertools.count():
        html = gen_page_html(page_idx, num_pages, args.num_per_page, args.column_name, args.image_size, img_list,
                             has_geojsons)
        if html is None:
            break

        with open(os.path.join(args.output_dir, f'page-{f"%0{page_digits}i" % (page_idx + 1)}.html'), 'w') as f:
            f.write(html)
