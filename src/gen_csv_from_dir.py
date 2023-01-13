import argparse
import time
from pathlib import Path

from geotiff import GeoTiff
from tqdm import tqdm


def gen_csv_from_dir(input_dir, output_path=None):
    if output_path is None:
        output_path =\
            f'supremap_tiffs_{next(filter(str.__len__, reversed(input_dir.split("/"))), "")}_{int(time.time())}.csv'
    
    output_lines = ['file_path,x_center,y_center']
    for entry in tqdm(Path(input_dir).iterdir()):
        enl = entry.name.lower()
        if not enl.endswith('.tif') and not enl.endswith('.tiff') or enl.startswith('_'):
            continue
        
        gt = GeoTiff(str(entry))
        x_center = (gt.tif_bBox[0][0] + gt.tif_bBox[1][0]) / 2
        y_center = (gt.tif_bBox[0][1] + gt.tif_bBox[1][1]) / 2
        line = f'"{str(entry.absolute())}",{x_center},{y_center}'
        output_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', help='Directory to process.', type=str)
    parser.add_argument('-o', '--output-path', help='Path to which to write CSV.', type=str, default=None)
    args = parser.parse_args()

    gen_csv_from_dir(args.input_dir, args.output_path)
