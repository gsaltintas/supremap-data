import argparse
from glob import glob
from pathlib import Path
from typing import Union

import cv2


def downsample(img_path: Union[str, Path], save_dir: Union[str, Path], downsampling_factor:int, interpolation) -> None:
    im = cv2.imread(img_path)
    h, w = im.shape[0], im.shape[1]
    dstsize = (w + 1) // downsampling_factor, (h + 1) // downsampling_factor
    # dst = cv2.pyrDown(im, dstsize=dstsize)
    dst = cv2.resize(im, dstsize, interpolation=interpolation)
    cv2.imwrite(Path(save_dir).joinpath(Path(img_path).name).as_posix(), dst)

def main(args):
    if not args.path.exists():
        raise NotADirectoryError(f'Provided input directory {args.path} does not exist.')

    paths = []
    for extension in args.suffix:
        paths.extend(glob(args.path.joinpath(f'*.{extension}').as_posix()))

    for img_path in paths:
        downsample(img_path, args.save_dir, args.factor, args.interpolation)
    
    pass

def parse_path(x):
    p = Path(x).resolve().absolute()
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Input directory containing images.', type=lambda x: Path(x).resolve().absolute())
    parser.add_argument('--save_dir', help='Target directory to save the downsampled images.', type=parse_path, default='../out/downsampled')
    parser.add_argument('--factor', help='Downsampling factor', type=int, default=4)
    parser.add_argument('-i', '--interpolation', help='Downsampling strategy', default=cv2.INTER_AREA)
    parser.add_argument('-s', '--suffix', help='Input image formats, possible options: tif, png, jpg', nargs='*', default=['tif'])
    args = parser.parse_args()
    main(args)

