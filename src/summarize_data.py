import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

color_mapping = {'water': 'blue', 'field': 'green', 'beach': 'brown',
                                        'road': 'red', 'building': 'black',
                                        'background': (255, 255, 255)}
color_back_mapping = {(255, 255, 255): 'background', (255, 0, 0): 'road',(0,0,0): 'building',  (0, 0, 255):'water', (0, 128, 0): 'field',
                    #   brown has different hues
                      (165, 42, 42): 'beach',}


def summarize(path, hist=True):
    count = pd.DataFrame(0,columns=[*list(color_back_mapping.values())], index=['train', 'val'])
    for prefix in ['train', 'val']:
        for file in path.joinpath(prefix, 'seg_maps_visual').iterdir():
            im  = Image.open(file).convert('RGB')
            im = np.asarray(im)
            unique_colors =np.unique(im.reshape((-1, im.shape[-1])), axis=0, return_counts=True)
            for col, count_ in zip(*unique_colors):
                col_name = color_back_mapping.get(tuple(col))
                if col_name is None:
                    print(f'Unkown color: {col}')
                
                count.loc[prefix, col_name] += count_
    # drop zeros
    count = count.loc[:, (count != 0).any(axis=0)]
    if hist:
        df = count.reset_index(names=('prefix'))
        df = pd.melt(df, id_vars='prefix', var_name='category', value_name='count')
        df.sort_values('count', ascending=False, inplace=True)
        print(df)

        sns.barplot(data=df, palette='viridis', hue='prefix', x='category', y='count')
        plt.savefig(f'{path.name}_count.png')
    print(count)
    
    
    
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Creates a basic summary of the pixel densities of each object')
    parser.add_argument('-p', '--data-path', help='Path to the data', default='/data/supremap_imaginaire_mini_dataset_v4_filtered/', type=lambda x:  Path(x).resolve().absolute())
    
    parser.add_argument('--hist', help='Pass to create a histogram.', action='store_true')
    args=parser.parse_args()
    
    summarize(args.data_path, args.hist)