import argparse
import os
import urllib.request
import numpy as np
from generate_data import generate_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Quick, Draw! data from Google and then dump the raw data into cache.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', '-root', type=str, default='Data',
                        help='root for the data directory.')
    parser.add_argument('--valfold', '-v', type=float,
                        default=0.2, help='Specify the val fold ratio.')
    parser.add_argument('--max_samples_category', '-msc', type=int, default=5000,
                        help='Specify the max samples per category for your generated dataset.')
    parser.add_argument('--show_random_imgs', '-show', action='store_true',
                        default=False, help='show some random images while generating the dataset.')
    parser.add_argument('--img_size', '-size', type=int,
                        default=28, help='size of image')
    args = parser.parse_args()

    # Generate dataset
    generate_dataset(rawdata_root=args.data_root, vfold_ratio=args.valfold, max_samples_per_class=args.max_samples_category,
                     show_imgs=args.show_random_imgs, frame_size=args.img_size)
