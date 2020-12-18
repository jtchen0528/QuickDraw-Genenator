import argparse
import os
import urllib.request
import numpy as np

def download():
    """
    args: 
    - nums: str, specify how many categories you want to download to your device
    """
    # The file 'categories.txt' includes all categories you want to download as dataset
    with open("./categories.txt", "r") as f:
        classes = f.readlines()
    classes = [c.replace('\n', '').replace(' ', '_') for c in classes]
    print(classes)
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:
        if os.path.exists('./Data/'+c+'.npy') == False:
            cls_url = c.replace('_', '%20')
            path = base+cls_url+'.npy'
            print(path)
            urllib.request.urlretrieve(path, './Data/'+c+'.npy')

    print('download completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Quick, Draw! data from Google and then dump the raw data into cache.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    # Download data.
    download()