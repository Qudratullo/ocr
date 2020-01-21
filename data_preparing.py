import os
import numpy as np
import scipy.io as sio

from prepare_image import prepare_image


def prepare_data(folder_name, save_file_name):
    data_X = []
    data_y = []

    # read input images from 'in' directory
    img_files = os.listdir(f'{folder_name}/')
    for (i, file_name) in enumerate(img_files):
        images = prepare_image(f'{folder_name}/{file_name}')

        data_X.extend(images)
        data_y.extend([ord(file_name[0])] * len(images))

    print(len(data_X))
    print(len(data_y))
    print(data_X[0:1])
    sio.savemat(f'data/{save_file_name}_32x32.mat', {'X': data_X, 'y': data_y})


def main():
    prepare_data('train_data', 'train')
    prepare_data('test_data', 'test')


if __name__ == '__main__':
    main()

# venv/bin/trdg -c 517 -l al -m 0,0,0,0 -b 1 --output_dir train_data
