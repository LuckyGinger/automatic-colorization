import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import skimage.transform
import argparse, os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import caffe


def main(args):
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorize black and white pictures')
    # parser.add_argument('-d', '--data_dir',
    #                     help='where the pictures are located',
    #                     required=False,
    #                     default='.')
    # parser.add_argument('-m', '--max_images',
    #                     help='max number of images to load',
    #                     required=False,
    #                     default=100000)
    parser.add_argument('-v', '--verbose',
                        help='be verbose',
                        required=False,
                        default=0)
    #parser.add_argument('-f', '--input_files',
    #                    help='specify a list of file paths in a file',
    #                    default=False)
    parser.add_argument('-i', '--iteration',
                        help='which snapshot to use',
                        required=True)
    parser.add_argument('-p', '--picture',
                        help='the picture to colorize, it can be color or black and white',
                        required=True)

    args = parser.parse_args()
    main(args)