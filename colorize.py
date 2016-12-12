#!/usr/bin/env python3

print("Loading Libraries...")
import sys
import numpy as np
from skimage.io import imsave, imread
import skimage.transform
import argparse, os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import caffe

def colorize(filepaths,modelpath, prototext, hullpoints, output, gpu_on):
    if gpu_on:
        gpu_id = 1
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

    # Select desired model
    net = caffe.Net(prototext,
                    modelpath,
                    caffe.TEST)

    (H_in, W_in) = net.blobs['data_l'].data.shape[2:]       # get input shape
    (H_out, W_out) = net.blobs['class8_ab'].data.shape[2:]  # get output shape

    pts_in_hull = np.load('/home/alanxoc3/Projects/colorization/resources/pts_in_hull.npy')  # load cluster centers
    net.params['class8_ab'][0].data[:, :, 0, 0] = pts_in_hull.transpose(
        (1, 0))  # populate cluster centers as 1x1 convolution kernel

    for path in filepaths:
        # load the original image
        print('Loading image:', path)
        img_rgb = caffe.io.load_image(path)
        print('Classifying image:', path)
    
        img_lab = color.rgb2lab(img_rgb)  # convert image to lab color space
        img_l = img_lab[:, :, 0]  # pull out L channel
        (H_orig, W_orig) = img_rgb.shape[:2]  # original image size
    
        # resize image to network input size
        img_rs = caffe.io.resize_image(img_rgb, (H_in, W_in))  # resize image to network input size
        img_lab_rs = color.rgb2lab(img_rs)
        img_l_rs = img_lab_rs[:, :, 0]
    
        net.blobs['data_l'].data[0, 0, :, :] = img_l_rs - 50  # subtract 50 for mean-centering
        net.forward()  # run network
    
        ab_dec = net.blobs['class8_ab'].data[0, :, :, :].transpose((1, 2, 0))  # this is our result
        ab_dec_us = sni.zoom(ab_dec,
                            (1. * H_orig / H_out, 1. * W_orig / W_out, 1))  # upsample to match size of original image L
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)  # concatenate with original image L
        img_rgb_out = np.clip(color.lab2rgb(img_lab_out), 0, 1)  # convert back to rgb
    
        file_name = os.path.basename(path)
        save_path = output + file_name
        print('Saving image to: ' + save_path)
        imsave(save_path, img_rgb_out)
        print() # New line!
    
def main(args):
    if len(args.files) >= 2:
        in_files = args.files[:-1]
        out_file = args.files[-1]
        colorize(in_files, args.model, args.prototext, args.hullpoints, out_file, args.gpu)
    else:
        print("There weren't enough arguments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorizes gray-scale images.')

    parser.add_argument('files', nargs='*')

    parser.add_argument('-m', '--model',
                        help='The model used to colorize, a caffemodel file.',
                        required=True)

    parser.add_argument('-p', '--prototext',
                        help='The prototext file for the model.',
                        required=True)

    parser.add_argument('-g', '--gpu',
                        help='Whether or not to enable the gpu.',
                        required=False,
                        action="store_true")

    parser.add_argument('-l', '--hullpoints',
                        help='The hull points numpy file.',
                        required=True)

    args = parser.parse_args()

    try:
        main(args)
    except:
        sys.exit(2)
