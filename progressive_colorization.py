import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import skimage.transform
import argparse, os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import caffe

INPUT_SIZE = 512


def colorize(paths):
    gpu_id = 1
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    for iter in range(370000, 382000, 1000): # change stop number when more snapshots have been saved
        # Select desired model
        net = caffe.Net('/mnt/6TB-WD-Black/cs450/repos/colorization/models/colorization_deploy_v2.prototxt',
                        '/mnt/6TB-WD-Black/cs450/repos/colorization/train/models/colornet_iter_'
                        + str(iter) +
                        '.caffemodel',
                        caffe.TEST)

        (H_in, W_in) = net.blobs['data_l'].data.shape[2:]  # get input shape
        (H_out, W_out) = net.blobs['class8_ab'].data.shape[2:]  # get output shape

        print 'Input dimensions: (%i,%i)' % (H_in, W_in)
        print 'Output dimensions: (%i,%i)' % (H_out, W_out)

        pts_in_hull = np.load(
            '/mnt/6TB-WD-Black/cs450/repos/colorization/resources/pts_in_hull.npy')  # load cluster centers
        net.params['class8_ab'][0].data[:, :, 0, 0] = pts_in_hull.transpose(
            (1, 0))  # populate cluster centers as 1x1 convolution kernel
        print 'Annealed-Mean Parameters populated'

        for path in paths:
            print('image: ' + path)

            # load the original image
            img_rgb = caffe.io.load_image(path)
            first_height, first_width = img_rgb.shape[:2]
            if first_height > INPUT_SIZE or first_width > INPUT_SIZE:
                img_rgb = caffe.io.resize_image(img_rgb, (INPUT_SIZE, INPUT_SIZE)) # in case they are too big

            img_lab = color.rgb2lab(img_rgb)  # convert image to lab color space
            img_l = img_lab[:, :, 0]  # pull out L channel
            (H_orig, W_orig) = img_rgb.shape[:2]  # original image size

            # create grayscale version of image (just for displaying)
            #img_lab_bw = img_lab.copy()
            #img_lab_bw[:, :, 1:] = 0
            #img_rgb_bw = color.lab2rgb(img_lab_bw)

            # resize image to network input size
            img_rs = caffe.io.resize_image(img_rgb, (H_in, W_in))  # resize image to network input size
            img_lab_rs = color.rgb2lab(img_rs)
            img_l_rs = img_lab_rs[:, :, 0]

            # show original image, along with grayscale input to the network
            # img_pad = np.ones((H_orig, W_orig / 10, 3))
            # plt.imshow(np.hstack((img_rgb, img_pad, img_rgb_bw)))
            # plt.title('(Left) Loaded image   /   (Right) Grayscale input to network')
            # plt.axis('off');

            net.blobs['data_l'].data[0, 0, :, :] = img_l_rs - 50  # subtract 50 for mean-centering
            net.forward()  # run network

            ab_dec = net.blobs['class8_ab'].data[0, :, :, :].transpose((1, 2, 0))  # this is our result
            ab_dec_us = sni.zoom(
                ab_dec,
                (1. * H_orig / H_out, 1. * W_orig / W_out, 1)
            )  # upsample to match size of original image L
            img_lab_out = np.concatenate(
                (img_l[:, :, np.newaxis], ab_dec_us),
                axis=2)  # concatenate with original image L
            img_rgb_out = np.clip(color.lab2rgb(img_lab_out), 0, 1)  # convert back to rgb

            # plt.imshow(img_rgb_out)
            # plt.axis('off')

            img_rgb_out = caffe.io.resize_image(img_rgb_out, (first_height, first_width))

            file_name = os.path.basename(path)
            save_path = '/mnt/6TB-WD-Black/cs450/automatic-colorization/SPECIAL_OUT/' + str(iter) + '/' + file_name
            print('save path: ' + save_path)
            imsave(save_path, img_rgb_out)


def get_paths(data_dir, num_images):
    '''
    get num_images images of type .JPEG from within data_dir
    '''
    paths = []

    full_dir = os.path.realpath(data_dir)
    for root, subdirs, files in os.walk(full_dir):
        for _file in files:
            if len(paths) < num_images:
                if _file.endswith('.JPEG'):
                    full_path = os.path.join(root, _file)
                    paths.append(full_path)
            else:
                break

        if len(paths) >= num_images:
            break

    return paths


def load_file(file_path):
    print('loading file: ' + file_path)

    paths = []

    with open(file_path, 'r') as f:
        for x in f:
            paths.append(x.rstrip())

    return paths


def main(args):
    print(args)

    if not args.input_files:
        paths = get_paths(args.data_dir, int(args.max_images))
    else:
        paths = load_file(str(args.input_files))

    # magic
    colorize(paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Colorize black and white pictures')
    parser.add_argument('-d', '--data_dir',
                        help='where the pictures are located',
                        required=False,
                        default='.')
    parser.add_argument('-m', '--max_images',
                        help='max number of images to load',
                        required=False,
                        default=100000)
    parser.add_argument('-v', '--verbose',
                        help='be verbose',
                        required=False,
                        default=0)
    parser.add_argument('-f', '--input_files',
                        help='specify a list of file paths in a file',
                        default=False)

    args = parser.parse_args()
    main(args)