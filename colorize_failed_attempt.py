# it's simple, put color in the black and white pictures

import tensorflow as tf
import argparse, os
import skimage.transform
from skimage.io import imsave, imread

IMG_SIZE = 224

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

def load_image(path):
    '''
    Load the image at path, make sure it is 224x224, grayscale it, return it
    '''
    image = imread(path)

    if (image.shape[0] > IMG_SIZE) or (image.shape[1] > IMG_SIZE):
        # crop image
        short_edge = min(image.shape[:2])
        yy = int((image.shape[0] - short_edge) / 2)
        xx = int((image.shape[1] - short_edge) / 2)
        image = image[ yy : yy + short_edge, xx : xx + short_edge]

    # resize image, do this every time so the image is the right format
    image = skimage.transform.resize(image, (IMG_SIZE, IMG_SIZE))

    # convert to grayscale
    image = (image[:,:,0] + image[:,:,1] + image[:,:,2]) / 3.0

    return image

def main(args):
    # this is the trained model
    with open("vgg16-20160129.tfmodel", mode='rb') as f:
        file_content = f.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file_content)
        grayscale = tf.placeholder("float", [1, 224, 224, 3])
        tf.import_graph_def(graph_def, input_map={"images": grayscale})

    # list of paths to the image files
    paths = get_paths(args.data_dir, int(args.max_images))

    for path in paths:
        print(path)
        gray_image = load_image(path).reshape(1, 224, 224, 3)
        #gray_image = tf.image.rgb_to_grayscale(imread(path))

        # we can now put the gray image through the network
        with tf.Session() as sess:
            inferred_rgb = sess.graph.get_tensor_by_name('import/conv4_3/Relu:0')
            inferred_batch = sess.run(inferred_rgb, feed_dict={grayscale: gray_image})

            # save files like this:
            save_path = "out/" + path
            print(save_path)
            imsave(save_path, inferred_batch[0]) # path must end in .JPEG

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

    args = parser.parse_args()

    main(args)
