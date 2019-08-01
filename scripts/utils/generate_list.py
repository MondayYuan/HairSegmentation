import sys
import os
import glob
from argparse import  ArgumentParser
import random


def find_corresponding_images(masks_dir, input_images_dir, input_ext='.png'):
        """
        :param masks_dir: path to the folder which contains the segmentation masks
        :param input_images_dir: path to the folder which contains the input images
        :param input_ext: the extension of the input images (by default png)
        :return a list of tuples (image-mask), where each element is a mapping between the path of the input image
                and the path of the corresponding segmentation mask
        """
        if not os.path.exists(masks_dir) or not os.path.isdir(masks_dir):
            raise ValueError('The path for the masks directory %s does not exist or is not a folder' % masks_dir)
        if not os.path.exists(input_images_dir) or not os.path.isdir(input_images_dir):
            raise ValueError('The path for the input images directory %s does not exist or is not a folder' % input_images_dir)

        # find the segmentation masks (bmp) images in the masks directory
        mask_files = glob.glob(os.path.join(masks_dir, "*.bmp"))
        print('Found %d segmentation masks ' % len(mask_files))

        num_images = len(mask_files)

        image_tuples = []
        for idx, mask_path in enumerate(mask_files):
            basename = os.path.basename(mask_path)
            sys.stdout.write('\r>> Searching image for  segmentation mask %s (%d/%d)\n ' % (basename, idx, num_images))
            sys.stdout.flush()
            img_path = os.path.join(input_images_dir, basename.replace('.bmp', input_ext))
            if os.path.exists(img_path):
                image_tuples.append((img_path, mask_path))
            else:
                sys.stdout.write('\r>> Error! Could not find input image for segmentation %s' % basename)
                sys.stdout.flush()

        return image_tuples

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--mask-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split-rate', type=float, default=0.8)

    args = parser.parse_args()

    image_tuples = find_corresponding_images(args.mask_dir, args.image_dir)

    image_names_list = [os.path.splitext(os.path.basename(x[0]))[0] for x in image_tuples]
    random.shuffle(image_names_list)

    num_train = int(args.split_rate * len(image_names_list))

    f_train = open(os.path.join(args.output_dir, 'train.txt'), 'w')
    f_val = open(os.path.join(args.output_dir, 'val.txt'), 'w')
    f_trainval = open(os.path.join(args.output_dir, 'trainval.txt'), 'w')

    for i, img_name in enumerate(image_names_list):
        if i < num_train:
            f_train.write(img_name + '\n')
        else:
            f_val.write(img_name + '\n')
        f_trainval.write(img_name + '\n')

    f_train.close()
    f_val.close()
    f_trainval.close()





