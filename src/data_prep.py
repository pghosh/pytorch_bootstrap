import numpy as np

import glob
import os
import shutil


def train_test_split_img_folder(
        class_label,
        root_dir="data/train",
        generated_root="data",
        test_size=0.2,
        file_ext='jpg'):
    '''
    Stratified train test split when data is split into class folders
    :param test_size:  should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
    :param file_ext: file extention in the folder
    :param class_label:
    :param root_dir:
    :param generated_root:
    :return:
    '''
    # first make directories for train and test
    test_folder = "{}/val".format(generated_root)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    for dir in class_label:
        file_lists = []
        for file in glob.glob("{}/{}/*.{}".format(root_dir, dir, file_ext)):
            file_lists.append(file)

        shuffled_indices = np.arange(len(file_lists))
        np.random.shuffle(shuffled_indices)
        pivot = int(len(file_lists) * test_size)
        test_idx = shuffled_indices[pivot:]
        os.makedirs("{}/{}".format(test_folder, dir))
        for idx in test_idx:
            test_file = file_lists[idx]
            shutil.move(test_file, "{}/{}/{}".format(test_folder, dir, os.path.basename(test_file)))

if __name__ == '__main__':
    class_label = ['c{}'.format(i) for i in range(10)]
    train_test_split_img_folder(class_label=class_label)