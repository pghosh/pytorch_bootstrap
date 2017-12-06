import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset

IMG_HEIGHT = 75
IMG_WIDTH = 75


class FileCsvJsonDataset(Dataset):
    '''
    Dataset wrapping csv or json details files iteration. This class assumes if file is not csv it will be
    json. if any other type is provided class will throw exception
    '''

    def __init__(self, target_col, file_name='train', is_csv=True, transform=None):
        if is_csv:
            self.df = pd.read_json('data/{}.csv'.format(file_name))
        else:
            self.df = pd.read_json('data/{}.json'.format(file_name))

        self.transform = transform
        self.phase = file_name
        self.target_col = target_col

    def train_test_split(self, use_prior=False, val_size=0.1):
        '''
        make train test split once and save the data
        :param val_size:
        :return: index for the data with train/test
        '''
        train_idx_file = "generated/data/train"
        val_idx_file = "generated/data/test"

        if use_prior and os.path.exists("{}.npy".format(train_idx_file)):
            X_train = np.load("{}.npy".format(train_idx_file))
            X_val = np.load("{}.npy".format(val_idx_file))
        else:
            target = self.df.is_iceberg
            X_train, X_val, y_train, y_val = train_test_split(np.array(self.df.index), np.array(target),
                                                              test_size=val_size)
            np.save(train_idx_file, X_train)
            np.save(val_idx_file, X_val)

        return X_train, X_val

    def load_test_data(self):
        test_idx_file = "generated/data/test"

        if not os.path.exists("{}.npy".format(test_idx_file)):
            raise ValueError("Train/test split should happen before calling this.")
        else:
            X = np.load("{}.npy".format(test_idx_file))
        return X

    def train_val_split(self, val_size=0.1):
        '''
        make train validation split once and save the data
        :param val_size:
        :return: index for the data with train/test
        '''
        train_idx_file = "generated/data/train"

        if not os.path.exists("{}.npy".format(train_idx_file)):
            raise ValueError("Train/test split should happen before calling this.")
        else:
            X = np.load("{}.npy".format(train_idx_file))
            np.random.shuffle(X)
            target = self.df.iloc[X][self.target_col]
            X_train, X_val, y_train, y_val = train_test_split(X, np.array(target),
                                                              test_size=val_size)

        return X_train, X_val

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = self.to_rgb_image(row=row)
        if self.transform is not None:
            img = self.transform(img)

        data = {
            'raw_data': img,
            'id': row['id']
        }
        if self.phase == 'train':
            data['target'] = row[self.target_col]
        return data

    def __len__(self):
        return self.df.shape[0]


if __name__ == '__main__':
    ds = FileCsvJsonDataset()
    train, test = ds.train_test_split(use_prior=False)
    train, val = ds.train_val_split()
