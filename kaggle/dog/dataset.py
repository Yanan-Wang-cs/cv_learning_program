from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from PIL import Image
import csv

class DogDataset(Dataset):
    def __init__(self, csv_file='', root_dir='/', transform=None, mode='train'):
        self.NUM_CLASSES = 120
        self.mode = mode
        self.labels = self.parseData(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def parseData(self, file_path):
        data = pd.read_csv(file_path)
        data['target'] = 1
        if self.mode is 'train':
            data_pivot = data.pivot('id', 'breed', 'target').reset_index().fillna(0)
        else:
            data_pivot = data.pivot('id', 'target').reset_index().fillna(0)
        if not os.path.exists('labels/class_names.csv'):
            with open('labels/class_names.csv', 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_pivot.columns.values)
            print('Record class name')
        return data_pivot

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0]+'.jpg')
        image = Image.open(img_name)
        if self.mode is 'train':
            labels = self.labels.iloc[idx, 1:].values.astype('float32')
            label = np.argmax(labels)
            # print(label, self.labels.columns.values[label+1], self.labels.iloc[idx, 0], self.labels.iloc[idx, label+1])
        else:
            label = self.labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        sample = (image, label)
        return sample
