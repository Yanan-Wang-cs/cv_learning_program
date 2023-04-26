import pandas as pd
from sklearn.model_selection import train_test_split
import os
import csv

labels = pd.read_csv('datasets/dog-breed-identification/labels.csv')
train, val = train_test_split(labels, train_size=0.8, random_state=0)

train.to_csv('labels/train.csv')
val.to_csv('labels/val.csv')

test_files = os.listdir(os.path.join('datasets/dog-breed-identification/test'))
with open('labels/test.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['','id'])
    for index, img in enumerate(test_files):
        writer.writerow([index, img.replace('.jpg', '')])
        print(index, img.replace('.jpg', ''))