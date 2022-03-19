from copy import deepcopy

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def main():
    for dataset in os.listdir('data'):
        df = pd.read_csv('data/' + dataset, sep=';')
        print(dataset)
        print(df.groupby(['Decision']).count())
        print(f'Dataset size: {len(df)}')
        X = df.drop(columns=['Decision']).values
        y = df['Decision'].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        print(f'Train class 0: {(y_train == 0).sum()}')
        print(f'Train class 1: {(y_train == 1).sum()}')
        print(f'Train sum: {len(y_train)}')
        print(f'Test class 0: {(y_test == 0).sum()}')
        print(f'Test class 1: {(y_test == 1).sum()}')
        print(f'Test sum: {len(y_test)}')


if __name__ == '__main__':
    main()
