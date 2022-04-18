import sys
import pandas as pd
import numpy as np


def main():
    results_file = sys.argv[1]
    df = pd.read_csv(results_file)
    test_f1 = np.asarray(df['test_f1'].tolist())
    print(f"Max test_f1: {test_f1.max()}")
    print(f"Min test_f1: {test_f1.min()}")
    print(f"Avg test_f1: {test_f1.mean()}")
    print(f"Std test_f1: {test_f1.std()}")
    #print(df[df['f1'] == test_f1.max()])

if __name__ == '__main__':
    main()
