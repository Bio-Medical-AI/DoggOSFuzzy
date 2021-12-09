import csv


class Logger:
    def __init__(self, metaheuristic, dataset):
        self.path = f"results\\{metaheuristic}_{dataset}.csv"
        with open(self.path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["val_f1", "test_f1", "n_folds", "n_mfs", "fuzz_type", "adjustment", "lmf_scaling"])

    def log(self, val_f1, test_f1, n_folds, n_mfs, fuzz_type, adjustment, sigma_offset):
        with open(self.path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([val_f1, test_f1, n_folds, n_mfs, fuzz_type, adjustment, sigma_offset])

# F1    n_folds n_mfs   fuzz_type   adjustment  sigma_offset date
