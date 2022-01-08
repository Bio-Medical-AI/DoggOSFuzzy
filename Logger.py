import csv
import os


class Logger:
    def __init__(self, metaheuristic, dataset):
        self.path = f"results\\{metaheuristic}_{dataset}.csv"
        if not os.path.isfile(self.path):
            with open(self.path, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(["val_f1", "test_f1", "accuracy", "recall", "precision", "balanced_accuracy", "roc_auc",
                                 "n_folds", "n_mfs", "fuzz_type", "adjustment", "lmf_scaling", "lin_fun_params"])

    def log(self, val_f1, test_f1, accuracy, recall, precision, balanced_accuracy, roc_auc,
            n_mfs, fuzz_type, adjustment, sigma_offset, fun_params, n_folds=0):
        with open(self.path, 'a') as file:
            writer = csv.writer(file)
            val_f1 = "%3.f" % val_f1
            test_f1 = "%3.f" % test_f1
            accuracy = "%3.f" % accuracy
            recall = "%3.f" % recall
            precision = "%3.f" % precision
            balanced_accuracy = "%3.f" % balanced_accuracy
            roc_auc = "%3.f" % roc_auc
            params = ""
            for param in fun_params:
                params += "%3.f" % param
            writer.writerow([val_f1, test_f1, accuracy, recall, precision, balanced_accuracy, roc_auc,
                             n_folds, n_mfs, fuzz_type, adjustment, sigma_offset, params])
