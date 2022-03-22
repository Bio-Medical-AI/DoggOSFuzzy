import csv
import os


class Logger:
    def __init__(self, metaheuristic, dataset):
        self.path = f"results/{metaheuristic}_{dataset}.csv"
        if not os.path.isfile(self.path):
            with open(self.path, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(["f1", "accuracy", "recall", "precision", "balanced_accuracy", "roc_auc",
                                 "n_folds", "n_mfs", "fuzz_type", "adjustment", "lmf_scaling"])

    def log(self, val_f1, test_f1, accuracy, recall, precision, balanced_accuracy, roc_auc,
            n_mfs, fuzz_type, adjustment, sigma_offset, fun_params, n_folds=0):
        with open(self.path, 'a') as file:
            writer = csv.writer(file)
            val_f1 = "%.3f" % val_f1
            test_f1 = "%.3f" % test_f1
            accuracy = "%.3f" % accuracy
            recall = "%.3f" % recall
            precision = "%.3f" % precision
            balanced_accuracy = "%.3f" % balanced_accuracy
            roc_auc = "%.3f" % roc_auc
            params = ""
            for param in fun_params:
                params += "%.3f" % param + " "
            writer.writerow([val_f1, test_f1, accuracy, recall, precision, balanced_accuracy, roc_auc,
                             n_folds, n_mfs, fuzz_type, adjustment, sigma_offset, params])

    def log_kfold(self, val_f1, val_acc, val_recall, val_precision, val_balacc, val_roc_auc, n_mfs, fuzz_type,
                  adjustment, sigma_offset, n_folds):
        with open(self.path, 'a') as file:
            writer = csv.writer(file)
            f1 = "%.3f" % val_f1
            accuracy = "%.3f" % val_acc
            recall = "%.3f" % val_recall
            precision = "%.3f" % val_precision
            balanced_accuracy = "%.3f" % val_balacc
            roc_auc = "%.3f" % val_roc_auc
            writer.writerow([f1, accuracy, recall, precision, balanced_accuracy, roc_auc,
                             n_folds, n_mfs, fuzz_type, adjustment, sigma_offset])
