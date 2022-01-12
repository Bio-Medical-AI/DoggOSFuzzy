import pandas as pd
import numpy as np
from typing import List, Dict, NoReturn


class InconsistenciesRemover:
    """
    Class for removing inconsistencies from a decision table for rule induction:
    https://www.mdpi.com/2076-3417/11/8/3484 - 2.2.3. Rule Induction with Information Systems: Inconsistency Elimination

    Attributes
    --------------------------------------------
    __dataset: pd.DataFrame
        fuzzified dataset containing possible inconsistencies

    __feature_labels: List[str]
        labels of features to consider for calculating samples identity

    __target_label: str
        label of the target prediction value

    __clean_decisions: pd.DataFrame
        subset of dataset that is consistent

    __lower_approx_for_decisions: Dict[int, int]
        count of consistent samples with given decisions

    Methods
    --------------------------------------------
    remove_inconsistencies(self) -> pd.DataFrame:
        removes inconsistencies from given dataset by applying lower approximation precision analysis
    """
    __dataset: pd.DataFrame
    __feature_labels: List[str]
    __target_label: str
    __clean_decisions: pd.DataFrame
    __lower_approx_for_decisions: Dict[int, int]

    def __init__(self, dataset: pd.DataFrame, feature_labels: List[str], target_label: str = 'Decision'):
        """
        Creates InconsistenciesRemover for removing inconsistent samples from given dataset.

        :param dataset: fuzzified dataset containing possible inconsistencies
        :param feature_labels: labels of features to consider for calculating samples identity
        :param target_label: label of the target prediction value
        """
        self.__dataset = dataset.reset_index().drop(columns=['index'])
        self.__feature_labels = feature_labels
        self.__target_label = target_label
        self.__clean_decisions = None
        self.__lower_approx_for_decisions = {}

    def _find_conflicts(self) -> List[pd.Series]:
        """
        Finds samples that are causing conflicts in dataset.

        :return: samples that are causing conflicts in dataset
        """
        columns = self.__feature_labels.copy()
        columns.append(self.__target_label)
        samples_with_decisions = self.__dataset.groupby(columns, as_index=False).size().reset_index()
        samples = self.__dataset.drop(columns=self.__target_label)
        samples = samples.groupby(self.__feature_labels, as_index=False).size().reset_index()
        conflicts = []
        for _, sample in samples.iterrows():
            truth_table = samples_with_decisions[self.__feature_labels] == sample[self.__feature_labels]
            truth_vector = np.where(truth_table.values.all(-1))[0]
            matches = samples_with_decisions.loc[truth_vector, samples_with_decisions.columns]
            if len(matches[self.__feature_labels[0]]) > 1:
                conflicts.append(sample)
        return conflicts

    def _find_lower_approx(self, conflicts: List[pd.Series]) -> NoReturn:
        """
        Calculates lower approximation of sets consisting of samples with corresponding decision.

        :param conflicts: samples that are causing conflicts in dataset
        :return: NoReturn
        """
        self.__clean_decisions = self.__dataset.copy()
        indices_to_remove = []
        for conflict in conflicts:
            indices = np.where((conflict[self.__feature_labels] == self.__clean_decisions[self.__feature_labels]
                                ).values.all(-1))[0]
            indices_to_remove.extend(indices)

        self.__clean_decisions = self.__clean_decisions.drop(index=indices_to_remove, axis=0)
        decisions = np.unique(self.__dataset[self.__target_label].values)
        for decision in decisions:
            self.__lower_approx_for_decisions[decision] = self.__clean_decisions.loc[
                self.__clean_decisions[self.__target_label] == decision].values.shape[0]

    def _solve_conflicts(self, conflicts: List[pd.Series]) -> NoReturn:
        """
        Changes decisions for conflicting rows to those occurring more often.

        :param conflicts: sets of features that cause conflicts
        :return: NoReturn
        """
        conflicting_rows = []
        conflicting_indices = []
        for conflict in conflicts:
            indices = np.where((conflict[self.__feature_labels] == self.__dataset[self.__feature_labels]).values.all(-1))[0]
            conflicting_indices.append(indices)

        for indices in conflicting_indices:
            batch = pd.DataFrame(columns=self.__dataset.columns)
            for index in indices:
                batch = batch.append(self.__dataset.loc[index, self.__dataset.columns])
            conflicting_rows.append(batch)

        for rows_to_solve in conflicting_rows:
            rows_to_solve = rows_to_solve.reset_index()
            decisions = np.unique(rows_to_solve[self.__target_label])
            highest_quality = None
            most_occurrences = -1
            for decision in decisions:
                if self.__lower_approx_for_decisions[decision] > most_occurrences:
                    most_occurrences = self.__lower_approx_for_decisions[decision]
                    highest_quality = decision
            idx_to_preserve = np.where(rows_to_solve[self.__target_label] == highest_quality)[0]
            rows_to_solve = rows_to_solve.loc[idx_to_preserve, rows_to_solve.columns]
            self.__clean_decisions = self.__clean_decisions.append(rows_to_solve)
            self.__clean_decisions = self.__clean_decisions.reset_index().drop(columns=['level_0', 'index'])

    def remove_inconsistencies(self) -> pd.DataFrame:
        """
        Removes inconsistencies from given dataset by applying lower approximation precision analysis.

        :return: decision table without inconsistencies and count of changed decisions
        """
        conflicts = self._find_conflicts()
        self._find_lower_approx(conflicts)
        self._solve_conflicts(conflicts)
        return self.__clean_decisions
