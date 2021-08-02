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
        dataset containing possible inconsistencies

    __feature_labels: List[str]
        labels of feature to consider for calculating identity

    __clean_decisions: int
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
    __clean_decisions: pd.DataFrame
    __lower_approx_for_decisions: Dict[int, int]

    def __init__(self, dataset: pd.DataFrame, feature_labels: List[str]):
        """
        Creates InconsistenciesRemover for removing inconsistent samples from given dataset\n
        :param dataset: dataset containing possible inconsistencies
        :param feature_labels: labels of feature to consider for calculating identity
        """
        self.__dataset = dataset
        self.__feature_labels = feature_labels
        self.__clean_decisions = None
        self.__lower_approx_for_decisions = {}

    def _find_conflicts(self) -> List[pd.Series]:
        """
        Finds samples that are causing conflicts in dataset\n
        :return: samples that are causing conflicts in dataset
        """
        samples = self.__dataset.copy()
        columns = self.__feature_labels.copy()
        columns.append('Decision')
        samples_with_decisions = samples.groupby(columns, as_index=False).size().reset_index()
        samples = self.__dataset.copy()
        samples = samples.drop(columns='Decision')
        samples = samples.groupby(self.__feature_labels, as_index=False).size().reset_index()
        conflicts = []
        for _, sample in samples.iterrows():
            truth_table = samples_with_decisions[self.__feature_labels] == sample[self.__feature_labels]
            truth_vector = np.where(truth_table.values.all(-1))[0]
            matches = samples_with_decisions.loc[truth_vector, samples_with_decisions.columns]
            if len(matches[self.__feature_labels[0]]) > 1:
                conflicts.append(sample)
        return conflicts

    def _find_lower_approx(self, conflicts: pd.DataFrame) -> NoReturn:
        """
        Calculates lower approximation of sets consisting of samples with corresponding decision\n
        :param conflicts: samples that are causing conflicts in dataset
        :return: NoReturn
        """
        self.__clean_decisions = self.__dataset.copy()
        indices_to_remove = []
        for conflict in conflicts:
            indices = np.where((conflict[self.__feature_labels] == self.__clean_decisions[self.__feature_labels]
                                ).values.all(-1))[0]
            indices_to_remove.extend(indices)

        for index in indices_to_remove:
            self.__clean_decisions = self.__clean_decisions.drop(index=index, axis=0)

        decisions = np.unique(self.__dataset['Decision'].values)
        for decision in decisions:
            self.__lower_approx_for_decisions[decision] = self.__clean_decisions.loc[
                self.__clean_decisions['Decision'] == decision].values.shape[0]

    def _solve_conflicts(self, conflicts: pd.DataFrame) -> NoReturn:
        """
        Changes decisions for conflicting rows to those occurring more often\n
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
            decisions = np.unique(rows_to_solve['Decision'])
            highest_quality = None
            most_occurrences = -1
            for decision in decisions:
                if self.__lower_approx_for_decisions[decision] > most_occurrences:
                    most_occurrences = self.__lower_approx_for_decisions[decision]
                    highest_quality = decision
            idx_to_preserve = np.where(rows_to_solve['Decision'] == highest_quality)[0]
            rows_to_solve = rows_to_solve.loc[idx_to_preserve, rows_to_solve.columns]
            self.__clean_decisions = self.__clean_decisions.append(rows_to_solve)
            self.__clean_decisions = self.__clean_decisions.reset_index().drop(columns=['level_0', 'index'])

    def remove_inconsistencies(self) -> pd.DataFrame:
        """
        Removes inconsistencies from given dataset by applying lower approximation precision analysis\n
        :return: decision table without inconsistencies and count of changed decisions
        """
        conflicts = self._find_conflicts()
        self._find_lower_approx(conflicts)
        self._solve_conflicts(conflicts)
        return self.__clean_decisions
