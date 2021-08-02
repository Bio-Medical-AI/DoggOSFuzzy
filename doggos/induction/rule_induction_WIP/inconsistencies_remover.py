import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


class InconsistenciesRemover(object):
    """
    Class for removing inconsistencies from a decision table for rule induction:
    https://www.mdpi.com/2076-3417/11/8/3484 - 2.2.3. Rule Induction with Information Systems: Inconsistency Elimination

    Attributes
    --------------------------------------------
    __decision_table: pd.DataFrame
        fuzzy sets that are describing columns of given dataset

    __feature_labels: List[str]
        dataset to fuzzify - target column should be named 'Decision'

    __changed_decisions: int
        linguistic variables (columns) from given dataset

    Methods
    --------------------------------------------
    remove_inconsistencies(self) -> Tuple[pd.DataFrame, int]:
        removes inconsistencies from given dataset by applying lower approximation precision analysis\n
    """

    __decision_table: pd.DataFrame
    __feature_labels: List[str]
    __changed_decisions: int

    def __init__(self, decision_table: pd.DataFrame, feature_labels: List[str]):
        self.__decision_table = decision_table
        self.__feature_labels = feature_labels
        self.__changed_decisions = 0

    def __get_occurrence_of_rows(self, df: pd.DataFrame,
                                 columns_to_remove: List[str] or Optional[str] = None,
                                 by_columns: List[str] = None) -> pd.DataFrame:
        """
        Calculates number of occurrences of identical rows in dataframe, then puts it into Occurrence column\n
        :param df: dataframe containing same rows to count
        :param columns_to_remove: columns to be removed from returned dataframe
        :return: dataframe grouped by identical rows with 'Occurrence' column which describes count of identical rows
        """
        if columns_to_remove is not None:
            df = df.drop(columns_to_remove, axis=1)

        df = df.groupby(by_columns, as_index=False).size().reset_index()
        df.rename(columns={'size': 'Occurrence'}, inplace=True)

        return df

    def __get_certain_decision_rows(self, features_occurrence: pd.DataFrame,
                                    features_decisions_occurrence: pd.DataFrame) -> pd.DataFrame:
        """
        Selects sets of attributes that led to the same decision\n
        :param features_occurrence: dataframe with sets of features and their occurrence count in a dataset
        :param features_decisions_occurrence: dataframe with sets of features, decisions for them and their
                                              occurrence count in a dataset
        :return: Objects with unique set of attributes that leads to the same decision
        """
        feature_sets_single_occurrences = \
            features_occurrence.loc[features_occurrence['Occurrence'] == 1.].copy()

        feature_decision_single_occurrences = \
            features_decisions_occurrence.loc[features_decisions_occurrence['Occurrence'] == 1].copy()
        print(feature_sets_single_occurrences)
        print(feature_decision_single_occurrences)
        decision_indices = []
        for _, row in feature_sets_single_occurrences.iterrows():
            left = feature_decision_single_occurrences[self.__feature_labels].values
            right = row[self.__feature_labels].values
            # Creates truth matrix that matches condition, then flattens it to 1-D by performing logical AND operation
            # on rows. Numpy.where returns 2-elem tuple with second element being empty, thus 0-th is taken.
            indices = np.where((left == right).all(-1))[0]
            decision_indices.extend(indices)
        decision_indices = set(decision_indices)
        decisions = [feature_decision_single_occurrences['Decision'].values[idx] for idx in decision_indices]
        feature_sets_single_occurrences['Decision'] = decisions

        return feature_sets_single_occurrences.drop(['Occurrence', 'index'], axis=1)

    def __get_number_of_clear_decisions(self, features_occurrence: pd.DataFrame,
                                        features_decisions_occurrence: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates number of sets of attributes that did not lead to conflicting decisions for each class\n
        :param features_occurrence: dataframe with sets of features and their occurrence count in a dataset
        :param features_decisions_occurrence: dataframe with sets of features, decisions for them and their
                                              occurrence count in a dataset
        :return: number of clear decisions for each class
        """
        target_label = 'Decision'
        features_certain_decision = self.__get_certain_decision_rows(features_occurrence, features_decisions_occurrence)
        print(features_certain_decision)
        merged_decisions = pd.merge(
            features_decisions_occurrence,
            features_certain_decision,
            on=self.__feature_labels,
            )

        merged_decisions = merged_decisions.drop(['Decision_y'], axis=1).rename(
            index=str,
            columns={
                "Decision_x": "Decision",
                "Occurrence_x": "Occurrence"
            })

        number_of_clear_decisions = pd.DataFrame(
            merged_decisions.groupby([target_label], as_index=False)['Occurrence'].agg(np.sum))

        return number_of_clear_decisions

    def __solve_conflicts(self, nums_of_conflicting_decisions: pd.DataFrame,
                          problems_to_solve: pd.DataFrame,
                          features_decisions_occurrence: pd.DataFrame,
                          num_of_clear_decisions: pd.DataFrame) -> pd.DataFrame:
        """
        Changes decisions for conflicting rows to those occurring more often\n
        :param nums_of_conflicting_decisions: sets of features that cause conflicts and their occurrence count
        :param problems_to_solve: rows with conflicts to resolve
        :param features_decisions_occurrence: dataframe with features of objects, decisions for them and count of those instances
        :param num_of_clear_decisions: counts of decisions for all classes that did not cause any conflicts
        :return: dataframe with resolved conflicts
        """
        proba_columns = ['Decision', 'Probability']
        for _, row in nums_of_conflicting_decisions.iterrows():
            proba_df = pd.DataFrame(columns=proba_columns)

            for _, row_2 in problems_to_solve.iterrows():
                is_same_row = (row[self.__feature_labels].values == row_2[self.__feature_labels]).all()
                if is_same_row:
                    if row_2['Decision'] in num_of_clear_decisions['Decision'].values:
                        occurrence = num_of_clear_decisions.loc[
                            num_of_clear_decisions['Decision'] == row_2['Decision']
                            ]['Occurrence'].values.item()
                    else:
                        occurrence = 0

                    probability = occurrence / len(self.__decision_table)
                    proba_df = proba_df.append({
                        'Decision': row_2['Decision'],
                        'Probability': probability
                    }, ignore_index=True)

            new_value = proba_df.loc[
                proba_df['Probability'].idxmax()
            ]['Decision'].item()

            for idx, row_decision_table in features_decisions_occurrence.iterrows():
                if (row[self.__feature_labels].values == row_decision_table[self.__feature_labels]).all():
                    if row_decision_table['Decision'] != new_value:
                        features_decisions_occurrence.loc[idx, 'Decision'] = new_value
                        self.__changed_decisions = self.__changed_decisions + 1

        return features_decisions_occurrence

    def remove_inconsistencies(self) -> Tuple[pd.DataFrame, int]:
        """
        Removes inconsistencies from given dataset by applying lower approximation precision analysis\n
        :return: decision table without inconsistencies and count of changed decisions
        """
        considering_columns = self.__feature_labels.copy()
        considering_columns.append('Decision')
        features_decisions_occurrence = self.__get_occurrence_of_rows(self.__decision_table, None, considering_columns)
        features_decisions_occurrence = features_decisions_occurrence.drop(['index'], axis=1)

        feature_sets_occurrence = self.__get_occurrence_of_rows(self.__decision_table, ['Decision'],
                                                                self.__feature_labels)

        nums_of_conflicting_decisions = feature_sets_occurrence[feature_sets_occurrence['Occurrence'] > 1]
        num_of_clear_decisions = self.__get_number_of_clear_decisions(feature_sets_occurrence,
                                                                      features_decisions_occurrence)
        print(num_of_clear_decisions)
        problems_to_solve = pd.merge(
            features_decisions_occurrence,
            nums_of_conflicting_decisions,
            how='inner',
            on=self.__feature_labels).drop(['Occurrence_x', "Occurrence_y"], axis=1)

        features_decisions_occurrence = self.__solve_conflicts(
            nums_of_conflicting_decisions, problems_to_solve,
            features_decisions_occurrence, num_of_clear_decisions)

        decision_table = features_decisions_occurrence.drop(['Occurrence'], axis=1).drop_duplicates()

        return decision_table, self.__changed_decisions

class InconsistenciesRemoverFix:
    """
    Class for removing inconsistencies from a decision table for rule induction:
    https://www.mdpi.com/2076-3417/11/8/3484 - 2.2.3. Rule Induction with Information Systems: Inconsistency Elimination

    Attributes
    --------------------------------------------
    __decision_table: pd.DataFrame
        fuzzy sets that are describing columns of given dataset

    __feature_labels: List[str]
        dataset to fuzzify - target column should be named 'Decision'

    __changed_decisions: int
        linguistic variables (columns) from given dataset

    Methods
    --------------------------------------------
    remove_inconsistencies(self) -> pd.DataFrame:
        removes inconsistencies from given dataset by applying lower approximation precision analysis\n
    """
    def __init__(self, dataset: pd.DataFrame, feature_labels: List[str]):
        self.dataset = dataset
        self.feature_labels = feature_labels
        self.clean_decisions = None
        self.samples_by_decisions = {}
        self.lower_approx_for_decisions = {}

    def _find_conflicts(self):
        samples = self.dataset.copy()
        columns = self.feature_labels.copy()
        columns.append('Decision')
        samples_with_decisions = samples.groupby(columns, as_index=False).size().reset_index()
        samples = self.dataset.copy()
        samples = samples.drop(columns='Decision')
        samples = samples.groupby(self.feature_labels, as_index=False).size().reset_index()
        conflicts = []
        for _, sample in samples.iterrows():
            truth_table = samples_with_decisions[self.feature_labels] == sample[self.feature_labels]
            truth_vector = np.where(truth_table.values.all(-1))[0]
            matches = samples_with_decisions.loc[truth_vector, samples_with_decisions.columns]
            if len(matches[self.feature_labels[0]]) > 1:
                conflicts.append(sample)
        return conflicts

    def _find_lower_approx(self, conflicts):
        self.clean_decisions = self.dataset.copy()
        indices_to_remove = []
        for conflict in conflicts:
            indices = np.where((conflict[self.feature_labels] == self.clean_decisions[self.feature_labels]
                                ).values.all(-1))[0]
            indices_to_remove.extend(indices)

        for index in indices_to_remove:
            self.clean_decisions = self.clean_decisions.drop(index=index, axis=0)

        decisions = np.unique(self.dataset['Decision'].values)
        for decision in decisions:
            self.lower_approx_for_decisions[decision] = self.clean_decisions.loc[
                self.clean_decisions['Decision'] == decision].values.shape[0]

    def _solve_conflicts(self, conflicts):
        conflicting_rows = []
        conflicting_indices = []
        for conflict in conflicts:
            indices = np.where((conflict[self.feature_labels] == self.dataset[self.feature_labels]).values.all(-1))[0]
            conflicting_indices.append(indices)

        for indices in conflicting_indices:
            batch = pd.DataFrame(columns=self.dataset.columns)
            for index in indices:
                batch = batch.append(self.dataset.loc[index, self.dataset.columns])
            conflicting_rows.append(batch)

        for rows_to_solve in conflicting_rows:
            rows_to_solve = rows_to_solve.reset_index()
            decisions = np.unique(rows_to_solve['Decision'])
            highest_quality = None
            most_occurrences = -1
            for decision in decisions:
                if self.lower_approx_for_decisions[decision] > most_occurrences:
                    most_occurrences = self.lower_approx_for_decisions[decision]
                    highest_quality = decision
            idx_to_preserve = np.where(rows_to_solve['Decision'] == highest_quality)[0]
            rows_to_solve = rows_to_solve.loc[idx_to_preserve, rows_to_solve.columns]
            self.clean_decisions = self.clean_decisions.append(rows_to_solve)
            self.clean_decisions = self.clean_decisions.reset_index().drop(columns=['level_0', 'index'])

    def remove_inconsistencies(self):
        conflicts = self._find_conflicts()
        self._find_lower_approx(conflicts)
        self._solve_conflicts(conflicts)
        return self.clean_decisions

