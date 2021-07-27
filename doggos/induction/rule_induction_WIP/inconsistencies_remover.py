import pandas as pd
import numpy as np
from typing import Optional, List


class InconsistenciesRemover(object):

    def __init__(self, decision_table: pd.DataFrame, feature_labels: List[str]):
        self.decision_table = decision_table
        self.feature_labels = feature_labels
        self.changed_decisions = 0
        self.samples = None

    def __get_occurrence_of_rows(self, df: pd.DataFrame,
                                 columns_to_remove: List[str] or Optional[str] = None) -> pd.DataFrame:
        """
        Calculates number of occurrences of identical rows in dataframe, then puts it into Occurrence column
        :param df: dataframe containing same rows to count
        :param columns_to_remove: columns to be removed from returned dataframe
        :return: dataframe grouped by identical rows with 'Occurrence' column which describes count of identical rows
        """
        if columns_to_remove is not None:
            df = df.drop(columns_to_remove, axis=1)

        df = df.groupby(df.columns.tolist(), as_index=False).size().reset_index()
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

        decision_indices = []

        for _, row in feature_sets_single_occurrences.iterrows():
            left = feature_decision_single_occurrences[self.feature_labels].values
            right = row[self.feature_labels].values
            # Creates truth matrix that matches condition, then flattens it to 1-D by performing logical AND operation
            # on rows. Numpy.where returns 2-elem tuple with second element being empty, thus 0-th is taken.
            decision_indices.append(np.where((left == right).all(-1))[0].item())

        decisions = [features_decisions_occurrence['Decision'].values[idx] for idx in decision_indices]
        feature_sets_single_occurrences['Decision'] = decisions

        return feature_sets_single_occurrences.drop(['Occurrence', 'index'], axis=1)

    def __get_number_of_clear_decision(self, features_occurrence: pd.DataFrame,
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

        decisions = features_certain_decision[target_label]
        features_certain_decision = features_certain_decision.drop(columns=target_label)
        features_decisions_occurrence = features_decisions_occurrence.drop(columns=target_label)

        merged_decisions = pd.merge(
            features_decisions_occurrence,
            features_certain_decision,
            on=self.feature_labels,
            suffixes=[])
        merged_decisions[target_label] = decisions

        number_of_clear_decision = pd.DataFrame(
            merged_decisions.groupby([target_label], as_index=False)['Occurrence'].agg(np.sum))

        return number_of_clear_decision

    def solve_conflicts(self, number_of_conflicts_decision, problems_to_solve,
                        features_decisions_occurence, number_of_clear_decision, general_features_occurence):

        for _, row in number_of_conflicts_decision.iterrows():
            new_df = pd.DataFrame(columns={"Decision", "Probability"})

            for _, row_2 in problems_to_solve.iterrows():
                if (row[self.feature_labels].values == row_2[self.feature_labels]).all():

                    try:
                        occurence = (number_of_clear_decision.loc[
                            number_of_clear_decision['Decision'] == row_2[
                                ['Decision']].values[0]]).values[0][1]
                    except:
                        occurence = 0

                    probability = occurence / len(self.decision_table)
                    new_df = new_df.append({
                        'Decision': row_2[['Decision']].values,
                        'Probability': probability
                    },
                        ignore_index=True)

            new_value = new_df.loc[new_df['Probability'].idxmax()]['Decision'][0]
            for idx, row_decision_table in features_decisions_occurence.iterrows():
                if (row[self.feature_labels].values == row_decision_table[self.feature_labels]).all():
                    if features_decisions_occurence.loc[features_decisions_occurence.index == idx].Decision.values[
                        0] != new_value:
                        # if self.settings.show_results:
                        # print("Current value: {}".format(features_decisions_occurence.loc[features_decisions_occurence.index == idx].Decision.values[0]))
                        # print("New value: {}".format(new_value))
                        # for idy, row_general_occurence in general_features_occurence.iterrows():
                        #     if (row_general_occurence[self.feature_labels].values == row_decision_table[self.feature_labels]).all():
                        #         if self.settings.show_results:
                        #             print(row_general_occurence.Occurence)
                        #         break
                        features_decisions_occurence.loc[idx, 'Decision'] = new_value
                        self.changed_decisions = self.changed_decisions + 1

        return features_decisions_occurence

    def inconsistencies_removing(self):
        features_decisions_occurrence = self.__get_occurrence_of_rows(self.decision_table, None)
        features_decisions_occurrence = features_decisions_occurrence.drop(['index'], axis=1)

        general_features_occurence = features_decisions_occurrence.copy()
        self.samples = sum(features_decisions_occurrence['Occurrence'])
        feature_sets_occurrence = self.__get_occurrence_of_rows(self.decision_table, ['Decision'])

        nums_of_conflicting_decisions = feature_sets_occurrence[feature_sets_occurrence['Occurrence'] > 1]

        num_of_clear_decision = self.__get_number_of_clear_decision(feature_sets_occurrence,
                                                                    features_decisions_occurrence)

        problems_to_solve = pd.merge(
            features_decisions_occurrence,
            nums_of_conflicting_decisions,
            how='inner',
            on=self.feature_labels).drop(['Occurrence_x', "Occurrence_y"], axis=1)

        features_decisions_occurence = self.solve_conflicts(
            nums_of_conflicting_decisions, problems_to_solve,
            features_decisions_occurrence, num_of_clear_decision, general_features_occurence)
        decision_table = features_decisions_occurence.drop(['Occurrence'],
                                                           axis=1).drop_duplicates(
            keep='first',
            inplace=False)

        return decision_table, self.changed_decisions
