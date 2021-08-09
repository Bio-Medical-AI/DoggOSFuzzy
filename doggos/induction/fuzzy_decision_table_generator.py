import sys

import pandas as pd
import numpy as np
from typing import Dict, List, Iterable

from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.knowledge import Clause, LinguisticVariable, Domain


class FuzzyDecisionTableGenerator:
    """
    Class used to fuzzify dataset of crisp values into fuzzy decision table for rule induction system:
    https://www.mdpi.com/2076-3417/11/8/3484 - 2.2.3. Rule Induction with Information Systems

    Attributes
    --------------------------------------------
    __fuzzy_sets: Dict[str, Dict]
        fuzzy sets that are describing columns of given dataset

    __dataset: pd.DataFrame
        dataset to fuzzify - target column should be named 'Decision'

    __features: List[LinguisticVariable]
        linguistic variables (columns) from given dataset

    __features_clauses: Dict[str, List[Clause]]
        combination of all possible clauses from given fuzzy sets and linguistic variables

    __target_label: str
        label of the target prediction value

    Methods
    --------------------------------------------
    fuzzify(self) -> pd.DataFrame:
        performs fuzzification on given dataset and returns fuzzified decision table
    """
    __fuzzy_sets: Dict[str, Dict]
    __dataset: pd.DataFrame
    __features: List[LinguisticVariable]
    __features_clauses: Dict[str, List[Clause]]
    __target_label: str

    def __init__(self, X: pd.DataFrame, y: pd.Series, domain: Domain, target_label: str = 'Decision'):
        """
        Create FuzzyDecisionTableGenerator with given fuzzy sets, data and target values.

        :param X: data representing objects
        :param y: target values for corresponding objects
        :param domain: domain of dataset features
        :param target_label: label of the target prediction value
        """
        self.__fuzzy_sets = None
        self.__dataset = X
        for column in self.__dataset.columns:
            self.__dataset.rename({column: str(column)})

        self.__features = [LinguisticVariable(feature_name, domain)
                           for feature_name in self.__dataset.columns]
        self.__features_clauses = {feature_name: [] for feature_name in self.__dataset.columns}
        self.__target_label = target_label
        self.__dataset[target_label] = y

    def __get_highest_membership(self, feature: str, crisp: float) -> str:
        """
        Determines a name of a fuzzy set with the highest membership value for a given feature.

        :param feature: a name of a feature for which to determine a name of the highest membership set
        :param crisp: a crisp value of the feature
        :return: a name of a fuzzy set with the highest membership value for a given feature
        """
        max_feature = ''
        max_value = -sys.maxsize
        for clause in self.__features_clauses[feature]:
            clause_value = np.mean(clause.get_value(crisp))
            if clause_value > max_value:
                max_feature = clause.gradation_adjective
                max_value = clause_value
        return max_feature

    def __fuzzify_dataset(self) -> pd.DataFrame:
        """
        Replaces crisp values of a dataset with names of highest membership fuzzy sets.

        :return: dataset with crisp values replaced by names of highest membership fuzzy sets
        """
        fuzzy_dataset = pd.DataFrame(columns=self.__dataset.columns, dtype='string')
        fuzzy_dataset[self.__target_label] = pd.to_numeric(fuzzy_dataset[self.__target_label], errors='ignore')

        for row, _ in self.__dataset.iterrows():
            for column in self.__dataset.columns:
                if column == self.__target_label:
                    fuzzy_dataset.at[row, column] = self.__dataset.at[row, column]
                else:
                    fuzzy_dataset.at[row, column] = self.__get_highest_membership(column,
                                                                                  self.__dataset.at[row, column])

        return fuzzy_dataset

    def fuzzify(self, fuzzy_sets: Dict[str, Dict[str, FuzzySet]]) -> pd.DataFrame:
        """
        Performs fuzzification on given dataset and returns fuzzified decision table.

        :param fuzzy_sets: fuzzy sets that are describing columns of given dataset
        :return: dataset with crisp values replaced by names of highest membership fuzzy sets
        """
        self.__fuzzy_sets = fuzzy_sets
        for feature in self.__features:
            if feature.name != self.__target_label:
                self.__features_clauses[feature.name] = []
                for key in self.__fuzzy_sets[feature.name].keys():
                    self.__features_clauses[feature.name].append(Clause(feature,
                                                                        key,
                                                                        self.__fuzzy_sets[feature.name][key]))

        fuzzy_dataset = self.__fuzzify_dataset()
        return fuzzy_dataset
