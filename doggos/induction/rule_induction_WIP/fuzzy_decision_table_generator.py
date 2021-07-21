import pandas as pd
from typing import Dict, List
from doggos.knowledge import Clause, LinguisticVariable, Domain


class FuzzyDecisionTableGenerator:
    """
    Class used to fuzzify dataset of crisp values into fuzzy decision table for rule induction system

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

    Methods
    --------------------------------------------
    get_highest_membership(self, feature: str, input: float) -> str:
        returns highest membership for given feature's input value

    fuzzify(self):
        performs fuzzification on given dataset and returns fuzzified decision table
    """
    __fuzzy_sets: Dict[str, Dict]
    __dataset: pd.DataFrame
    __features: List[LinguisticVariable]
    __features_clauses: Dict[str, List[Clause]]

    def __init__(self, fuzzy_sets: Dict[str, Dict], X: pd.DataFrame, y: pd.Series):
        self.__fuzzy_sets = fuzzy_sets
        self.__dataset = X
        self.__dataset['Decision'] = y
        self.__features = [LinguisticVariable(str(feature), Domain(0, 1.001, 0.001))
                           for feature in self.__dataset.columns]
        self.__features_clauses = {col: [] for col in list(self.__dataset.columns)}

    def get_highest_membership(self, feature: str, input: float) -> str:
        max_feature = None
        max_value = 0
        for clause in self.__features_clauses[feature]:
            if clause.get_value(input) > max_value:
                max_feature = clause.gradation_adjective
                max_value = clause.get_value(input)
        return max_feature

    def fuzzify(self):
        for feature in self.__features:
            if feature.name == 'Decision':
                continue
            self.__features_clauses[feature.name] = []
            for key in self.__fuzzy_sets:
                self.__features_clauses[feature.name].append(Clause(feature, key, self.__fuzzy_sets[feature.name][key]))

        fuzzy_dataset = pd.DataFrame(list([self.__dataset.columns]), dtype="string")
        fuzzy_dataset.columns = self.__dataset.columns
        fuzzy_dataset.astype('str')
        fuzzy_dataset["Decision"] = pd.to_numeric(fuzzy_dataset["Decision"], errors='ignore')
        for i, row in self.__dataset.iterrows():
            for f in self.__dataset:
                if f == 'Decision':
                    var = self.__dataset.at[i, f]
                    fuzzy_dataset.at[i, f] = var
                else:
                    fuzzy_dataset.at[i, f] = self.get_highest_membership(f, self.__dataset.at[i, f])

        fuzzy_dataset.drop(index=fuzzy_dataset[fuzzy_dataset['Decision'] == 'Decision'].index, inplace=True)
        return fuzzy_dataset

