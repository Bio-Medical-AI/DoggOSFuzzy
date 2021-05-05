from typing import Dict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from doggos.fuzzy_sets.type1_fuzzy_set import Type1FuzzySet
from doggos.knowledge import Clause, LinguisticVariable, Domain
from doggos.utils.membership_functions import triangular, trapezoidal, gaussian
from sklearn import preprocessing


class FuzzyDecisionTableGenerator:

    def __init__(self, fuzzy_sets: Dict[str, Type1FuzzySet], dataset: pd.DataFrame):
        self.__fuzzy_sets = fuzzy_sets
        self.__dataset = dataset
        self.__features = []
        for feature in dataset.columns:
            self.__features.append(LinguisticVariable(str(feature), Domain(0, 1.001, 0.001)))
        self.__features_clauses = {col: [] for col in list(dataset.columns)}


    def get_highest_membership(self, feature: str, input: float):
        max_feature = None
        max_value = 0
        for clause in self.__features_clauses[feature]:
            if clause.get_value(input) > max_value:
                max_feature = clause.gradation_adjective
                max_value = clause.get_value(input)
        return max_feature

    def fuzzify(self):
        for feature in self.__features:
            self.__features_clauses[feature] = []
            for key in self.__fuzzy_sets:
                self.__features_clauses[feature.name].append(Clause(feature, key, self.__fuzzy_sets[key]))

        fuzzy_dataset = pd.DataFrame(list([self.__dataset.columns]), dtype="string")
        fuzzy_dataset.columns = self.__dataset.columns
        fuzzy_dataset.astype('str')
        fuzzy_dataset["Value"] = pd.to_numeric(fuzzy_dataset["Value"], errors='ignore')
        for i, row in self.__dataset.iterrows():
            for f in self.__dataset:
                if f == 'Value':
                    var = self.__dataset.at[i, f]
                    fuzzy_dataset.at[i, f] = var
                else:
                    fuzzy_dataset.at[i, f] = self.get_highest_membership(f, self.__dataset.at[i, f])

        return fuzzy_dataset


small = Type1FuzzySet(triangular(0, 0.25, 0.5, 1))
medium = Type1FuzzySet(triangular(0.25, 1, 0.75, 1))
large = Type1FuzzySet(triangular(0.5, 0.75, 1., 1))
fuzzy_sets = {'small': small, 'medium': medium, 'large': large}

df = pd.read_csv('D:\\magis\\data\\DataBanknoteAuthenticationB.csv', sep=';')
df_ar = df.values
min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df_ar)
df = pd.DataFrame(df_scaled, columns=df.columns)

gen = FuzzyDecisionTableGenerator(fuzzy_sets, df)
fuzzified_dataset = gen.fuzzify()
print(fuzzified_dataset)
