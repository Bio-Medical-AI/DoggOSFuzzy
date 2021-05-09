from typing import Dict

import pandas as pd

from doggos.algebras import LukasiewiczAlgebra, GodelAlgebra
from doggos.fuzzy_sets import Type1FuzzySet
from doggos.knowledge import Clause, LinguisticVariable, Domain, Term, Antecedent, Rule


class RuleBuilder:

    def __init__(self, dataset: pd.DataFrame):

        self.clazz_to_rules = {}
        self.clazzes = None
        self.rule_and_functions = None
        self.dataset = dataset
        self.__features = []
        columns = list(dataset.columns)
        columns.remove('Decision')
        for feature in columns:
            self.__features.append(LinguisticVariable(str(feature), Domain(0, 1.001, 0.001)))
        self.terms = {}
        self.__clauses = []

    def induce_rules(self, fuzzy_sets):
        algebra = GodelAlgebra()
        for feature in self.__features:
            for key in fuzzy_sets:
                clause = Clause(feature, key, fuzzy_sets[key])
                self.terms[f"{feature.name}_{key}"] = Term(algebra, clause)
                self.__clauses.append(clause)
        differences = self.get_differences(self.dataset)
        classes = set(self.dataset['Decision'])
        clazz_to_records = dict([(clazz, []) for clazz in classes])
        for i, row in self.dataset.iterrows():
            clazz_to_records[row['Decision']].append((row, i))

        self.clazzes = []
        clazz_to_rules = {}
        for clazz in clazz_to_records:
            self.clazzes.append(clazz)
            row = clazz_to_records[clazz]
            clazz_to_rules[clazz] = self.build_rule(differences, row)
        self.clazz_to_rules = clazz_to_rules
        return self.clazz_to_rules

    def build_rule(self, differences, records):
        all_conjunction = []
        for r, i in records:
            conjunction = self.get_implicants(differences, r, i)
            if len(conjunction) > 0:
                all_conjunction.append(conjunction)
        all_conjunction.sort(key=lambda x: len(x))
        res_conjunction = []
        for con in all_conjunction:
            res_conjunction.append(sorted(con, key=lambda x: len(x)))
        if len(res_conjunction) == 0:
            alternative = None
        else:
            alternative = ""
            for ai, a in enumerate(res_conjunction):
                if ai != 0:
                    alternative += " | "
                alternative += "("
                for bi, b in enumerate(a):
                    if bi != 0:
                        alternative += " & "
                    alternative += "("
                    for ci, c in enumerate(b):
                        if ci != 0:
                            alternative += " | "
                        alternative += c
                    alternative += ")"
                alternative += ")"
        return eval(alternative, self.terms)

    def get_implicants(self, differences, record, index):

        diff_copy = []
        for df in differences[index]:
            if df not in diff_copy:
                diff_copy.append(df)
        diff_copy = sorted(diff_copy)
        diff_copy = sorted(diff_copy, key=lambda x: 1 / len(x))
        all_alternatives = []
        for diff in diff_copy:
            alternative = None
            for a in diff:
                if alternative is None:
                    alternative = [a + "_" + record[a]]
                elif a + "_" + record[a] not in alternative:
                    alternative.append(a + "_" + record[a])
            all_alternatives.append(alternative)
        all_alternatives.sort(key=lambda x: len(x))
        res_alternatives = []
        for alt in all_alternatives:
            res_alternatives.append(sorted(alt))

        return res_alternatives

    def get_differences(self, dataset):
        differences = dict()
        for i, row in dataset.iterrows():
            first = row
            differences[i] = []
            for j, row2 in dataset.iterrows():
                second = row2
                if i == j:
                    continue
                difference = [attr for attr in dataset if 'Decision' not in attr and first[attr] != second[attr]]
                if len(differences) != 0:
                    differences[i].append(difference)
        return differences

    @property
    def clauses(self):
        return self.__clauses

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, features):
        self.__features = features

