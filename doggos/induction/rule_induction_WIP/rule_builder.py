import pandas as pd
import numpy as np
import boolean

from doggos.algebras import GodelAlgebra
from doggos.knowledge import Clause, LinguisticVariable, Domain, Term


class RuleBuilder:

    def __init__(self, dataset: pd.DataFrame):

        self.__decision_rules = {}
        self.__decisions = None
        self.__dataset = dataset.reset_index().drop(columns=['index'])
        self.__features = []
        columns = list(dataset.columns)
        columns.remove('Decision')
        for feature in columns:
            self.__features.append(LinguisticVariable(str(feature), Domain(0, 1.001, 0.001)))
        self.__terms = {}
        self.__clauses = []
        self.boolean_algebra = boolean.BooleanAlgebra()

    def induce_rules(self, fuzzy_sets):
        algebra = GodelAlgebra()
        for feature in self.__features:
            for key in fuzzy_sets[feature.name]:
                clause = Clause(feature, key, fuzzy_sets[feature.name][key])
                self.__terms[f"{feature.name}_{key}"] = Term(algebra, clause)
                self.__clauses.append(clause)
        differences = self.get_differences(self.__dataset)
        #print('Differences:\n', differences)

        decisions = np.unique(self.__dataset['Decision'])
        rows_with_decisions = {}
        for decision in decisions:
            indices = np.where(self.__dataset['Decision'].values == decision)[0]
            idx_rows = [(index, self.__dataset.loc[index, self.__dataset.columns]) for index in indices]
            rows_with_decisions[decision] = idx_rows

        decision_rules = {}
        string_antecedents = {}
        for decision in decisions:
            idx_rows = rows_with_decisions[decision]
            decision_rules[decision], string_antecedents[decision] = self.simplified_build_rule(differences, idx_rows)
        self.__decision_rules = decision_rules
        return self.__decision_rules, string_antecedents

    def build_rule(self, differences, idx_rows):
        all_conjunction = []
        for idx, row in idx_rows:
            #print(row)
            conjunction = self.get_implicants(differences, row, idx)
            if len(conjunction) > 0:
                all_conjunction.append(conjunction)
        all_conjunction.sort(key=lambda x: len(x))
        res_conjunction = []
        for con in all_conjunction:
            res_conjunction.append(sorted(con, key=lambda x: len(x)))
        if len(res_conjunction) == 0:
            antecedent = None
        else:
            antecedent = ""
            for ai, a in enumerate(res_conjunction):
                if ai != 0:
                    antecedent += " | "
                antecedent += "("
                for bi, b in enumerate(a):
                    if bi != 0:
                        antecedent += " & "
                    antecedent += "("
                    for ci, c in enumerate(b):
                        if ci != 0:
                            antecedent += " | "
                        if c == 'F0_F0':
                            print(c)
                        antecedent += c
                    antecedent += ")"
                antecedent += ")"
        return eval(antecedent, self.__terms), antecedent

    def simplified_build_rule(self, differences, idx_rows):
        #print('Implicants:\n')
        all_conjunction = []
        for idx, row in idx_rows:
            #print(row)
            conjunction = self.get_implicants(differences, row, idx)
            if len(conjunction) > 0:
                all_conjunction.append(conjunction)
        all_conjunction.sort(key=lambda x: len(x))
        res_conjunction = []
        for con in all_conjunction:
            res_conjunction.append(sorted(con, key=lambda x: len(x)))
        if len(res_conjunction) == 0:
            antecedent = None
        else:
            antecedent = ""
            for ai, a in enumerate(res_conjunction):
                subexpression = ""
                if ai != 0:
                    subexpression += " | "
                subexpression += "("
                inner_subexpression = ""
                for bi, b in enumerate(a):
                    if bi != 0:
                        inner_subexpression += " & "
                    inner_subexpression += "("
                    for ci, c in enumerate(b):
                        if ci != 0:
                            inner_subexpression += " | "
                        inner_subexpression += c
                    inner_subexpression += ")"
                #print('Before simplify: ', inner_subexpression)
                inner_subexpression = self.boolean_algebra.parse(inner_subexpression).simplify()
                #print(inner_subexpression)
                #print('After simplify: ', inner_subexpression)
                subexpression += str(inner_subexpression)
                subexpression += ")"
                antecedent += subexpression
        #print('Before simplify: ', antecedent)
        antecedent = str(self.boolean_algebra.parse(antecedent).simplify())
        #print('After simplify', antecedent)
        return eval(str(antecedent), self.__terms), antecedent

    def get_implicants(self, differences, row, index):

        diffs_for_row = []
        for diffs_set in differences[index]:
            if len(diffs_set) > 0:
                contains = np.array([np.array_equal(diffs_set, elem) for elem in diffs_for_row]).any()
                if not contains:
                    diffs_for_row.append(diffs_set)

        diffs_for_row = sorted(diffs_for_row, key=lambda x: len(x), reverse=True)
        all_alternatives = []
        for diff_set in diffs_for_row:
            alternative = None
            #print(diff_set)
            for diff_feature in diff_set:
                #print(diff_feature)
                #print(row[diff_feature].values)
                if alternative is None:
                    alternative = [str(diff_feature) + "_" + row[diff_feature]]
                else:
                    alternative.append(str(diff_feature) + "_" + row[diff_feature])
            all_alternatives.append(alternative)
        all_alternatives.sort(key=lambda x: len(x))
        res_alternatives = []
        #print(all_alternatives)
        for alt in all_alternatives:
            #print(alt)
            res_alternatives.append(sorted(alt))

        return res_alternatives

    def get_differences(self, dataset):
        n_objects = dataset.values.shape[0]
        differences = np.zeros(shape=(n_objects, n_objects), dtype=set)
        for i, i_row in dataset.iterrows():
            for j, j_row in dataset.iterrows():
                if i != j:
                    attributes = dataset.columns
                    if i_row['Decision'] == j_row['Decision']:
                        differences[i][j] = {}
                    else:
                        difference = {attr for attr in attributes if attr != 'Decision' and i_row[attr] != j_row[attr]}
                        differences[i][j] = difference
                else:
                    differences[i][j] = {}

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

