from typing import Dict, Tuple, Any, List

import pandas as pd
import numpy as np
import boolean

from doggos.algebras import GodelAlgebra
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.knowledge import Clause, LinguisticVariable, Domain, Term


class RuleBuilder:
    """
    Class for constructing rules from a decision table:
    https://www.mdpi.com/2076-3417/11/8/3484 - 2.2.3. Rule Induction with Information Systems: Rule Induction

    Attributes
    --------------------------------------------
    __decision_rules: Dict[Any, Term]
        rules as aggregated terms

    __dataset: pd.DataFrame
        dataset to induce rules from

    __features: List[LinguisticVariable]
        columns of the dataset as LinguisticVariables

    __clauses: List[Clause]
        clauses from features names and corresponding datasets

    __terms: Dict[str, Term]
        terms made from __clauses

    __boolean_algebra: boolean.BooleanAlgebra
        standard boolean algebra for simplifying implicants

    __target_label: str
        label of the target prediction value

    Methods
    --------------------------------------------
    induce_rules(self, fuzzy_sets: Dict[str, Dict[str, FuzzySet]]) -> Tuple[Dict[Any, Term], Dict[Any, str]]:
        induces rules from dataset with given fuzzy sets
    """
    __decision_rules: Dict[Any, Term]
    __dataset: pd.DataFrame
    __features: List[LinguisticVariable]
    __clauses: List[Clause]
    __terms: Dict[str, Term]
    __boolean_algebra: boolean.BooleanAlgebra
    __target_label: str

    def __init__(self, dataset: pd.DataFrame, clauses, target_label: str = 'Decision'):
        """
        Creates RuleBuilder for final step of rules induction from given dataset.

        :param dataset: dataset to induce rules from
        :param domain: domain of dataset features
        :param target_label: label of the target prediction value
        """
        self.__target_label = target_label
        self.__clauses = clauses
        self.__decision_rules = {}
        self.__dataset = dataset.reset_index().drop(columns=['index'])
        self.__features = []
        columns = list(dataset.columns)
        columns.remove(target_label)
        self.__terms = {}
        self.__boolean_algebra = boolean.BooleanAlgebra()

    def induce_rules(self, fuzzy_sets: Dict[str, Dict[str, FuzzySet]],
                     algebra: Algebra = GodelAlgebra()) -> Tuple[Dict[Any, Term], Dict[Any, str]]:
        """
        Induces rules from dataset with given fuzzy sets.

        :param fuzzy_sets: dictionary of fuzzy sets for dataset features
        :param algebra: represents algebra for specific fuzzy logic
        :return: rules as aggregated terms and as strings
        """
        columns = list(self.__dataset.columns)
        columns.remove(self.__target_label)
        for feature in columns:
            for key in fuzzy_sets[feature]:
                self.__terms[f"{feature}_{key}"] = Term(algebra, self.__clauses[feature][key])
        differences = self.__get_differences(self.__dataset)

        decisions = np.unique(self.__dataset[self.__target_label])
        decision_rules = {}
        string_antecedents = {}
        for decision in decisions:
            indices = np.where(self.__dataset[self.__target_label].values == decision)[0]
            idx_rows = [(index, self.__dataset.loc[index, self.__dataset.columns]) for index in indices]
            decision_rules[decision], string_antecedents[decision] = self.__build_rules(differences, idx_rows)

        self.__decision_rules = decision_rules
        return self.__decision_rules, string_antecedents

    def __build_rules(self, differences: np.ndarray, idx_rows: List[Tuple[int, pd.Series]]) -> Tuple[Term, str]:
        """
        Builds rule from given samples and discernibility matrix.

        :param differences: discernibility matrix for given dataset
        :param idx_rows: samples with corresponding indices for which to induce rule
        :return: term with aggregated firing function and string form of antecedent
        """
        all_conjunction = []
        for idx, row in idx_rows:
            conjunction = self.__get_implicants(differences, row, idx)
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
                inner_subexpression = self.__boolean_algebra.parse(inner_subexpression).simplify()
                subexpression += str(inner_subexpression)
                subexpression += ")"
                antecedent += subexpression
        if antecedent is None:
            raise Exception("Inconsistencies remover removed one class. Dataset is too inconsistent to resolve.")
        else:
            antecedent = str(self.__boolean_algebra.parse(antecedent).simplify())
            return eval(str(antecedent), self.__terms), antecedent

    def __get_implicants(self, differences: np.ndarray, row: pd.Series, index: int) -> List[List[str]]:
        """
        Produces all possible sets of alternatives or given sample from dataset.

        :param differences: discernibility matrix for given dataset
        :param row: sample from dataset for which to produce implicants
        :param index: index of the sample in dataset
        :return: list of lists of implicants from the dataset for given sample
        """
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
            for diff_feature in diff_set:
                if alternative is None:
                    alternative = [str(diff_feature) + "_" + row[diff_feature]]
                else:
                    alternative.append(str(diff_feature) + "_" + row[diff_feature])
            all_alternatives.append(alternative)
        all_alternatives.sort(key=lambda x: len(x))
        res_alternatives = []
        for alt in all_alternatives:
            res_alternatives.append(sorted(alt))

        return res_alternatives

    def __get_differences(self, dataset: pd.DataFrame) -> np.ndarray:
        """
        Creates discernibility matrix for given dataset.

        :param dataset: dataset of NxD shape
        :return: discernibility matrix of NxN shape and dtype=set
        """
        n_objects = dataset.values.shape[0]
        differences = np.zeros(shape=(n_objects, n_objects), dtype=set)
        for i, i_row in dataset.iterrows():
            for j, j_row in dataset.iterrows():
                if i != j:
                    attributes = dataset.columns
                    if i_row[self.__target_label] == j_row[self.__target_label]:
                        differences[i][j] = {}
                    else:
                        difference = {attr for attr in attributes
                                      if attr != self.__target_label and i_row[attr] != j_row[attr]}
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
