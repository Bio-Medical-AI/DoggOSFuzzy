from typing import List, Dict, Tuple, Any

import pandas as pd
import time

from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.induction.fuzzy_decision_table_generator import FuzzyDecisionTableGenerator
from doggos.induction.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_builder import RuleBuilder
from doggos.knowledge import Domain, Term


class InformationSystem:
    """
    Class for inducing rules antecedents from a dataset:
    https://www.mdpi.com/2076-3417/11/8/3484 - 2.2.3. Rule Induction with Information Systems

    Attributes
    --------------------------------------------
    __X: pd.DataFrame
        data representing samples

    __y: pd.Series
        target values for corresponding samples

    __feature_labels: List[str]
        labels of features to consider for calculating samples identity

    __decision_table_generator: FuzzyDecisionTableGenerator
        class used to fuzzify dataset of crisp values into fuzzy decision table for rule induction system

    __inconsistencies_remover: InconsistenciesRemover
        class for removing inconsistencies from a decision table for rule induction

    __rule_builder: RuleBuilder
        class for constructing rules from a decision table

    Methods
    --------------------------------------------
    induce_rules(self, fuzzy_sets: Dict[str, Dict[str, FuzzySet]], domain: Domain):
        performs rule induction from dataset using given fuzzy sets on given domain
    """
    __X: pd.DataFrame
    __y: pd.Series
    __feature_labels: List[str]
    __decision_table_generator: FuzzyDecisionTableGenerator
    __inconsistencies_remover: InconsistenciesRemover
    __rule_builder: RuleBuilder

    def __init__(self, X: pd.DataFrame, y: pd.Series, feature_labels: List[str]):
        """
        Creates information system for rule induction.

        :param X: data representing samples
        :param y: target values for corresponding samples
        :param feature_labels: labels of features to consider for calculating samples identity
        """
        self.__X = X
        self.__y = y
        self.__feature_labels = feature_labels
        self.__decision_table_generator = None
        self.__inconsistencies_remover = None
        self.__rule_builder = None

    def induce_rules(self, fuzzy_sets: Dict[str, Dict[str, FuzzySet]], clauses) \
            -> Tuple[Dict[Any, Term], Dict[Any, str]]:
        """
        Performs rule induction from dataset using given fuzzy sets on given domain.

        :param fuzzy_sets: dict of structure -> feature_names: fuzzy_set.name: FuzzySet
        :param domain: domain on which features are defined
        :return: terms for corresponding decisions and antecedents in string form
        """
        start = time.time()
        self.__decision_table_generator = FuzzyDecisionTableGenerator(self.__X, self.__y, fuzzy_sets, clauses)
        decision_table = self.__decision_table_generator.fuzzify()
        end = time.time()
        print('decision_table_generator: ', end - start)

        start = time.time()
        self.__inconsistencies_remover = InconsistenciesRemover(decision_table, self.__feature_labels)
        consistent_decision_table = self.__inconsistencies_remover.remove_inconsistencies()
        end = time.time()
        print('inconsistencies_remover: ', end - start)
        start = time.time()
        self.__rule_builder = RuleBuilder(consistent_decision_table, clauses)
        antecedents, str_antecedents = self.__rule_builder.induce_rules(fuzzy_sets)
        end = time.time()
        print('rule_builder: ', end - start)
        return antecedents, str_antecedents

    @property
    def rule_builder(self):
        return self.__rule_builder