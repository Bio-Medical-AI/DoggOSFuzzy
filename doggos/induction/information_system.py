from typing import List, Dict

import pandas as pd

from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.induction import FuzzyDecisionTableGenerator
from doggos.induction.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_builder import RuleBuilder
from doggos.knowledge import Domain


class InformationSystem:
    def __init__(self, X: pd.DataFrame, y: pd.Series, feature_labels: List[str]):
        self.X = X
        self.y = y
        self.feature_labels = feature_labels
        self.decision_table_generator = None
        self.inconsistencies_remover = None
        self.rule_builder = None

    def induce_rules(self, fuzzy_sets: Dict[str, Dict[str, FuzzySet]], domain: Domain):
        self.decision_table_generator = FuzzyDecisionTableGenerator(self.X, self.y, domain)
        decision_table = self.decision_table_generator.fuzzify(fuzzy_sets)

        self.inconsistencies_remover = InconsistenciesRemover(decision_table, self.feature_labels)
        consistent_decision_table = self.inconsistencies_remover.remove_inconsistencies()

        self.rule_builder = RuleBuilder(consistent_decision_table)
        rules, antecedents = self.rule_builder.induce_rules(fuzzy_sets)
        return rules, antecedents
