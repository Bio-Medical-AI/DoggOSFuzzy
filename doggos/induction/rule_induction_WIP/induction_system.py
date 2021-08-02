from typing import Dict

import pandas as pd

from doggos.induction import FuzzyDecisionTableGenerator
from doggos.induction.rule_induction_WIP.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_induction_WIP.rule_builder import RuleBuilder


class InductionSystem:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.feature_labels = list(X.columns)
        self.decision_table_generator = None
        self.inconsistencies_remover = None
        self.rule_builder = None

    def induce_rules(self, fuzzy_sets):
        self.decision_table_generator = FuzzyDecisionTableGenerator(fuzzy_sets, self.X, self.y)
        decision_table = self.decision_table_generator.fuzzify()
        print('Decision table: \n', decision_table)
        self.inconsistencies_remover = InconsistenciesRemover(decision_table, self.feature_labels)
        consistent_decision_table, _ = self.inconsistencies_remover.remove_inconsistencies()
        print('\nConsistent decision table: \n', consistent_decision_table)
        self.rule_builder = RuleBuilder(consistent_decision_table)
        rules, antecedents = self.rule_builder.induce_rules(fuzzy_sets)
        for term_key in rules.keys():
            print('Term ' + str(term_key) + ": \n" + rules[term_key].name)
            print('Antecedent: \n', antecedents[term_key])
        return rules, antecedents
