from doggos.fuzzy_sets import Type1FuzzySet
from doggos.induction.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_builder import RuleBuilder
from doggos.knowledge import Domain
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses

import numpy as np
import pandas as pd

"""
np.random.seed(41)

columns = ['f0', 'f1', 'f2']
n_samples = 10
n_features = len(columns)
X = np.random.random_sample((n_samples, n_features))
X_frame = pd.DataFrame(data=X, columns=columns)
y = pd.Series(np.round(np.random.random_sample(n_samples)))
df = X_frame.copy()
df['Decision'] = y
print(df)
small, medium, high = generate_equal_gausses(3, 0, 1)

fuzzy_sets = {}
for column in columns:
    fuzzy_sets[column] = {'small': Type1FuzzySet(small), 'high': Type1FuzzySet(high)}

induction = InductionSystem(X_frame, y)
rules, antecedents = induction.induce_rules(fuzzy_sets)
"""

# print(rules)
# rule_base = []
# for key in rules.keys():
#    term: Term = rules[key]
#    print(term.name)
# consequent = MamdaniConsequent(Clause(LinguisticVariable(str(key), Domain(0, 1, 0.001))))
# rule_base.append(Rule(term))

# for key in antecedents.keys():
#    print(key, antecedents[key])
"""
# Example from article for rule building
np.random.seed(41)
small, medium, high = generate_equal_gausses(3, 0, 1)
columns = ['a', 'b', 'c']
fuzzy_sets = {}
for column in columns:
    fuzzy_sets[column] = {'small': Type1FuzzySet(small), 'medium': Type1FuzzySet(medium), 'high': Type1FuzzySet(high)}

data = [[1, 0, 1], [0, 0, 0], [2, 0, 1], [0, 0, 1], [1, 1, 1]]
y = [0, 1, 0, 2, 0]
map_ = {0: 'small', 1: 'medium', 2: 'high'}
non_fuzzy_frame = pd.DataFrame(data=data, columns=columns)
non_fuzzy_frame['Decision'] = y
#print('Not mapped: \n', non_fuzzy_frame)
data = list(map(lambda x: [map_[elem] for elem in x], data))
#print(data)

X = pd.DataFrame(data=data, columns=columns)
y = pd.Series(data=y)
X['Decision'] = y
#print('Decision table: \n', X)
#remover = InconsistenciesRemover(X, columns)
#consistent_decision_table, _ = remover.remove_inconsistencies()
#print('Consistent decision table: \n', consistent_decision_table)
rule_builder = RuleBuilder(X)
terms, antecedents = rule_builder.induce_rules(fuzzy_sets)
print('Antecedents')
for key in antecedents.keys():
    print(antecedents[key])
"""
# induction = InductionSystem(X, y)
# terms, antecedents = induction.induce_rules()

# Example from article for inconsistency removing
np.random.seed(41)
very_small, small, medium, high, very_high = generate_equal_gausses(5, 0, 1)
columns = ['a', 'b', 'c']
fuzzy_sets = {}
for column in columns:
    fuzzy_sets[column] = {'very_small': Type1FuzzySet(very_small), 'small': Type1FuzzySet(small), 'medium':
                          Type1FuzzySet(medium), 'high': Type1FuzzySet(high), 'very_high': Type1FuzzySet(very_high)}

data_inconsistency = [[1, 4, 1], [0, 3, 2], [1, 2, 1], [0, 2, 2],
                      [0, 2, 0], [1, 3, 1]]
y_inconsistency = [1, 2, 1, 1, 2, 1]
map_ = {0: 'very_small', 1: 'small', 2: 'medium', 3: 'high', 4: 'very_high'}
reverse_map = {k: v for k, v in zip(map_.values(), map_.keys())}
non_fuzzy_frame = pd.DataFrame(data=data_inconsistency, columns=columns)
non_fuzzy_frame['Decision'] = y_inconsistency
print('Not mapped: \n', non_fuzzy_frame)
data_inconsistency = list(map(lambda x: [map_[elem] for elem in x], data_inconsistency))
print('Mapped: \n', data_inconsistency)

X_incon = pd.DataFrame(data=data_inconsistency, columns=columns)
y_incon = pd.Series(data=y_inconsistency)
X_incon['Decision'] = y_incon
print('Decision table: \n', X_incon)
remover = InconsistenciesRemover(X_incon, ['a', 'c'])
consistent_decision_table = remover.remove_inconsistencies()
print('Consistent decision table: \n', consistent_decision_table)
# defuzzified = list(map(lambda x: [reverse_map[elem] for elem in x], consistent_decision_table.values[:, :-1]))
# final_result = pd.DataFrame(data=defuzzified)
# final_result['Decision'] = consistent_decision_table['Decision']
# print('Final result: ', final_result)

rule_builder = RuleBuilder(consistent_decision_table, Domain(0, 1.001, 0.001))
terms, antecedents = rule_builder.induce_rules(fuzzy_sets)
print('Antecedents')
for key in antecedents.keys():
    print(terms[key].name)
    print(antecedents[key])
