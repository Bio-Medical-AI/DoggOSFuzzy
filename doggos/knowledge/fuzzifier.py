import pandas as pd
import numpy as np
from typing import Iterable, Dict

from doggos.knowledge import Clause


def fuzzify(dataset: pd.DataFrame, clauses: Iterable[Clause]) -> Dict[Clause, np.ndarray]:
    """
    Calculates membership degrees of features of the dataset for each clause.

    Example:\n
    import pandas as pd\n
    from doggos.knowledge import Clause, LinguisticVariable, Domain\n
    from doggos.fuzzy_sets import Type1FuzzySet\n
    df = pd.DataFrame({'fire': [1.0, 2.3], 'air': [0, 2]})\n
    lingustic_variable_1 = LinguisticVariable('fire', Domain(0, 5, 0.1))\n
    lingustic_variable_2 = LinguisticVariable('air', Domain(0, 5, 0.1))\n
    clause1 = Clause(lingustic_variable_1, 'high', Type1FuzzySet(lambda x: 1))\n
    clause2 = Clause(lingustic_variable_1, 'low', Type1FuzzySet(lambda x: 0.8))\n
    clause3 = Clause(lingustic_variable_2, 'high', Type1FuzzySet(lambda x: 0.9))\n
    clause4 = Clause(lingustic_variable_2, 'low', Type1FuzzySet(lambda x: 0.7))\n
    fuzzified = fuzzify(df, [clause1, clause2, clause3, clause4])\n

    Clause fire is high: [1, 1]\n
    Clause fire is low: [0.8, 0.8]\n
    Clause air is high: [0.9, 0.9]\n
    Clause air is low: [0.7, 0.7]

    :param dataset: input dataset
    :param clauses: iterable structure of clauses
    :return: membership degrees of features of the dataset for each clause
    """
    fuzzified_dataset = dict()
    for category in clauses.keys():
        for adj in clauses[category].keys():
            data = dataset[category].values
            fuzzified_dataset[clauses[category][adj]] = clauses[category][adj].get_value(data)
    return fuzzified_dataset
