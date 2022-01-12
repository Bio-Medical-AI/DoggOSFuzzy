import matplotlib.pyplot as plt
import numpy as np

from doggos.fuzzy_sets import Type1FuzzySet
from doggos.utils.grouping_functions import create_gausses_t1, create_set_of_variables
from doggos.utils.membership_functions import generate_progressive_gausses

n_mfs = [11]
middle_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for n_mf in n_mfs:
    for middle_val in middle_vals:
        print(n_mf, middle_val)
        _, fuzzy_sets, _ = create_set_of_variables(['sth'], n_mfs=n_mf, mode='progressive', middle_vals=middle_val
                                                   , fuzzy_set_type='it2')
        _, axis = plt.subplots()
        for fuzzy_set in fuzzy_sets['sth'].values():
            fuzzy_set.plot(axis)

        plt.show()
