from doggos.utils.grouping_functions import create_set_of_variables
import numpy as np
import matplotlib.pyplot as plt


def main():
    ling_vars, fuzzy_sets, clauses = create_set_of_variables(['temperature', 'pressure', 'light'],
                                                             n_mfs=7,
                                                             mf_type='gaussian',
                                                             mode='equal',
                                                             fuzzy_set_type='it2',
                                                             lower_scaling=0.8)
    print(len(ling_vars))
    print(len(fuzzy_sets))
    print(clauses)
    for clause in clauses:
        clause.fuzzy_set.plot(domain=np.arange(clause.linguistic_variable.domain.min,
                                               clause.linguistic_variable.domain.max,
                                               clause.linguistic_variable.domain.precision),
                              title=clause.linguistic_variable.name + " " + clause.gradation_adjective)
        plt.show()


if __name__ == '__main__':
    main()
