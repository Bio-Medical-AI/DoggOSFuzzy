from doggos.knowledge import LinguisticVariable, Clause, Domain
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.utils.membership_functions import generate_equal_gausses,\
                                              generate_progressive_gausses,\
                                              generate_even_triangulars,\
                                              generate_full_triangulars,\
                                              generate_even_trapezoidals,\
                                              generate_full_trapezoidals

from typing import Sequence, List, Tuple


def create_gausses_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    if mode == 'equal':
        fuzzy_sets = generate_equal_gausses(n_mfs, domain.min, domain.max)
    elif mode == 'progressive':
        fuzzy_sets = generate_progressive_gausses(n_mfs, domain.min, domain.max)
    else:
        raise NotImplemented(f'Gaussian fuzzy sets mode can be either equal or progressive, not {mode}')
    return fuzzy_sets


def create_triangular_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    fuzzy_sets = generate_even_triangulars(n_mfs, domain.min, domain.max)
    return fuzzy_sets


def create_trapezoidal_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    fuzzy_sets = generate_even_trapezoidals(n_mfs, domain.min, domain.max)
    return fuzzy_sets


def create_gausses_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    raise NotImplemented('Generating it2 functions is not yet implemented')


def create_triangular_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    raise NotImplemented('Generating it2 functions is not yet implemented')


def create_trapezoidal_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    raise NotImplemented('Generating it2 functions is not yet implemented')


def create_set_of_variables(ling_var_names: Sequence[str],
                            domain: Domain = Domain(0, 1.001, 0.001),
                            mf_type: str = 'gaussian',
                            n_mfs: int = 3,
                            fuzzy_set_type: str = 't1') \
        -> Tuple[List[LinguisticVariable], List[FuzzySet], List[Clause]]:
    """
    Creates a list of Linguistic Variables with provided names and domain. For each Linguistic Variable creates a number
    of Fuzzy Sets equal to n_mfs of type fuzzy_set_type. For each Linguistic Variable and Fuzzy Set creates a Clause.
    :param fuzzy_set_type:
    :param domain:
    :param ling_var_names:
    :param mf_type:
    :param n_mfs:
    :return:
    """
    ling_vars = []
    for var in ling_var_names:
        ling_vars.append(LinguisticVariable(var, domain))

    fuzzy_sets = []
    if fuzzy_set_type == 't1':
        if mf_type == 'gaussian':
            fuzzy_sets = create_gausses_t1(n_mfs=n_mfs)
        elif mf_type == 'triangular:':
            fuzzy_sets = create_triangular_t1(n_mfs=n_mfs)
        elif mf_type == 'trapezoidal':
            fuzzy_sets = create_trapezoidal_t1(n_mfs=n_mfs)
        else:
            raise Exception(f"mf_type cannot be of type {mf_type}")
    elif fuzzy_set_type == 'it2':
        if mf_type == 'gaussian':
            fuzzy_sets = create_gausses_it2(n_mfs=n_mfs)
        elif mf_type == 'triangular:':
            fuzzy_sets = create_triangular_it2(n_mfs=n_mfs)
        elif mf_type == 'trapezoidal':
            fuzzy_sets = create_trapezoidal_it2(n_mfs=n_mfs)
        else:
            raise Exception(f"mf_type cannot be of type {mf_type}")
    elif fuzzy_set_type == 't2':
        raise NotImplemented('Type 2 Fuzzy Sets are not yet implemented')

    if n_mfs == 3:
        grad_adjs = ['Low', 'Medium', 'High']
    elif n_mfs == 5:
        grad_adjs = ['Low', 'Medium_Low' 'Medium', 'Medium_High', 'High']
    elif n_mfs == 7:
        grad_adjs = ['Low', 'Medium_Low -', 'Medium_Low', 'Medium', 'Medium_High', 'Medium_High +', 'High']
    elif n_mfs == 9:
        grad_adjs = ['Low', 'Medium_Low -', 'Medium_Low', 'Medium_Low +', 'Medium', 'Medium_High -', 'Medium_High',
                     'Medium_High +', 'High']
    elif n_mfs == 11:
        grad_adjs = ['Low', 'Low_High', 'Medium_Low -', 'Medium_Low', 'Medium_Low +', 'Medium', 'Medium_High -',
                     'Medium_High', 'Medium_High +', 'High_Low', 'High']
    else:
        raise NotImplemented('n_mfs must have value from set {3, 5, 7, 9, 11}')

    clauses = []
    for var in ling_vars:
        for fuzzy_set, grad_adjs in zip(fuzzy_sets, grad_adjs):
            clauses.append(Clause(var, grad_adjs, fuzzy_set))

    return ling_vars, fuzzy_sets, clauses
