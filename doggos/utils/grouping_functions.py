from doggos.knowledge import LinguisticVariable, Clause, Domain
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.utils.membership_functions import generate_equal_gausses,\
                                              generate_progressive_gausses,\
                                              generate_even_triangulars,\
                                              generate_full_triangulars,\
                                              generate_even_trapezoidals,\
                                              generate_full_trapezoidals

from typing import Sequence, List, Tuple, Iterable
from copy import deepcopy, copy


def create_gausses_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    if mode == 'equal' or mode == 'default':
        fuzzy_sets = generate_equal_gausses(n_mfs, domain.min, domain.max)
    elif mode == 'progressive':
        fuzzy_sets = generate_progressive_gausses(n_mfs, domain.min, domain.max)
    else:
        raise NotImplemented(f'Gaussian fuzzy sets mode can be either equal or progressive, not {mode}')
    return fuzzy_sets


def create_triangular_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'even'):
    if mode == 'even' or mode == 'default':
        fuzzy_sets = generate_even_triangulars(n_mfs, domain.min, domain.max)
    elif mode == 'full':
        fuzzy_sets = generate_full_triangulars(n_mfs, domain.min, domain.max)
    else:
        raise NotImplemented(f'Triangular fuzzy sets mode can be either even or full, not {mode}')
    return fuzzy_sets


def create_trapezoidal_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'even'):
    if mode == 'even' or mode == 'default':
        fuzzy_sets = generate_even_trapezoidals(n_mfs, domain.min, domain.max)
    elif mode == 'full':
        fuzzy_sets = generate_full_trapezoidals(n_mfs, domain.min, domain.max)
    else:
        raise NotImplemented(f'Trapezoidal fuzzy sets mode can be either even or full, not {mode}')
    return fuzzy_sets


def create_gausses_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal', lower_scaling: float = 0.8):
    if mode == 'equal' or mode == 'default':
        upper_fuzzy_sets = generate_equal_gausses(n_mfs, domain.min, domain.max)
        lower_fuzzy_sets = generate_equal_gausses(n_mfs, domain.min, domain.max, lower_scaling)
    elif mode == 'progressive':
        upper_fuzzy_sets = generate_progressive_gausses(n_mfs, domain.min, domain.max)
        lower_fuzzy_sets = generate_progressive_gausses(n_mfs, domain.min, domain.max, lower_scaling)
    else:
        raise NotImplemented(f'Gaussian fuzzy sets mode can be either equal or progressive, not {mode}')
    return zip(lower_fuzzy_sets, upper_fuzzy_sets)


def create_triangular_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal', lower_scaling: float = 0.8):
    if mode == 'even' or mode == 'default':
        upper_fuzzy_sets = generate_even_triangulars(n_mfs, domain.min, domain.max)
        lower_fuzzy_sets = generate_even_triangulars(n_mfs, domain.min, domain.max, lower_scaling)
    elif mode == 'full':
        upper_fuzzy_sets = generate_full_triangulars(n_mfs, domain.min, domain.max)
        lower_fuzzy_sets = generate_full_triangulars(n_mfs, domain.min, domain.max, lower_scaling)
    else:
        raise NotImplemented(f'Triangular fuzzy sets mode can be either even or full, not {mode}')
    return zip(lower_fuzzy_sets, upper_fuzzy_sets)


def create_trapezoidal_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal', lower_scaling: float = 0.8):
    if mode == 'even' or mode == 'default':
        upper_fuzzy_sets = generate_even_trapezoidals(n_mfs, domain.min, domain.max)
        lower_fuzzy_sets = generate_even_trapezoidals(n_mfs, domain.min, domain.max, lower_scaling)
    elif mode == 'full':
        upper_fuzzy_sets = generate_full_trapezoidals(n_mfs, domain.min, domain.max)
        lower_fuzzy_sets = generate_full_trapezoidals(n_mfs, domain.min, domain.max, lower_scaling)
    else:
        raise NotImplemented(f'Trapezoidal fuzzy sets mode can be either even or full, not {mode}')
    return zip(lower_fuzzy_sets, upper_fuzzy_sets)


def create_set_of_variables(ling_var_names: Iterable[str],
                            domain: Domain = Domain(0, 1.001, 0.001),
                            mf_type: str = 'gaussian',
                            n_mfs: int = 3,
                            fuzzy_set_type: str = 't1',
                            mode: str = 'default',
                            lower_scaling: float = 0.8) \
        -> Tuple[List[LinguisticVariable], List[FuzzySet], List[Clause]]:
    """
    Creates a list of Linguistic Variables with provided names and domain. For each Linguistic Variable creates a number
    of Fuzzy Sets equal to n_mfs of type fuzzy_set_type. For each Linguistic Variable and Fuzzy Set creates a Clause.

    :param lower_scaling:
    :param mode:
    :param fuzzy_set_type:
    :param domain:
    :param ling_var_names:
    :param mf_type:
    :param n_mfs:
    :return:
    """
    ling_vars = []
    for var in ling_var_names:
        ling_vars.append(LinguisticVariable(var, deepcopy(domain)))

    fuzzy_sets = []
    if fuzzy_set_type == 't1':
        if mf_type == 'gaussian':
            membership_functions = create_gausses_t1(n_mfs=n_mfs, domain=domain, mode=mode)
        elif mf_type == 'triangular':
            membership_functions = create_triangular_t1(n_mfs=n_mfs, domain=domain, mode=mode)
        elif mf_type == 'trapezoidal':
            membership_functions = create_trapezoidal_t1(n_mfs=n_mfs, domain=domain, mode=mode)
        else:
            raise Exception(f"mf_type cannot be of type {mf_type}")

        for mf in membership_functions:
            fuzzy_sets.append(Type1FuzzySet(mf))
        base_funcs = copy(fuzzy_sets)
        for _ in range(len(ling_vars) - 1):
            fuzzy_sets.extend(deepcopy(base_funcs))

    elif fuzzy_set_type == 'it2':
        if mf_type == 'gaussian':
            membership_functions = create_gausses_it2(n_mfs=n_mfs, mode=mode)
        elif mf_type == 'triangular':
            membership_functions = create_triangular_it2(n_mfs=n_mfs, mode=mode)
        elif mf_type == 'trapezoidal':
            membership_functions = create_trapezoidal_it2(n_mfs=n_mfs, mode=mode)
        else:
            raise Exception(f"mf_type cannot be of type {mf_type}")

        for lmf, umf in membership_functions:
            fuzzy_sets.append(IntervalType2FuzzySet(lmf, umf))
        base_funcs = copy(fuzzy_sets)
        for _ in range(len(ling_vars) - 1):
            fuzzy_sets.extend(deepcopy(base_funcs))

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
        for fuzzy_set, grad_adj in zip(fuzzy_sets, grad_adjs):
            clauses.append(Clause(var, grad_adj, fuzzy_set))

    return ling_vars, fuzzy_sets, clauses
