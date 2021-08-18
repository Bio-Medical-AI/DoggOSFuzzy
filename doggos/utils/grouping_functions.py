from doggos.knowledge import LinguisticVariable, Clause, Domain
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.utils.membership_functions import generate_equal_gausses, \
    generate_progressive_gausses, \
    generate_even_triangulars, \
    generate_full_triangulars, \
    generate_even_trapezoidals, \
    generate_full_trapezoidals

from typing import Sequence, List, Tuple, Iterable, Dict
from copy import deepcopy, copy


def create_gausses_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal'):
    if mode == 'equal' or mode == 'default':
        fuzzy_sets = generate_equal_gausses(n_mfs, domain.min, domain.max - domain.precision)
    elif mode == 'progressive':
        fuzzy_sets = generate_progressive_gausses(n_mfs, domain.min, domain.max - domain.precision)
    else:
        raise NotImplemented(f'Gaussian fuzzy sets mode can be either equal or progressive, not {mode}')
    return fuzzy_sets


def create_triangular_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'even'):
    if mode == 'even' or mode == 'default':
        fuzzy_sets = generate_even_triangulars(n_mfs, domain.min, domain.max - domain.precision)
    elif mode == 'full':
        fuzzy_sets = generate_full_triangulars(n_mfs, domain.min, domain.max - domain.precision)
    else:
        raise NotImplemented(f'Triangular fuzzy sets mode can be either even or full, not {mode}')
    return fuzzy_sets


def create_trapezoidal_t1(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'even'):
    if mode == 'even' or mode == 'default':
        fuzzy_sets = generate_even_trapezoidals(n_mfs, domain.min, domain.max - domain.precision)
    elif mode == 'full':
        fuzzy_sets = generate_full_trapezoidals(n_mfs, domain.min, domain.max - domain.precision)
    else:
        raise NotImplemented(f'Trapezoidal fuzzy sets mode can be either even or full, not {mode}')
    return fuzzy_sets


def create_gausses_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal',
                       lower_scaling: float = 0.8):
    if mode == 'equal' or mode == 'default':
        upper_fuzzy_sets = generate_equal_gausses(n_mfs, domain.min, domain.max - domain.precision)
        lower_fuzzy_sets = generate_equal_gausses(n_mfs, domain.min, domain.max - domain.precision, lower_scaling)
    elif mode == 'progressive':
        upper_fuzzy_sets = generate_progressive_gausses(n_mfs, domain.min, domain.max - domain.precision)
        lower_fuzzy_sets = generate_progressive_gausses(n_mfs, domain.min, domain.max - domain.precision, lower_scaling)
    else:
        raise NotImplemented(f'Gaussian fuzzy sets mode can be either equal or progressive, not {mode}')
    functions = []
    for lmf, umf in zip(lower_fuzzy_sets, upper_fuzzy_sets):
        functions.append((lmf, umf))
    return functions


def create_triangular_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal',
                          lower_scaling: float = 0.8):
    if mode == 'even' or mode == 'default':
        upper_fuzzy_sets = generate_even_triangulars(n_mfs, domain.min, domain.max - domain.precision)
        lower_fuzzy_sets = generate_even_triangulars(n_mfs, domain.min, domain.max - domain.precision, lower_scaling)
    elif mode == 'full':
        upper_fuzzy_sets = generate_full_triangulars(n_mfs, domain.min, domain.max - domain.precision)
        lower_fuzzy_sets = generate_full_triangulars(n_mfs, domain.min, domain.max - domain.precision, lower_scaling)
    else:
        raise NotImplemented(f'Triangular fuzzy sets mode can be either even or full, not {mode}')
    functions = []
    for lmf, umf in zip(lower_fuzzy_sets, upper_fuzzy_sets):
        functions.append((lmf, umf))
    return functions


def create_trapezoidal_it2(n_mfs, domain: Domain = Domain(0, 1.001, 0.001), mode: str = 'equal',
                           lower_scaling: float = 0.8):
    if mode == 'even' or mode == 'default':
        upper_fuzzy_sets = generate_even_trapezoidals(n_mfs, domain.min, domain.max - domain.precision)
        lower_fuzzy_sets = generate_even_trapezoidals(n_mfs, domain.min, domain.max - domain.precision, lower_scaling)
    elif mode == 'full':
        upper_fuzzy_sets = generate_full_trapezoidals(n_mfs, domain.min, domain.max - domain.precision)
        lower_fuzzy_sets = generate_full_trapezoidals(n_mfs, domain.min, domain.max - domain.precision, lower_scaling)
    else:
        raise NotImplemented(f'Trapezoidal fuzzy sets mode can be either even or full, not {mode}')
    functions = []
    for lmf, umf in zip(lower_fuzzy_sets, upper_fuzzy_sets):
        functions.append((lmf, umf))
    return functions


def create_set_of_variables(ling_var_names: Iterable[str],
                            domain: Domain = Domain(0, 1.001, 0.001),
                            mf_type: str = 'gaussian',
                            n_mfs: int = 3,
                            fuzzy_set_type: str = 't1',
                            mode: str = 'default',
                            lower_scaling: float = 0.8) \
        -> Tuple[List[LinguisticVariable], Dict[str, Dict[str, FuzzySet]], Dict[str, Dict[str, Clause]]]:
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
    fuzzy_sets = {}
    ling_vars = []
    for var in ling_var_names:
        ling_vars.append(LinguisticVariable(var, deepcopy(domain)))
        fuzzy_sets[var] = {}

    if n_mfs == 2:
        grad_adjs = ['Zero', 'One']
    elif n_mfs == 3:
        grad_adjs = ['Low', 'Medium', 'High']
    elif n_mfs == 5:
        grad_adjs = ['Low', 'Medium_Low', 'Medium', 'Medium_High', 'High']
    elif n_mfs == 7:
        grad_adjs = ['Low', 'Medium_Low_Minus', 'Medium_Low', 'Medium', 'Medium_High', 'Medium_High_Plus', 'High']
    elif n_mfs == 9:
        grad_adjs = ['Low', 'Medium_Low_Minus', 'Medium_Low', 'Medium_Low_Plus', 'Medium', 'Medium_High_Minus',
                     'Medium_High', 'Medium_High_Plus', 'High']
    elif n_mfs == 11:
        grad_adjs = ['Low', 'Low_High', 'Medium_Low_Minus', 'Medium_Low', 'Medium_Low_Plus', 'Medium',
                     'Medium_High_Minus', 'Medium_High', 'Medium_High_Plus', 'High_Low', 'High']
    else:
        raise NotImplemented('n_mfs must have value from set {2, 3, 5, 7, 9, 11}')

    if fuzzy_set_type == 't1':
        if mf_type == 'gaussian':
            membership_functions = create_gausses_t1(n_mfs=n_mfs, domain=domain, mode=mode)
        elif mf_type == 'triangular':
            membership_functions = create_triangular_t1(n_mfs=n_mfs, domain=domain, mode=mode)
        elif mf_type == 'trapezoidal':
            membership_functions = create_trapezoidal_t1(n_mfs=n_mfs, domain=domain, mode=mode)
        else:
            raise NotImplemented(f"mf_type of type {mf_type} is not yet implemented.")

        for var in ling_var_names:
            for adj, mf in zip(grad_adjs, membership_functions):
                fuzzy_sets[var].update({adj: Type1FuzzySet(mf)})

    elif fuzzy_set_type == 'it2':
        if mf_type == 'gaussian':
            membership_functions = create_gausses_it2(n_mfs=n_mfs, domain=domain,
                                                      mode=mode, lower_scaling=lower_scaling)
        elif mf_type == 'triangular':
            membership_functions = create_triangular_it2(n_mfs=n_mfs, domain=domain,
                                                         mode=mode, lower_scaling=lower_scaling)
        elif mf_type == 'trapezoidal':
            membership_functions = create_trapezoidal_it2(n_mfs=n_mfs, domain=domain,
                                                          mode=mode, lower_scaling=lower_scaling)
        else:
            raise NotImplemented(f"mf_type of type {mf_type} is not yet implemented.")

        for var in ling_var_names:
            for adj, mfs in zip(grad_adjs, membership_functions):
                lmf, umf = mfs
                fuzzy_sets[var].update({adj: IntervalType2FuzzySet(lmf, umf)})

    elif fuzzy_set_type == 't2':
        raise NotImplemented('Type 2 Fuzzy Sets are not yet implemented')

    clauses = {}
    for var in ling_vars:
        clauses[var.name] = {}
        for adj in fuzzy_sets[var.name].keys():
            fuzzy_set = fuzzy_sets[var.name][adj]
            clauses[var.name][adj] = Clause(var, adj, fuzzy_set)

    return ling_vars, fuzzy_sets, clauses
