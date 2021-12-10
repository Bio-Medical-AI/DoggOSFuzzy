import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from doggos.fuzzy_sets.interval_type2_fuzzy_set import IntervalType2FuzzySet
from doggos.induction import InformationSystem
from doggos.inference import MamdaniInferenceSystem
from doggos.inference.defuzzification_algorithms import karnik_mendel
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge import Domain, fuzzify, Rule, Clause
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.grouping_functions import create_set_of_variables



