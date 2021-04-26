from typing import Tuple, List
import numpy as np

from doggos.knowledge.consequents.consequent import Consequent


class TakagiSugenoConsequent(Consequent):
    """
        Class used to represent a fuzzy rule consequent in Takagi-Sugeno model:
        http://researchhubs.com/post/engineering/fuzzy-system/takagi-sugeno-fuzzy-model.html

        Attributes
        --------------------------------------------
        __function_parameters : List[float]
            supplies parameters to consequent output function, which takes form y = ax1 + bx2 + ... + c

        __rule_output: float
            value representing output from consequent function

        Methods
        --------------------------------------------
        calculate_cut(rule_firing: Tuple[float, ...] or float) -> Tuple[List[float]] or List[float]
            process rule output with firing strength

        Examples:
        --------------------------------------------

        """

    def calculate_cut(self, rule_firing: Tuple[float, ...] or float) -> Tuple[List[float]] or List[float]:
        pass

    def __init__(self, function_parameters: List[float]):
        """
        Create Rules Consequent used in Takagi-Sugeno Inference System. Provided Clause holds fuzzy set describing Consequent
        and Linguistic Variable which value user wants to compute.
        :param clause: Clause holding fuzzy set and linguistic variable
        """
        self.__function_parameters = function_parameters
        self.__rule_output = None

    def calculate_rule_output(self, inputs: List[float]) -> float:
        """
        Return rule output level by calculating consequent function with inputs as variables.
        :param inputs: list of inputs for which rule is triggered.
        :return: crisp rule output value that needs to be used in aggregation process
        """
        if len(inputs) == len(self.__function_parameters):
            for idx, inp in enumerate(inputs):
                self.__rule_output += self.__function_parameters[idx] * inputs[idx]
            self.__rule_output += self.__function_parameters[-1]
            return self.__rule_output
        else:
            raise Exception

    def return_rule_result(self, rule_firing: float or Tuple[float, float]) -> Tuple[float, float] or Tuple[float, Tuple[float, float]]:
        """
        Return rule output level along with corresponding firing strength
        :param rule_firing: crisp value or interval, calculated firing strength of the rule
        :return: crisp rule output and firing strength that needs to be used in aggregation process
        """
        if isinstance(rule_firing, float):
            return self.__rule_output, rule_firing
        elif isinstance(rule_firing, tuple):
            if len(rule_firing) == 2:
                return self.__rule_output, rule_firing
        else:
            raise Exception
