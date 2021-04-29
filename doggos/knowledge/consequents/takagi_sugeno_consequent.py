from typing import List, Callable

from doggos.knowledge.consequents.consequent import Consequent


class TakagiSugenoConsequent(Consequent):
    """
        Class used to represent a fuzzy rule consequent in Takagi-Sugeno model:
        http://researchhubs.com/post/engineering/fuzzy-system/takagi-sugeno-fuzzy-model.html

        Attributes
        --------------------------------------------
        __function_parameters : List[float]
            supplies parameters to consequent output function, which takes form y = ax1 + bx2 + ... + c
        __consequent_output : float
            represents output of consequent function

        Methods
        --------------------------------------------

        output: float
            value representing output from calculating consequent function with provided input parameters

        Examples:
        --------------------------------------------
        ts = TakagiSugenoConsequent([2, 10, 1])
        out = ts.output([0, 0])
        """

    def __init__(self, function_parameters: List[float]):
        """
        Create Rules Consequent used in Takagi-Sugeno Inference System.
        :param function_parameters: List[float] of parameters used for calculating output of consequent function
        """
        self.__function_parameters = function_parameters
        self.__consequent_output = 0

    def output(self, consequent_input: List[float]) -> float:
        """
        Return rule output level by calculating consequent function with inputs as variables.
        :param consequent_input: inputs of the inference system
        :return: crisp rule output value that needs to be used in aggregation process
        """
        self.__consequent_output = 0
        if len(consequent_input) == len(self.__function_parameters) - 1:
            for idx, inp in enumerate(consequent_input):
                self.__consequent_output += self.__function_parameters[idx] * consequent_input[idx]
            self.__consequent_output += self.__function_parameters[-1]
            return self.__consequent_output
        else:
            raise Exception("Number of inputs must be one less than number of consequent parameters!")
