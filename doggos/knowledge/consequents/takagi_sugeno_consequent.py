from typing import List, Callable, NoReturn, Dict

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
        out = ts.output([0.5, 0.9])
        """

    def __init__(self, function_parameters: Dict[str, float]):
        """
        Create Rules Consequent used in Takagi-Sugeno Inference System.
        :param function_parameters: Dict[str, float] of input LinguisticVariable name and parameter used for calculating
         output of consequent function
        """
        self.__function_parameters = function_parameters
        self.__consequent_output = 0

    @property
    def function_parameters(self) -> Dict[str, float]:
        """
        Getter of function parameters
        :return: function_parameters
        """
        return self.__function_parameters

    @function_parameters.setter
    def function_parameters(self, new_function_parameters: Dict[str, float]) -> NoReturn:
        """
        Sets new list of consequent's function parameters
        :param new_function_paramteres: new dictionary of consequent's function parameters
        :return: NoReturn
        """
        if (not isinstance(new_function_parameters, dict) or
                not all(isinstance(x, float) or isinstance(x, int) for x in new_function_parameters.values()) or
                not all(isinstance(x, str) for x in new_function_parameters.keys())):
            raise ValueError("Takagi-Sugeno consequent parameters must be Dict[str, float]!")
        self.__function_parameters = new_function_parameters

    def output(self, consequent_input: Dict[str, float]) -> float:
        """
        Return rule output level by calculating consequent function with inputs as variables.
        :param consequent_input: inputs of the inference system in Dict[str, float], which reflects input linguistic
                variable name and value
                IMPORTANT: Number of inputs must be one less than number of function parameters!
                           Last parameter should have key 'const' as it reflects form of output function.
        :return: crisp rule output value that needs to be used in aggregation process
        """
        self.__consequent_output = 0
        if all(key in self.__function_parameters.keys() for key in list(consequent_input.keys()) + ['const']):
            for key in consequent_input:
                self.__consequent_output += consequent_input[key] * self.__function_parameters[key]
            if 'const' in self.__function_parameters.keys():
                self.__consequent_output += self.__function_parameters['const']
            return self.__consequent_output
        else:
            raise Exception("Function parameters contain value for input which was not provided!")
