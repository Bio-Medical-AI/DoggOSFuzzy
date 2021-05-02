from typing import NoReturn, Dict

from doggos.knowledge import LinguisticVariable
from doggos.knowledge.consequents.consequent import Consequent


class TakagiSugenoConsequent(Consequent):
    """
        Class used to represent a fuzzy rule consequent in Takagi-Sugeno model:
        http://researchhubs.com/post/engineering/fuzzy-system/takagi-sugeno-fuzzy-model.html

        Attributes
        --------------------------------------------
        __function_parameters : Dict[str, float]
            supplies parameters to consequent output function, which takes form y = ax1 + bx2 + ... + c
        __consequent_output : float
            represents output of consequent function

        Methods
        --------------------------------------------

        output: float
            value representing output from calculating consequent function with provided input parameters

        Examples:
        --------------------------------------------
        ts = TakagiSugenoConsequent({'F1': 0.1, 'const': -2.5})
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('F1', domain)
        output = ts.output({ling_var: 1}) == -2.4)
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
        :param new_function_parameters: new dictionary of consequent's function parameters
        :return: NoReturn
        """
        if (not isinstance(new_function_parameters, dict) or
                not all(isinstance(x, float) or isinstance(x, int) for x in new_function_parameters.values()) or
                not all(isinstance(x, str) for x in new_function_parameters.keys())):
            raise ValueError("Takagi-Sugeno consequent parameters must be Dict[str, float]!")
        self.__function_parameters = new_function_parameters

    def output(self, consequent_input: Dict[LinguisticVariable, float]) -> float:
        """
        Return rule output level by calculating consequent function with inputs as variables.
        :param consequent_input: inputs of the inference system in Dict[LinguisticVariable, float], which reflects
         input feature name and value
                IMPORTANT: Each of input variable which will be considered in inference process, needs to have
                corresponding function parameter provided.
        :return: crisp rule output value that needs to be used in aggregation process
        """
        self.__consequent_output = 0
        if all(key in self.__function_parameters.keys() for key in [x.name for x in consequent_input.keys()] +
                                                                   ['const']):
            for key in consequent_input:
                self.__consequent_output += consequent_input[key] * self.__function_parameters[key.name]
            if 'const' in self.__function_parameters.keys():
                self.__consequent_output += self.__function_parameters['const']
            return self.__consequent_output
        else:
            raise Exception("Function parameters contain value for input which was not provided!")
