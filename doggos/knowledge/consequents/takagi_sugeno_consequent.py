from typing import List, Callable, NoReturn

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

    def __init__(self, function_parameters: List[float]):
        """
        Create Rules Consequent used in Takagi-Sugeno Inference System.
        :param function_parameters: List[float] of parameters used for calculating output of consequent function
        """
        self.__function_parameters = function_parameters
        self.__consequent_output = 0

    @property
    def function_parameters(self) -> List[float]:
        """
        Getter of function parameters
        :return: function_parameters
        """
        return self.__function_parameters

    @function_parameters.setter
    def function_parameters(self, new_function_paramteres: List[float]) -> NoReturn:
        """
        Sets new list of consequent's function parameters
        :param new_function_paramteres: new list of consequent's function parameters
        :return: NoReturn
        """
        if (not isinstance(new_function_paramteres, list) or
                not all(isinstance(x, float) or isinstance(x, int) for x in new_function_paramteres)):
            raise ValueError("Takagi-Sugeno consequent parameters must be list of floats!")
        self.__function_parameters = new_function_paramteres

    def output(self, consequent_input: List[float]) -> float:
        """
        Return rule output level by calculating consequent function with inputs as variables.
        :param consequent_input: inputs of the inference system
                IMPORTANT: Number of inputs must be one less than number of function parameters!
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


tss = TakagiSugenoConsequent([1, 2, 3])
tss.output([0.5, 0.2]) == 3.9
tss.function_parameters = [0.1, 0.1, 1.]
tss.output([0.99, 0.88]) == 1