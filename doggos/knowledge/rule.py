from typing import Dict, NoReturn, List, Tuple

from doggos.fuzzy_sets.membership import MembershipDegree
from doggos.knowledge import Antecedent, Clause
from doggos.knowledge.consequents.consequent import Consequent
from doggos.knowledge.consequents.takagi_sugeno_consequent import TakagiSugenoConsequent
from doggos.knowledge.consequents.mamdani_consequent import MamdaniConsequent


class Rule:
    """
    Class used to represent a fuzzy rule,
    that is used in fuzzy logic systems to infer an output based on input variables.
    It consists from an antecedent(premise) and a consequent.
    https://en.wikipedia.org/wiki/Fuzzy_rule

    Attributes
    --------------------------------------------

    __antecedent : Antecedent
        Antecedent of the rule. Has getter and setter.

    __consequent : Consequent
        Consequent of the rule. Has getter and setter.

    Methods
    --------------------------------------------
    output(self, features: Dict[str, float], get_firing: bool = False) -> List[MembershipDegree] or float \
                                                    or Tuple[List[MembershipDegree] or float, MembershipDegree]:
        Calculate firing value and output of rule, then return output and optionally firing.

    Examples:
    --------------------------------------------

    """

    def __init__(self, antecedent: Antecedent, consequent: Consequent):
        """
        Create a new rule with defined antecedent and consequent.

        :param antecedent: antecedent of the rule
        :param consequent: consequent of the rule
        """
        self.__antecedent = None
        self.__consequent = None
        if isinstance(antecedent, Antecedent):
            self.__antecedent = antecedent
        if isinstance(consequent, Consequent):
            self.__consequent = consequent

    @property
    def antecedent(self) -> Antecedent:
        return self.__antecedent

    @antecedent.setter
    def antecedent(self, new_antecedent: Antecedent) -> NoReturn:
        """Sets firing value and rule output to None"""
        if isinstance(new_antecedent, Antecedent):
            self.__antecedent = new_antecedent
        else:
            raise ValueError("antecedent of rule must be set to instance of Antecedent class")

    @property
    def consequent(self) -> Consequent:
        return self.__consequent

    @consequent.setter
    def consequent(self, new_consequent: Consequent) -> NoReturn:
        """Sets rule output to None"""
        if isinstance(new_consequent, Consequent):
            self.__consequent = new_consequent
        else:
            raise ValueError("consequent of rule must be set to instance of Consequent class")

    def output(self, features: Dict[Clause, MembershipDegree]) -> List[MembershipDegree] or Tuple[float, MembershipDegree]:
        """
        Method that is calculating firing value and output of the rule.

        :param features: a dictionary with names of Linguistic Variables and elements of their domains.
        :return: if get_firing is False - consequent output, that is tuple of list of membership degrees or one float,
        if get_firing is True - tuple of consequent output and membership degree.
        """
        if not isinstance(self.__antecedent, Antecedent):
            raise ValueError("antecedent of rule must be set to instance of Antecedent class")
        if not isinstance(self.__consequent, Consequent):
            raise ValueError("consequent of rule must be set to instance of Consequent class")

        firing = self.__antecedent.firing(features)
        if isinstance(self.__consequent, MamdaniConsequent):
            return self.__mamdani_output(firing)
        elif isinstance(self.__consequent, TakagiSugenoConsequent):
            return self.__takagi_sugeno_output(features), firing
        else:
            raise NotImplementedError("Behaviour for that type of consequent is not implemented.")

    def __mamdani_output(self, firing: MembershipDegree) -> List[MembershipDegree]:
        return self.__consequent.output(firing)

    def __takagi_sugeno_output(self, features: Dict[str, float]) -> float:
        # return self.__consequent.output()
        pass
