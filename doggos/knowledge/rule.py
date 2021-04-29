from typing import Dict, NoReturn, List, Tuple

from doggos.fuzzy_sets.membership import MembershipDegree
from doggos.knowledge import Antecedent
from doggos.knowledge.consequents.consequent import Consequent


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

        self.__antecedent = antecedent
        self.__consequent = consequent

    @property
    def antecedent(self) -> Antecedent:
        return self.__antecedent

    @antecedent.setter
    def antecedent(self, new_antecedent: Antecedent) -> NoReturn:
        """Sets firing value and rule output to None"""
        self.__antecedent = new_antecedent

    @property
    def consequent(self) -> Consequent:
        return self.__consequent

    @consequent.setter
    def consequent(self, new_consequent: Consequent) -> NoReturn:
        """Sets rule output to None"""
        self.__consequent = new_consequent

    def output(self, features: Dict[str, float], get_firing: bool = False) -> List[MembershipDegree] or float \
                                                    or Tuple[List[MembershipDegree] or float, MembershipDegree]:
        """
        Method that is calculating firing value and output of the rule.

        :param get_firing: an optional argument. If True, it will return also firing value of rule.
        :param features: a dictionary with names of Linguistic Variables and elements of their domains.
        :return: if get_firing is False - consequent output, that is tuple of list of membership degrees or one float,
        if get_firing is True - tuple of consequent output and membership degree.
        """
        firing = self.__antecedent.firing(features)
        if get_firing:
            return self.__consequent.output(firing), firing
        return self.__consequent.output(firing)
