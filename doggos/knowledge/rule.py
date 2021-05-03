from doggos.knowledge.antecedent import Antecedent
from doggos.knowledge.consequents.consequent import Consequent


class Rule:
    """
    Class used to represent a fuzzy rule.
    It is used in fuzzy logic systems to infer an output based on input variables.
    It consists from an antecedent(premise) and a consequent.
    https://en.wikipedia.org/wiki/Fuzzy_rule
    Attributes
    --------------------------------------------
    antecedent : Antecedent
        Antecedent of the rule. Has only getter (is immutable).
    consequent : Consequent
        Consequent of the rule. Has only getter (is immutable).
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
        """
        Getter of antecedent
        :return: antecedent
        """
        return self.__antecedent

    @property
    def consequent(self) -> Consequent:
        """
        Getter of consequent
        :return: consequent
        """
        return self.__consequent
