from typing import Dict, NoReturn, List, NewType

from doggos.fuzzy_sets.membership import MembershipDegree
from doggos.knowledge.antecedent import Antecedent
from doggos.knowledge.clause import Clause
from doggos.knowledge.linguistic_variable import LinguisticVariable
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
    output(self, features: Dict[Clause, MembershipDegree], measures: Dict[LinguisticVariable, float] = None) -> OutputType:
        Calculate firing value and output of rule, then return output of type that is depending on type of consequent.

    Examples:
    --------------------------------------------
    first_rule = Rule(rule_antecedent, takagi_sugeno_rule_consequent)
    output, firing = first_rule.output(features_dictionary, measures_dictionary)
    second_rule = Rule(rule_antecedent, mamdani_rule_consequent)
    output = second_rule.output(features_dictionary)
    """

    def __init__(self, antecedent: Antecedent, consequent: Consequent):
        """
        Create a new rule with defined antecedent and consequent.

        :param antecedent: antecedent of the rule
        :param consequent: consequent of the rule
        """
        if isinstance(antecedent, Antecedent):
            self.__antecedent = antecedent
        else:
            raise ValueError("antecedent of rule must be set to instance of Antecedent class")
        if isinstance(consequent, Consequent):
            self.__consequent = consequent
        else:
            raise ValueError("consequent of rule must be set to instance of Consequent class")

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

    """Types returned by Output"""
    MamdaniOutputType = NewType('MamdaniOutputType', List[MembershipDegree])
    TakagiSugenoOutputType = NewType('TakagiSugenoOutputType', tuple[float, MembershipDegree])
    OutputType = NewType('OutputType', MamdaniOutputType or TakagiSugenoOutputType)

    def output(self, features: Dict[Clause, MembershipDegree], measures: Dict[LinguisticVariable, float] = None) -> OutputType:
        """
        Method that is calculating firing value and output of the rule.

        :param measures: a dictionary of names of linguistic variables, and crisp values of them
        :param features: a dictionary of clauses membership degree calculated for them
        :return: if consequent type is MamdaniConsequent - consequent output, that is tuple of list of membership degrees or one float,
        if consequent type is TakagiSugenoConsequent - tuple of consequent output and membership degree.
        """
        if not isinstance(self.__antecedent, Antecedent):
            raise ValueError("antecedent of rule must be set to instance of Antecedent class")
        if not isinstance(self.__consequent, Consequent):
            raise ValueError("consequent of rule must be set to instance of Consequent class")

        firing = self.__antecedent.firing(features)
        if isinstance(self.__consequent, MamdaniConsequent):
            return self.__mamdani_output(firing)
        elif isinstance(self.__consequent, TakagiSugenoConsequent):
            return self.__takagi_sugeno_output(measures, firing)
        else:
            raise NotImplementedError("Behaviour for that type of consequent is not implemented.")

    def __mamdani_output(self, firing: MembershipDegree) -> MamdaniOutputType:
        return self.__consequent.output(firing)

    def __takagi_sugeno_output(self, measures: Dict[str, float], firing: MembershipDegree) -> TakagiSugenoOutputType:
        return self.__consequent.output(measures), firing
