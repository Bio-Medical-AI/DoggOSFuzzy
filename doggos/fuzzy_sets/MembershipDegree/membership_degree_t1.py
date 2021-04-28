from doggos.fuzzy_sets.MembershipDegree.membership_degree import MembershipDegree


class MembershipDegreeT1(MembershipDegree):
    def __init__(self, value: float):
        self.value = value
