from collections import OrderedDict


class LeftFilledNode:
    """
    x: OrderedDict of the form {i: x_i} where i is an integer and x_i is a number in [0,1] or None
    x must be such that all its non-None values are to the left, and all the None ones are to the right (if they exist)
    """

    def __init__(self, x: OrderedDict, active=None):
        self.x = x
        self.is_active = active

    def __str__(self):
        return str(self.x)

    def get_children(self, activate=None):
        x_left, x_right = self.x.copy(), self.x.copy()
        k = self._find_child_address()

        if k is None:
            return None
        else:
            x_left[k], x_right[k] = 0, 1
            return LeftFilledNode(x_left, active=activate), LeftFilledNode(x_right, active=activate)

    def get_depth(self):
        non_none_values = [val for val in self.x.values() if val is not None]
        return len(non_none_values)

    def set_active(self):
        self.is_active = True

    def set_inactive(self):
        self.is_active = False

    def is_integer_solution(self, thresh=1e-6):
        """Checks whether the non-None entries are integers or not"""
        non_none_values = [val for val in self.x.values() if val is not None]
        int_check = [(abs(x - round(x)) < thresh) for x in non_none_values]
        return sum(int_check) == len(int_check)

    def to_list(self):
        return list(self.x.values())

    def _find_child_address(self):
        """ Finds first address for which x_i is None"""
        try:
            return list(self.x.values()).index(None)
        except ValueError:  # no index has associated value None
            return None