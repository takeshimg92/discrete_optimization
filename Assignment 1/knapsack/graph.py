from node import LeftFilledNode
from collections import OrderedDict


class BnBGraph:

    def __init__(self, n_vars: int):
        assert n_vars > 0, 'Enter a number of variables > 0'
        self.n_vars = n_vars
        self.active_nodes = [LeftFilledNode(x=OrderedDict({i: None for i in range(n_vars)}),
                                            active=True)
                             ]
        self.current_node = None

        # Optimization-specific
        self.incumbent = LeftFilledNode(x=OrderedDict({i: 0 for i in range(n_vars)}))
        self.incumbent_obj = 0

    def get_number_active_nodes(self):
        return len(self.active_nodes)

    def define_next_node_to_activate(self, strategy='depth'):
        assert strategy in ['best', 'depth'], 'strategy parameter must be best or depth'
        if strategy == 'depth':
            # sort so that deepest nodes come first
            self.active_nodes = sorted(self.active_nodes, key=lambda x: x.get_depth(), reverse=True)
        else:
            raise Exception("Best strategy not yet implemented")

    def set_next_current_node_and_deactivate(self):
        if self.get_number_active_nodes() > 0:
            self.current_node = self.active_nodes.pop(0)
            self.current_node.set_inactive()

    def prune_current_node(self):
        self.current_node = None

    def set_new_incumbent(self, x: LeftFilledNode):
        self.incumbent = x

    def set_new_incumbent_obj(self, z):
        self.incumbent_obj = z

    def activate_children_current_node(self):
        children = self.current_node.get_children(activate=True)
        if children is not None:
            self.active_nodes.extend(children)


