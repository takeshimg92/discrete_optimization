from node import LeftFilledNode
import pandas as pd
from collections import OrderedDict
import numpy as np


def solve_relaxed_knapsack_pandas(node: LeftFilledNode, items_df: pd.DataFrame, capacity: int):

    # join x's with other variables by using data frames
    items_df['allocation'] = node.to_list()

    # use already allocated values to reduce total capacity
    is_allocated = items_df[~items_df['allocation'].isna()]
    to_allocate = items_df[items_df['allocation'].isna()]

    k = capacity - is_allocated[is_allocated['allocation'] == 1]['weight'].sum()  # works even if empty df
    value = is_allocated[is_allocated['allocation'] == 1]['value'].sum()

    if k < 0:
        # we are starting from an infeasible solution
        # in the knapsack, this is equivalent to a very bad z (objective)
        node_sol = node
        value = - 1e16
    else:
        if to_allocate.empty:
            node_sol = node
        else:
            # code until here was exactly the same as before.
            # sort unassigned values by density
            to_allocate = to_allocate.sort_values('density', ascending=False)

            # Create columns with cumulative weight
            to_allocate['cum_weight'] = to_allocate['weight'].cumsum()
            to_allocate['within_capacity'] = to_allocate['cum_weight'] <= k

            # three options:
            # 1. all rows w/ within_capacity = False - we cannot add anything else from the beginning!
            # 2. there is a transition, with some at True and then all at False - we can use our usual strategy
            # 3. all rows w/ within_capacity = True - trivial case where everything fits.
            if to_allocate['within_capacity'].sum() == 0:  # case 1
                first_row = to_allocate.index[0]
                to_allocate.loc[first_row, 'allocation'] = k/to_allocate.loc[first_row, 'weight']
                to_allocate['allocation'] = to_allocate['allocation'].fillna(0)

            elif to_allocate['within_capacity'].sum() == len(to_allocate):  # case 3
                to_allocate['allocation'] = 1
            else:  # case 2
                to_allocate['transition'] = np.where(
                    (~to_allocate['within_capacity']) & (to_allocate['within_capacity'].shift(1)),
                    1,
                    np.nan)
                to_allocate['allocation'] = to_allocate['transition']\
                    .fillna(method='bfill')\
                    .fillna(0)

                # fill for the allocation of the transition guy
                w = to_allocate[to_allocate['transition'] == 1]['weight'].values
                occupied_space = to_allocate[to_allocate['transition'].shift(-1) == 1]['cum_weight'].values
                to_allocate.loc[to_allocate['transition'] == 1, 'allocation'] = (k-occupied_space)/w

            joined = is_allocated.\
                append(to_allocate[is_allocated.columns]).\
                sort_values('index')

            value = (joined['allocation'] * joined['value']).sum()
            node_sol = LeftFilledNode(x=OrderedDict({
                int(x[0]): x[1] for x in joined[['index', 'allocation']].values}
            ))

    return node_sol, value


def solve_relaxed_knapsack(node: LeftFilledNode, items_df: pd.DataFrame, capacity: int):

    # join x's with other variables by using data frames
    items_df['allocation'] = node.to_list()

    # use already allocated values to reduce total capacity
    is_allocated = items_df[~items_df['allocation'].isna()]
    to_allocate = items_df[items_df['allocation'].isna()]

    k = capacity - is_allocated[is_allocated['allocation'] == 1]['weight'].sum()  # works even if empty df
    value = is_allocated[is_allocated['allocation'] == 1]['value'].sum()
    if k < 0:
        # we are starting from an infeasible solution
        # in the knapsack, this is equivalent to a very bad z (objective)
        node_sol = node
        value = - 1e16
    else:
        if to_allocate.empty:
            node_sol = node
        else:
            # solve linear relaxation for knapsack. Order values by density (in the end we will re-order by index)
            x = []
            to_allocate = to_allocate.sort_values('density', ascending=False)
            original_to_allocate = to_allocate.copy()

            # iterate to add full weights in the beginning...
            weight = 0
            for i in range(len(to_allocate)):
                index_first_row = to_allocate.index[0]
                w_to_add = to_allocate.loc[index_first_row]['weight']
                v_to_add = to_allocate.loc[index_first_row]['value']
                if weight + w_to_add <= k:
                    weight += w_to_add
                    value += v_to_add
                    to_allocate = to_allocate.drop(index_first_row)
                    x.append(1)
                else:
                    break
            # ... then add fraction of next value..
            if (weight < k) & (len(to_allocate) > 0):
                index_first_row = to_allocate.index[0]
                w_to_add = to_allocate.loc[index_first_row]['weight']
                v_to_add = to_allocate.loc[index_first_row]['value']

                prop_factor = (k-weight)/w_to_add

                weight += prop_factor * w_to_add
                value += prop_factor * v_to_add
                x.append(prop_factor)

            # ... then fill whatever is left with zeros
            if len(x) < len(original_to_allocate):
                x = x + [0]*(len(original_to_allocate) - len(x))

            original_to_allocate['allocation'] = x

            # join back with previously allocated ones
            updated_items = is_allocated.append(original_to_allocate)

            # resort by index and convert to dictionary to pass as argument to Node
            updated_items = updated_items.sort_values('index')
            node_sol = LeftFilledNode(x=OrderedDict({
                int(x[0]): x[1] for x in updated_items[['index', 'allocation']].values}
            ))

    return node_sol, value
