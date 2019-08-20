#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from graph import BnBGraph
from timers import timer
from knapsack_relaxation import solve_relaxed_knapsack, solve_relaxed_knapsack_pandas
from time import time

import numpy as np
import pandas as pd

Item = namedtuple("Item", ['index', 'value', 'weight', 'density'])


def inputs_to_item_list(lines, item_count):
    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        v_i, w_i = line.split()
        v_i, w_i = int(v_i), int(w_i)
        rho_i = v_i / w_i
        items.append(Item(i - 1, v_i, w_i, rho_i))

    return items


@timer
def exhaustive_search(capacity, item_count, items):

    from itertools import product
    import operator

    # transform values, weights in vectors; create full 2ˆn search space
    values_vec, weight_vec = np.array(items['value']), np.array(items['weight'])
    search_space = product(*([0, 1] for i in range(item_count)))

    def dot_prod(x: tuple, v: np.array):
        return np.dot(np.array(x), v)

    # create search dict
    search_dict = {x: dot_prod(x, values_vec) for x in search_space if dot_prod(x, weight_vec) <= capacity}

    # get maximum value and the corresponding assignments
    allocations, value = max(search_dict.items(), key=operator.itemgetter(1))

    return value, list(allocations)


@timer
def branch_and_bound(capacity, item_count, items, time_limit_hours=1e5):

    graph = BnBGraph(item_count)
    iter = 0
    time_limit_seconds = time_limit_hours*60*60
    t0 = time()
    while (graph.get_number_active_nodes() > 0) & (time() - t0 <= time_limit_seconds):
        iter += 1
        graph.define_next_node_to_activate()
        graph.set_next_current_node_and_deactivate()
        x, z = solve_relaxed_knapsack(graph.current_node, items, capacity)
        if graph.incumbent_obj >= z:
            graph.prune_current_node()
        elif (graph.incumbent_obj < z) & x.is_integer_solution():
            print(f'New incumbent: {z} (at iteration {iter})')
            graph.set_new_incumbent(x)
            graph.set_new_incumbent_obj(z)
            graph.prune_current_node()
        else:
            graph.activate_children_current_node()
    if time() - t0 > time_limit_seconds:
        print(f"Time limit of {time_limit_hours} hours exceeded.")
    print(f"Total of nodes visited: {iter}")

    value = int(graph.incumbent_obj)
    allocations = [int(x) for x in graph.incumbent.to_list()]

    return value, allocations

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    print("="*30 + "\n" + f"Solving knapsack problem with\n  K = {capacity}, n = {item_count}\n" + "="*30)

    items = pd.DataFrame(inputs_to_item_list(lines, item_count))

    if item_count <= 24:
        print("Solving by exhaustive search:")
        value, taken = exhaustive_search(capacity, item_count, items)
    else:
        print("Solving by branch and bound (depth-first):")
        value, taken = branch_and_bound(capacity, item_count, items, time_limit_hours=4.5)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
