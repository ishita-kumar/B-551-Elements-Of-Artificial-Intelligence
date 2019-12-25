#!/usr/local/bin/python3
#
# choose_team.py : Choose a team of maximum skill under a fixed budget
#
# Code by: Ishita Kumar, Hrishikesh Paul, Rushabh Shah (ishkumar@iu.edu, hrpaul@iu.edu, shah12@iu.edu)
#
# Based on skeleton code by D. Crandall, September 2019
#
import sys


# Node stores the profit, level, weight, bound and name of all nodes
# that is used to compute the optimal solution for choosing a team
class Node:
    def __init__(self, level=None, profit=None, bound=None, weight=None, name=[]):
        self.profit = profit
        self.level = level
        self.weight = weight
        self.bound = bound
        self.name = name

    def __eq__(self, other):
        self.other = other


def load_people(filename):
    people = {}
    with open(filename, "r") as file:
        for line in file:
            l = line.split()
            people[l[0]] = [float(i) for i in l[1:]]
    return people


# The following code for the branch and bound methodology in 0/1 knapsack inspired by
# https://www.geeksforgeeks.org/implementation-of-0-1-knapsack-using-branch-and-bound/
# starts here
# Computing an upper bound using greedy search
def get_bound(node, sorted_people):
    if node.weight >= budget:
        return 0

    profit_bound = node.profit
    j = node.level + 1
    total_weight = node.weight
    while j < len(sorted_people) and (total_weight + sorted_people[j][1][1] <= budget):
        total_weight = total_weight + sorted_people[j][1][1]
        profit_bound = profit_bound + sorted_people[j][1][0]
        j = j + 1
    # checking if the index value and budget parameters exceed
    if j < len(sorted_people):
        profit_bound = profit_bound + (budget - total_weight) * (
            sorted_people[j][1][0] / sorted_people[j][1][1]
        )
    # return profit value of sub-tree
    return profit_bound


def approx_solve(people, budget):
    # sorting the list in a decreasing order of skills/profit
    sorted_people = sorted(
        people.items(), key=lambda x: x[1][0] / x[1][1], reverse=True
    )
    max_profit = 0
    fringe = []
    names = []
    initial_node = Node()
    initial_node.level = -1
    initial_node.profit = initial_node.weight = 0
    # creating a queue with initial node values
    fringe.append(initial_node)

    while len(fringe) > 0:
        current_node = fringe.pop(0)
        new_node = Node()
        # Starting Node
        if current_node.level == -1:
            new_node.level = 0
        # Leaf Node
        if current_node.level == len(sorted_people) - 1:
            continue
        # Computing profit and weight of child nodes
        new_node.level = current_node.level + 1
        new_node.weight = current_node.weight + sorted_people[new_node.level][1][1]
        new_node.profit = current_node.profit + sorted_people[new_node.level][1][0]
        new_node.name = list(current_node.name)
        new_node.name.append(sorted_people[new_node.level][0])

        if new_node.weight <= budget and new_node.profit > max_profit:
            names = new_node.name
            max_profit = new_node.profit
        # calculate upper bound on profit
        new_node.bound = get_bound(new_node, sorted_people)

        if new_node.bound > max_profit:
            fringe.append(new_node)

        new_node = Node(
            current_node.level + 1,
            current_node.profit,
            get_bound(new_node, sorted_people),
            current_node.weight,
            list(current_node.name),
        )

        # append into fringe only if its value is greater than the max profit value
        if new_node.bound > max_profit:
            fringe.append(new_node)

    return max_profit, names


# Inspired code ends here

if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise Exception('Error: expected 2 command line arguments')

    budget = float(sys.argv[2])
    people = load_people(sys.argv[1])
    solution = approx_solve(people, budget)

    if solution[0] > 0:
        print(
            f'Found a group with {len(solution[1])} people costing {budget} with '
            f'total skill {solution[0]}'
        )
        for name in solution[1]:
            print(f'{name} {float(1.00000)}')
    else:
        print('Inf')
