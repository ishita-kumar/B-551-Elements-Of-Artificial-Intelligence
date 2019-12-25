import numpy as np
import pickle
import time


class Node:
    """
    This is the class Node that contains the decision at every level, and pointers
    to the respective true and false branches
    """

    def __init__(self, decision, true_branch, false_branch):
        """
        Class constructor
        :param decision: Soft of like a question to determine the classification
        :param true_branch: Reference to the tree/node that satisfies the decision
        :param false_branch: Reference to the tree/node that does not satisfies
        the decision
        """
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.decision = decision

    def __repr__(self):
        return f'{self.decision}'


def calculate_impurity(data):
    """
    Calculates the impurity, called gini impurity that measures the probability
    of a label being incorrectly classified.
    :param data: The dataset
    :return: The impurity
    """

    mis = 1

    # Count the classified labels
    label_count = {}
    for row in data:
        # Extract label as the 193rd index is the class label
        label = row[-1]
        label_count[label] = label_count.get(label, 0) + 1

    # User formula to calculate the impurity

    for label in label_count:
        mis -= (label_count[label] / float(len(data))) ** 2
    return mis


def calculate_info_gain(false, true, entropy):
    """
    Calculates the decrease in entropy by the true and false branches
    :param false: False branch
    :param true: True branch
    :param entropy: Parent entropy
    :return: Decrease in entropy
    """

    # Calculating the weight for weighted sum of branches
    val = float(len(false)) / (len(false) + len(true))
    return entropy - (1 - val) * calculate_impurity(
        true) - val * calculate_impurity(false)


def split(data, decision):
    """
    Function to partition the data set, by checking the decision at every level.
    :param data: The data-set
    :param decision: Condition determining the split
    :return: Partitioned data into true_data and false_data
    """
    true_data = []
    false_data = []

    for row in data:
        # Decision is a tuple of - (feature, threshold)
        if row[decision[0]] >= decision[1]:
            true_data.append(row)
        else:
            false_data.append(row)
    return true_data, false_data


def decide_labels(data, node):
    """
    Function to classify the data into their labels
    :param data: Image
    :param node: Root node of the decision tree
    :return: Leaf node with the classified label
    """
    while not isinstance(node, int):
        if data[node.decision[0]] >= node.decision[1]:
            node = node.true_branch
        else:
            node = node.false_branch
    return node


def get_best_decision(data):
    """
    Calculates the best decision that determines the partition into true and false
    branches.
    :param data: Data-set
    :return: Maximum gain and best decision
    """

    # Calculating the current impurity
    impurity = calculate_impurity(data)

    # Initializing the gain and decision
    max_gain = 0
    best_decision = (0, 0)

    # Iterating for every feature
    for feature in range(len(data[0]) - 1):
        # Getting unique values for every feature
        values = set([row[feature] for row in data])

        for val in values:
            # Getting the partitioned data
            true_data, false_data = split(data, (feature, val))

            # Skip iteration if there is on split
            if len(true_data) == 0 or len(false_data) == 0:
                continue

            # Calculate the information gain from this split
            gain = calculate_info_gain(true_data, false_data, impurity)

            # Update the best gain
            if gain >= max_gain:
                max_gain, best_decision = gain, (feature, val)

    return max_gain, best_decision


def make_decision_tree(data):
    """
    Function to build the decision tree. It recursively builds the branches from
    a node.
    :param data: Data-set
    :return: A node of the tree
    """

    # Get the maximum gain and the best splitting criteria
    gain, decision = get_best_decision(data)

    # Base case to return the label if the tree's leaf has been reached
    if gain == 0:
        # print(data[0][len(data[0]) - 1])
        return int(data[0][len(data[0]) - 1])

    # Splitting it based on the decision
    true_rows, false_rows = split(data, decision)

    # Building the true branch
    true_branch = make_decision_tree(true_rows)

    # Building the false branch
    false_branch = make_decision_tree(false_rows)

    # Return a node of the tree at the end
    return Node(decision, true_branch, false_branch)


def generate_accuracy(cart_tree, testing_data, image_array):
    """
    Generates the accuracy of a given model.
    :param cart_tree:
    :param testing_data:
    :return:
    """

    # Array of the current labels of the test data
    target_labels = []
    for col in testing_data:
        target_labels.append(col[len(col) - 1])
    target_labels = np.array(target_labels)

    # Array of the labels after the model has been tested
    test_results = []
    for data in testing_data:
        test_results.append(decide_labels(data, cart_tree))
    test_results = np.array(test_results)

    if len(image_array) > 0:
        f = open('tree/output.txt', 'w')
        for i in range(len(image_array)):
            f.write(f'{image_array[i]} {test_results[i]}\n')
        f.close()

    return round(np.sum(np.equal(test_results, target_labels)) / len(
            target_labels) * 100, 2)


def cart(mode, mode_file, model_file):
    """
    Main function to implement the decision tree
    :param mode: Training mode or testing mode
    :param mode_file: nnet_model.txt or test-data.txt
    :param model_file: Link to the saved model file
    :return: Accuracy
    """

    if mode == 'train':
        start = time.time()
        training_data = []

        file = open(mode_file, 'r')
        for i in file:
            temp = []
            for px in i.split()[2:]:
                temp.append(int(px))
            temp.append(int(i.split()[1]))
            training_data.append(temp)

        cart_tree = make_decision_tree(training_data)

        # Save the model using Pickle
        with open(model_file, "wb") as f:
            pickle.dump(cart_tree, f, pickle.HIGHEST_PROTOCOL)
        print(f'Model Trained in {time.time() - start} seconds')

    else:
        # Retrieve the saved model
        with open(model_file, 'rb') as f:
            cart_tree = pickle.load(f)

        testing_data = []
        image_array = []

        file = open(mode_file, 'r')
        for i in file:
            temp = []
            image_array.append(i.split()[0])
            for px in i.split()[2:]:
                temp.append(int(px))
            temp.append(int(i.split()[1]))
            testing_data.append(temp)

        return generate_accuracy(cart_tree, testing_data, image_array)
