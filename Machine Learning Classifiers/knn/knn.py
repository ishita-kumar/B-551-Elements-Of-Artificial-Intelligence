from collections import Counter

import numpy as np


def knn(test_data, train_data):
    """
    This function classifies the images according to their orientation
    :param test_data: Input of file test_data.txt
    :param train_data: Input of file train_data.txt
    :return: Accuracy of model
    """
    data_train = np.genfromtxt(train_data)
    label_train = data_train[..., 1]

    # Generating the image name
    image_name = np.loadtxt(test_data, dtype='str')[..., 0]
    data_test = np.genfromtxt(test_data)
    data_train = np.delete(data_train, 1, 1)
    data_train = np.delete(data_train, 0, 1)
    data_test = np.delete(data_test, 0, 1)
    label_test = data_test[..., 0]
    data_test = np.delete(data_test, 0, 1)

    prediction_label = []

    output = open('knn/output.txt', 'w')

    # k_val set to 39 for the best trade-off between time and accuracy
    k_val = 39

    for j in range(len(data_test)):
        distance = []
        for i in range(len(data_train)):
            # Getting matrix values each row in test subject
            x1 = data_train[i].T
            z1 = data_test[j].T
            x2 = data_train[i]

            z2 = data_test[j]
            dist = np.dot(x1, x2) + np.dot(z1, z2) - 2 * np.dot(x1, z2)
            distance.append(dist)

        # Calculating Euclidean distance
        k_min_dist = np.argpartition(distance, k_val)[:k_val]

        prediction_label.append(
            (int(Counter(label_train[k_min_dist]).most_common(1)[0][0])))
    # Writing into file
    for i in range(len(image_name)):
        output.write(f'{image_name[i]} {prediction_label[i]}\n')

    return round(np.mean(np.equal(prediction_label, label_test)) * 100, 2)


