#!/usr/local/bin/python3
#
# orient.py : Main python file to call other classifiers
#
# Submitted by : Hrishikesh Paul (hrpaul), Ishita Kumar (ishkumar), Rushabh Shah (shah12)
#
# File to call different classifiers to classify images based on their
# orientation.
#

import sys

from knn import knn
from nn import nn
from tree import tree

if __name__ == '__main__':
    """
    How to run the test for a particular model, 
    1. knn: python3 orient.py test test-data.txt neighbor_model.txt nn
    2. nn: python3 orient.py test test-data.txt nnet_model.txt knn
    3. tree: python3 orient.py test test-data.txt tree_model.txt tree
    """

    if sys.argv[1] == 'test':
        if sys.argv[4] == 'tree':
            print('Testing model \n.\n.\n')
            accuracy = tree.cart(sys.argv[1], f'{sys.argv[2]}', f'{sys.argv[4]}/{sys.argv[3]}')
            print(f'Accuracy: {accuracy}%')
        if sys.argv[4] == 'nn':
            print('Testing model \n.\n.\n')
            accuracy = nn.nn(sys.argv[1], f'{sys.argv[2]}', f'{sys.argv[4]}/{sys.argv[3]}')
            print(f'Accuracy: {accuracy}%')
        if sys.argv[4] == 'knn':
            print('Testing model \n.\n.\n')
            accuracy = knn.knn(f'{sys.argv[2]}', f'{sys.argv[4]}/{sys.argv[3]}')
            print(f'Accuracy: {accuracy}%')
        if sys.argv[4] == 'best':
            print('Testing model \n.\n.\n')
            accuracy = knn.knn(f'{sys.argv[2]}', f'knn/{sys.argv[3]}')
            print(f'Accuracy: {accuracy}%')

    if sys.argv[1] == 'train':
        if sys.argv[4] == 'tree':
            print('Training model \n.\n.')
            tree.cart(sys.argv[1], f'{sys.argv[2]}', f'{sys.argv[4]}/{sys.argv[3]}')
        if sys.argv[4] == 'nn':
            print('Training model \n.\n.')
            tree.cart(sys.argv[1], f'{sys.argv[2]}', f'{sys.argv[4]}/{sys.argv[3]}')
