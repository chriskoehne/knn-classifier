import math
import sys

import pandas as pd
import numpy as np
import csv

# My version of an ID3 decision tree


if len(sys.argv) != 3:
    print("     Incorrect number of arguments! (Expected: 2, Actual: " + str(len(sys.argv) - 1) + ")")
    exit(0)

training = pd.read_csv(sys.argv[1])
# training = pd.read_csv("data.csv")
test = pd.read_csv(sys.argv[2])


def safe_fraction(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator


def safe_log(num):
    if num == 0:
        return 0
    return math.log(num, 2)


def increment(num):
    num += 1
    return num


def key_entropy(df):
    entropy = 0
    last = df.keys()[-1]
    values = df[last].unique()
    for value in values:
        # print(df[last].value_counts()[value])
        fraction = safe_fraction(df[last].value_counts()[value], len(training[last]))
        entropy -= fraction * safe_log(fraction)
    return entropy


def attr_entropy(df, attribute):
    last = df.keys()[-1]
    target_vars = df[last].unique()
    attr_vars = df[attribute].unique()
    entropy = 0
    for attr_var in attr_vars:
        sub_entropy = 0
        denominator = 0
        for target_var in target_vars:
            numerator = len(df[attribute][df[attribute] == attr_var][df[last] == target_var])
            denominator = len(df[attribute][df[attribute] == attr_var])
            fraction = safe_fraction(numerator, denominator)
            sub_entropy -= fraction * safe_log(fraction)
        fraction = safe_fraction(denominator, len(df))
        entropy -= fraction * sub_entropy
    return entropy


def information_gain(df, attribute):
    return key_entropy(df) + attr_entropy(df, attribute)


def information_gains(df):
    ig_arr = []
    for column in df:
        if column == df.keys()[-1]:
            continue
        ig_arr.append(information_gain(df, column))
    return ig_arr


def pick_best(df, ig_arr):
    return str(df.columns[ig_arr.index(max(ig_arr))])


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def predict(tree, data):
    if tree.is_last is True:
        return tree.attr
    through = False
    for child in tree.children:
        if str(data[tree.attr]) == child.path:
            # print("yes ", child.path)
            through = True
            return predict(child, data)
    if through is False and tree.children[0].path.isdigit():
        closest = float(math.inf)
        send = tree.children[0]
        for child in tree.children:
            temp = abs(int(child.path) - int(data[tree.attr]))
            if temp < closest:
                closest = temp
                send = child
        return predict(send, data)

    else:
        return '-'


class Node:

    def __init__(self, attr=""):
        self.children = []
        self.attr = attr
        self.path: str = ""
        self.df = None
        self.is_last = False

    def __repr__(self, level=0):
        if self.path == '':
            ret = "\t" * level + repr(self.attr) + "\n"
        else:
            ret = "\t" * level + repr(self.path) + " " + repr(self.attr) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def build_tree(self, df, depth, limit, path=""):
        self.df = df
        self.path = str(path)
        if depth == limit:
            self.is_last = True
            last = df.keys()[-1]
            plus = df[last].value_counts()['+']
            minus = df[last].value_counts()['-']
            if plus > minus:
                self.attr = '+'
            else:
                self.attr = '-'
            return

        if len(df.columns) == 1:
            return
        self.attr = pick_best(df, information_gains(df))
        attr_vars = np.unique(df[self.attr])
        # print(self.data, attr_vars)

        df_arr = []
        for attr_var in attr_vars:
            df_arr.append(get_subtable(df, self.attr, attr_var))

        if not df_arr:
            return

        i = 0
        for sub_df in df_arr:
            last = sub_df.keys()[-1]
            if len(np.unique(sub_df[last])) != 1:
                self.children.append(Node())
                self.children[-1].build_tree(sub_df, increment(depth), limit, attr_vars[i])
                i += 1
            else:
                end = sub_df[last][0]
                self.children.append(Node(end))
                self.children[-1].path = str(attr_vars[i])
                self.children[-1].is_last = True
                i += 1


root = Node()
max_depth = 3
root.build_tree(training, 0, max_depth)
print(root)

print("Testing with max depth of", max_depth)
file = open('output.txt', 'w')
for index, row in test.iterrows():
    # print(index, predict(root, row))
    print(predict(root, row), file=file)
