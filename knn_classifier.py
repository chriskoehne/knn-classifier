import math
# import time
import sys
import pandas as pd
import numpy as np
import split_data as sd

if len(sys.argv) != 3:
    print('     Incorrect number of arguments! (Expected:2, Actual:' + str(len(sys.argv) - 1) + ')')
    exit(0)

split = False
validation = pd.DataFrame
v_answers = validation
if sys.argv[2] == 'split':
    split = True
    dfs = sd.get_sets(pd.read_csv(sys.argv[1], sep=';'))
    training = dfs[0]
    validation = dfs[1]
    # validation.drop(validation.index[9:100], inplace=True)
    # validation.to_csv('my_validate.csv', sep=';', index=False)
    v_answers = validation['G']
    del validation['G']

    test = dfs[2]
    del test['G']

else:
    training = pd.read_csv(sys.argv[1], sep=';')
    test = pd.read_csv(sys.argv[2], sep=';')
    if 'G' in test.columns:
        del test['G']

sorted_data = training.sort_values(inplace=False, by='G')
sorted_data.to_csv('sorted_train.csv', sep=';', index=False)

file = open('outputKNN.txt', 'w')


def normalize(df):
    for index, row in df.iterrows():

        # First normalize numerical data to a 0-1 scale
        # print(training.at[index, 'age'])
        # print(index)
        # df.at[index, 'age'] = float(row['age'] - 15) / float(22 - 15)
        # print(float(row['age'] - 15) / float(22 - 15))
        # print(df.at[index, 'age'])
        df.loc[index, 'age'] = (df.at[index, 'age'] - 15) / (22 - 15)
        df.loc[index, 'Medu'] = (df.at[index, 'Medu'] - 0) / (4 - 0)
        df.loc[index, 'Fedu'] = (df.at[index, 'Fedu'] - 0) / (4 - 0)
        df.loc[index, 'traveltime'] = (df.at[index, 'traveltime'] - 1) / (4 - 1)
        df.loc[index, 'studytime'] = (df.at[index, 'studytime'] - 1) / (4 - 1)
        df.loc[index, 'failures'] = (df.at[index, 'failures'] - 0) / (3 - 0)
        df.loc[index, 'famrel'] = (df.at[index, 'famrel'] - 1) / (5 - 1)
        df.loc[index, 'freetime'] = (df.at[index, 'freetime'] - 1) / (5 - 1)
        df.loc[index, 'goout'] = (df.at[index, 'goout'] - 1) / (5 - 1)
        df.loc[index, 'Dalc'] = (df.at[index, 'Dalc'] - 1) / (5 - 1)
        df.loc[index, 'Walc'] = (df.at[index, 'Walc'] - 1) / (5 - 1)
        df.loc[index, 'health'] = (df.at[index, 'health'] - 1) / (5 - 1)
        df.loc[index, 'absences'] = (df.at[index, 'absences'] - 0) / (93 - 0)

        # Assign numbers to binary data
        df.loc[index, 'school'] = 0 if row['school'] == 'GP' else 1
        df.loc[index, 'sex'] = 0 if row['sex'] == 'F' else 1
        df.loc[index, 'address'] = 0 if row['address'] == 'U' else 1
        df.loc[index, 'famsize'] = 0 if row['famsize'] == 'LE3' else 1
        df.loc[index, 'Pstatus'] = 0 if row['Pstatus'] == 'T' else 1
        df.loc[index, 'schoolsup'] = 0 if row['schoolsup'] == 'no' else 1
        df.loc[index, 'famsup'] = 0 if row['famsup'] == 'no' else 1
        df.loc[index, 'paid'] = 0 if row['paid'] == 'no' else 1
        df.loc[index, 'activities'] = 0 if row['romantic'] == 'no' else 1
        df.loc[index, 'nursery'] = 0 if row['nursery'] == 'no' else 1
        df.loc[index, 'higher'] = 0 if row['higher'] == 'no' else 1
        df.loc[index, 'internet'] = 0 if row['internet'] == 'no' else 1
        df.loc[index, 'romantic'] = 0 if row['romantic'] == 'no' else 1

        if row['Mjob'] == 'teacher':
            df.at[index, 'Mjob'] = int(0b00001) / int(0b10000)
        elif row['Mjob'] == 'health':
            df.at[index, 'Mjob'] = int(0b00010) / int(0b10000)
        elif row['Mjob'] == 'services':
            df.at[index, 'Mjob'] = int(0b00100) / int(0b10000)
        elif row['Mjob'] == 'at_home':
            df.at[index, 'Mjob'] = int(0b01000) / int(0b10000)
        elif row['Mjob'] == 'other':
            df.at[index, 'Mjob'] = int(0b10000) / int(0b10000)

        if row['Fjob'] == 'teacher':
            df.at[index, 'Fjob'] = int(0b00001) / int(0b10000)
        elif row['Fjob'] == 'health':
            df.at[index, 'Fjob'] = int(0b00010) / int(0b10000)
        elif row['Fjob'] == 'services':
            df.at[index, 'Fjob'] = int(0b00100) / int(0b10000)
        elif row['Fjob'] == 'at_home':
            df.at[index, 'Fjob'] = int(0b01000) / int(0b10000)
        elif row['Fjob'] == 'other':
            df.at[index, 'Fjob'] = int(0b10000) / int(0b10000)

        if row['reason'] == 'home':
            df.at[index, 'reason'] = int(0b00001) / int(0b01000)
        elif row['reason'] == 'reputation':
            df.at[index, 'reason'] = int(0b00010) / int(0b01000)
        elif row['reason'] == 'course':
            df.at[index, 'reason'] = int(0b00100) / int(0b01000)
        elif row['reason'] == 'other':
            df.at[index, 'reason'] = int(0b01000) / int(0b01000)

        if row['guardian'] == 'mother':
            df.at[index, 'guardian'] = int(0b00001) / int(0b00100)
        elif row['guardian'] == 'father':
            df.at[index, 'guardian'] = int(0b00010) / int(0b00100)
        elif row['guardian'] == 'other':
            df.at[index, 'guardian'] = int(0b00100) / int(0b00100)

    return df


def lp_norm_distance(row1, row2, p):
    distance = 0.0
    for i in range(len(row2.columns)):
        distance += (float(row1.iloc[0][i]) - float(row2.iloc[0][i])) ** p
    distance = distance ** (1 / p)
    return distance


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2.columns)):
        if row2.columns[i] == 'failures' and row1.iloc[0][i] > 0:
            distance += 1000 * (np.square(float(row1.iloc[0][i]) - float(row2.iloc[0][i])))
        # elif row2.columns[i] == 'absences':
        #     distance += 50 * (np.square(float(row1.iloc[0][i]) - float(row2.iloc[0][i])))
        else:
            distance += np.square(float(row1.iloc[0][i]) - float(row2.iloc[0][i]))
        # print(distance)
    distance = np.sqrt(distance)
    return distance


def get_neighbors(train_df, test_row, num_neighbors):
    distances = []
    for i in range(0, len(train_df.index)):
        # for i in range(0, 1):
        train_row = train_df.iloc[[i]]
        dist = euclidean_distance(train_row, test_row)
        # dist = lp_norm_distance(train_row, test_row, len(test_row.columns))
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    # for i in range(0, len(distances)):
    #     print(distances[i])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i])
    # print(type(neighbors[0][1]))
    return neighbors


def predict(train_df, test_row, num_neighbors):
    neighbors = get_neighbors(train_df, test_row, num_neighbors)
    # print(neighbors)
    # print(neighbors[0].iloc[0][-1])
    total_distance = 0.0
    for row in neighbors:
        total_distance += row[1]
    output_values = [row[0].iloc[0][-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction, total_distance


def knn(train_df, test_df, num_neighbors, decimal_percent, testing):
    predictions = []
    plus_distances = []
    for i in range(0, len(test_df.index)):
        output = predict(train_df, test_df.iloc[[i]], num_neighbors)
        if output[0] == '+':
            plus_distances.append((i, output[1]))
        # print(output[0], output[1])
        # if i % 10 == 0:
        #     print('at index', i)
        predictions.append(output)
    if testing is True:
        return predictions
    else:
        return picky_knn(predictions, plus_distances, decimal_percent)


def picky_knn(predictions, plus_distances, decimal_percent):
    plus_distances.sort(key=lambda tup: tup[1])
    # print(plus_distances)
    # print(plus_distances)
    last = int(math.ceil(len(plus_distances) * decimal_percent))
    # print(last)
    del plus_distances[last:len(plus_distances)]
    # print(plus_distances)
    plus_distances = dict(plus_distances)
    # print(plus_distances)
    # exit(0)
    for i in range(0, len(predictions)):
        if predictions[i][0] == '+' and i in plus_distances:
            pass
        else:
            new_prediction = ('-', predictions[i][1])
            # print(predictions[i])
            predictions[i] = new_prediction
    return predictions


# print(float(training.at[0, 'age'] - 15) / float(22 - 15))
# start = time.time()
if split is True:
    # print('starting timer')
    training = normalize(training)
    validation = normalize(validation)
    test = normalize(test)
    print('starting testing\n', file=file)
    file = open('outputKNN.txt', 'a')
    for x in range(9, 10):
        if x % 2 == 0:
            continue
        print('k =', x)
        file = open('outputKNN.txt', 'a')
        print('k =', x, file=file)
        validate_predictions = knn(training, validation, x, .2, True)
        num_correct = 0
        num_incorrect = 0
        for y in range(0, len(validate_predictions)):
            if v_answers[y] == validate_predictions[y][0]:
                num_correct += 1
            else:
                num_incorrect += 1
        file = open('outputKNN.txt', 'a')
        result = 'correct ' + str(num_correct) + ', incorrect ' + str(num_incorrect)
        print(result)
        with open('outputKNN.txt', "a") as my_file:
            my_file.write(result + '\n')
        # current = time.time()
        # print('Time:', current - start)
    exit(0)

training = normalize(training)
test = normalize(test)
training.to_csv('transform_train.csv', sep=';', index=False)
test.to_csv('transform_test.csv', sep=';', index=False)
# predict(training, test, 1)

# print('starting timer')
# k = int(np.sqrt(len(training)))
# if k % 2 == 0:
#     k += 1
k = 9
picky_predictions = knn(training, test, k, .2, False)
for x in range(0, len(picky_predictions)):
    print(picky_predictions[x][0])
# end = time.time()
# print('Time:', end - start)
