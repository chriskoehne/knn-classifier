import pandas as pd
import sys

data_frame = pd.read_csv(sys.argv[1], sep=';')


def get_sets(df):
    train = pd.DataFrame(
        columns=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
                 'health', 'absences', 'G'])

    validate = train
    test = validate

    num_train_plus = len(df) * .7 * .5
    num_train_minus = num_train_plus
    num_validate_plus = len(df) * .2 * .5
    num_validate_minus = len(df) * .2 * .5
    num_test_plus = len(df) * .1 * .5
    num_test_minus = len(df) * .1 * .5

    n_tr_p_count = 0
    n_tr_m_count = 0
    n_v_p_count = 0
    n_v_m_count = 0
    n_te_p_count = 0
    n_te_m_count = 0

    for i in range(0, len(df)):
        if df.iloc[i]['G'] == '+':
            if n_tr_p_count < num_train_plus:
                train = train.append(df.iloc[i], ignore_index=True)
                n_tr_p_count += 1
            elif n_v_p_count < num_validate_plus:
                validate = validate.append(df.iloc[i], ignore_index=True)
                n_v_p_count += 1
            elif n_te_p_count < num_test_plus:
                test = test.append(df.iloc[i], ignore_index=True)
                n_te_p_count += 1
        else:
            if n_tr_m_count < num_train_minus:
                train = train.append(df.iloc[i], ignore_index=True)
                n_tr_m_count += 1
            elif n_v_m_count < num_validate_minus:
                validate = validate.append(df.iloc[i], ignore_index=True)
                n_v_m_count += 1
            elif n_te_m_count < num_test_minus:
                test = test.append(df.iloc[i], ignore_index=True)
                n_te_m_count += 1

    train.to_csv('my_train.csv', sep=';', index=False)
    validate.to_csv('my_validate.csv', sep=';', index=False)
    test.to_csv('my_test.csv', sep=';', index=False)

    return train, validate, test
