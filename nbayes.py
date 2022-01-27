import argparse
import os.path
import numpy as np
from sting.data import Feature, parse_c45
from math import log
from math import exp
import util
import copy
import pandas as pd
import matplotlib.pyplot as plt


# Discrete the continuous attributes
def discretization(schema, data, k):
    x = data  # train set
    discrete_x = x
    # first, judge the type of data
    for i in range(0, len(schema)):
        if "CONTINUOUS" in str(schema[i].ftype):
            # find out the sorting index
            continuous_column = x[:, i]
            dk = (max(continuous_column) - min(continuous_column)) / k
            dcon = np.floor((continuous_column - min(continuous_column)) / dk + 1)

            for j in range(0, len(continuous_column)):
                if dcon[j] > k: dcon[j] = k

            discrete_x[:, i] = dcon

    return discrete_x


# This is the fit function that calculates those probabilities we need
def naive_bayes_learning(d_training_x: np.ndarray, train_y: np.ndarray, m):
    # get all values of attribute and count them
    y_hat, y_counts = np.unique(train_y, return_counts=True)
    pr_y0 = y_counts[0] / sum(y_counts)  # P(Y=0)
    pr_y1 = 1 - pr_y0  # P(Y=1)

    # to split the original table into two tables, one is Y=1 and the other one is Y=0
    x_0_index = np.where(train_y == 0)
    x_1_index = np.where(train_y == 1)
    x_0 = d_training_x[x_0_index]
    x_1 = d_training_x[x_1_index]
    x_xi = []  # to save the result of P(X=xi)

    learn_y0 = []  # save the prob when Y=0
    learn_y1 = []  # save the prob when Y=1
    test_name_list_0 = []  # save the feature values when Y=0
    test_name_list_1 = []  # save the feature values when Y=1
    test_name_xi = []  # save the feature values

    # calculate P(X=xi)
    for i in range(len(d_training_x[0])):
        x_xi_temp = []
        xi_lable, xi_counts = np.unique(d_training_x[:, i], return_counts=True)
        test_name_xi.append(xi_lable)

        for j in range(1, int(max(d_training_x[:, i])) + 1):
            xi_index = np.where(d_training_x[:, i] == j)
            p_xi = len(xi_index[0]) / len(d_training_x)
            x_xi_temp.append(p_xi)

        x_xi.append(x_xi_temp)

    # calculate the prob when Y=0
    for i in range(0, len(x_0[1])):
        test_name_0, test_counts_0 = np.unique(x_0[:, i], return_counts=True)
        # print(test_name_0)
        test_name_list_0.append(test_name_0)
        p = 1 / len(test_name_0)
        temp = []

        for j in range(0, len(test_counts_0)):
            if m >= 0:
                pro_0 = (test_counts_0[j] + m * p) / (y_counts[0] + m)
            else:
                pro_0 = (test_counts_0[j] + 1) / (y_counts[0] + p)

            temp.append(pro_0)

        learn_y0.append(temp)

    # calculate the prob when Y=1
    for i in range(0, len(x_1[1])):
        test_name_1, test_counts_1 = np.unique(x_1[:, i], return_counts=True)
        test_name_list_1.append(test_name_1)
        p = 1 / len(test_name_1)
        temp = []

        for j in range(0, len(test_counts_1)):
            if m >= 0:
                pro_1 = (test_counts_1[j] + m * p) / (y_counts[1] + m)
            else:
                pro_1 = (test_counts_1[j] + 1) / (y_counts[1] + p)

            temp.append(pro_1)

        learn_y1.append(temp)

    return learn_y0, learn_y1, pr_y1, pr_y0, test_name_list_0, test_name_list_1, x_xi, test_name_xi


# This function is to classify new data
def classification(d_test_x, learn_y0, learn_y1, pr_y1, pr_y0, test_name_list_0, test_name_list_1, x_xi, test_name_xi):
    classified_y = []
    confidence = []

    for i in range(0, len(d_test_x)):
        decision_value = log(pr_y1) - log(pr_y0)  # the first item of decision value
        prob_y1_xi = pr_y1  # P(X=xi, Y=1)
        sum_xi = 1  # P(X=xi)

        for j in range(0, len(d_test_x[i, :])):
            v = d_test_x[i, j]
            # p = len(d_test_x[:, j])

            # find the P(X=xi)
            if v not in test_name_xi[j]:
                prob_xi = 1
            else:
                prob_xi = x_xi[j][int(v) - 1]

            # find the prob in table Y=1
            if v not in test_name_list_1[j]:
                temp_1 = 0
                pro_b1 = 1
            else:
                index_1 = list(test_name_list_1[j]).index(v)
                temp_1 = log(learn_y1[j][index_1])
                pro_b1 = exp(temp_1)

            # find the prob in table Y=0
            if v not in test_name_list_0[j]:
                temp_0 = 0
                pro_b0 = 1
            else:
                index_0 = list(test_name_list_0[j]).index(v)
                temp_0 = log(learn_y0[j][index_0])
                pro_b0 = exp(temp_0)

            decision_value += temp_1 - temp_0  # calculate the decision value
            prob_y1_xi *= pro_b1  # get the P(X=xi, Y=1)
            sum_xi *= prob_xi  # get the P(X=xi)

        # get the classified table
        if decision_value > 0:
            classified_y.append(1)
        elif decision_value < 0:
            classified_y.append(0)

        confidence.append(prob_y1_xi / sum_xi)  # The confidence is calculated by P(X=xi|Y=1)/P(X=xi)

    return classified_y, confidence

  
# This function is to do evaluation
def bayes_evaluate(test_y, classified_y, confidence):
    test_y = np.array(test_y)
    classified_y = np.array(classified_y)
    confidence = np.array(confidence)
    # calculate accuracy, precision, recall, roc_pairs and auc
    accuracy = util.accuracy(test_y, classified_y)
    precision = util.precision(test_y, classified_y)
    recall = util.recall(test_y, classified_y)
    roc_pairs = util.roc_curve_pairs(test_y, confidence)
    auc_area = util.auc(roc_pairs)

    return accuracy, precision, recall, roc_pairs, auc_area

# This function is to do cross validation
def cross_validation(x: np.ndarray, y: np.ndarray, m):
    data_sample_list, label_sample_list = util.cv_split(x, y, 5)
    acc_list = []
    precision_list = []
    recall_list = []
    roc_list = []

    # 5 folds cross validation
    for h in range(5):
        tmp_data_sample_list = copy.deepcopy(data_sample_list)
        tmp_label_sample_list = copy.deepcopy(label_sample_list)
        print("Cross validation time:" + str(h + 1))
        validation_training = tmp_data_sample_list.pop(h)
        validation_label = tmp_label_sample_list.pop(h)
        # get training data
        training_data = np.vstack((tmp_data_sample_list[0], tmp_data_sample_list[1]))
        training_data = np.vstack((training_data, tmp_data_sample_list[2]))
        training_data = np.vstack((training_data, tmp_data_sample_list[3]))
        # get the training label
        training_label = np.hstack((tmp_label_sample_list[0], tmp_label_sample_list[1]))
        training_label = np.hstack((training_label, tmp_label_sample_list[2]))
        training_label = np.hstack((training_label, tmp_label_sample_list[3]))
        # fit and predict
        learn_y0, learn_y1, pr_y1, pr_y0, test_name_list_0, test_name_list_1, x_xi, test_name_xi = naive_bayes_learning(
            training_data,
            training_label,
            m)
        classified_y, confidence = classification(validation_training, learn_y0, learn_y1, pr_y1, pr_y0,
                                                  test_name_list_0,
                                                  test_name_list_1, x_xi, test_name_xi)
        predict_value = np.asarray(classified_y)
        accuracy, precision, recall, roc_pairs, auc_area = bayes_evaluate(validation_label, predict_value, confidence)
        # calculate the evaluations
        acc_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        roc_list.extend(roc_pairs)

    print("Accuracy:", np.mean(acc_list), "std", np.std(acc_list))
    print("Precision:", np.mean(precision_list), "std", np.std(precision_list))
    print("Recall:", np.mean(recall_list), "std", np.std(recall_list))
    cv_auc = util.auc(roc_list)
    print("Area under ROC:", cv_auc)
    # this part is to plot the ROC figure
    # roc_pairs = np.sort(roc_list, axis=0)
    # plt.plot(roc_pairs[:,0],roc_pairs[:,1],c='#0000FF') # this is to plot the ROC figure
    # xp = [0,1]
    # yp = [0,1]
    # plt.plot(xp,yp,c='#FF0000')
    # plt.ylabel('TP rate')
    # plt.xlabel('FP rate')
    # plt.title('Cross Validation ROC of Volcanoes  '+'AUC= '+str(round(cv_auc,3)))
    # plt.grid()
    # plt.show()
    
if __name__ == '__main__':

    # np.set_printoptions(threshold=np.inf)
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a Naive Bayes.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('k', metavar='k', type=int,
                        help='k equal length')
    parser.add_argument('m', metavar='m', type=float,
                        help='parameter m')
    args = parser.parse_args()

    if args.k < 2:
        raise argparse.ArgumentTypeError('k must be larger than 2.')

    # You can access args with the dot operator like so: python nbayes.py 440data\spam 10 1
    data_path = os.path.expanduser(args.path)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    k = args.k
    m = args.m
    # Read the data
    schema, x, y = parse_c45(file_base, root_dir)
    # If use cross validation
    if args.cv:
        train_x = discretization(schema, x, k)
        cross_validation(train_x, y, m)
    else:
        train_x = discretization(schema, x, k)
        shuffle_ix = np.random.permutation(np.arange(len(x)))
        train_x = x[shuffle_ix]
        train_label = y[shuffle_ix]
        train_size = int(len(train_x) * 0.8)
        train_x, test_x, train_y, test_y = train_x[:train_size], train_x[train_size:], train_label[
                                                                                       :train_size], train_label[
                                                                                                     train_size:]
        # fit and predict
        learn_y0, learn_y1, pr_y1, pr_y0, test_name_list_0, test_name_list_1, x_xi, test_name_xi = naive_bayes_learning(
            train_x, train_y, m)
        classified_y, confidence = classification(test_x, learn_y0, learn_y1, pr_y1, pr_y0, test_name_list_0,
                                                  test_name_list_1, x_xi, test_name_xi)
        accuracy, precision, recall, roc_pairs, auc_area = bayes_evaluate(test_y, classified_y, confidence)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("Area under ROC:", auc_area)
        # this part is to plot the ROC figure
        # roc_pairs = np.sort(roc_pairs, axis=0)
        # plt.plot(roc_pairs[:,0],roc_pairs[:,1],c='#0000FF') # this is to plot the ROC figure
        # xp = [0,1]
        # yp = [0,1]
        # plt.plot(xp,yp,c='#FF0000')
        # plt.ylabel('TP rate')
        # plt.xlabel('FP rate')
        # plt.title('ROC of Volcanoes    '+'AUC= '+str(round(auc_area,3)))
        # plt.grid()
        # plt.show()
