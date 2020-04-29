from __future__ import print_function

import numpy as np
import pandas as pd
import bisect
import os
import math
from scipy.signal import *
from pylab import *
from random import shuffle
from collections import Counter
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import ElasticNet, LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from random import shuffle
from sklearn import svm
import mne
from mne.time_frequency import psd_array_multitaper, psd_multitaper
import matplotlib
import matplotlib.pyplot as plt
import pyriemann
from tabulate import tabulate

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

# for RL part
import os, sys, time, datetime, json, random
from keras import backend as K
from keras.layers import Layer
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU

p_err_sim = 0.9

mne.set_log_level('ERROR')
np.set_printoptions(precision=3)

# In[3]:


## Fixed Parameters of the Algorithm
data_dir = 'data'
train_dir = 'train'
test_dir = 'test'
channels = ['Pz', 'F4', 'C4', 'P4', 'O2', 'F8', 'Fp2', 'Cz', 'Fz', 'F3', 'C3', 'P3', 'O1', 'F7', 'Fp1', 'Fpz']

stim_nonErrp = 1
stim_Errp = 0
stim_dontknow = -1

th_1 = 75  # in uV : for adjacent spikes
th_2 = 150  # in uV : for min-max
freq = 125.0
time_delay = 0.0  # in seconds

# In[4]:

# Misc
flag_test = False  # True if testing o.w. Cross-validate
plt_err_dist = False  # Plots Error Distribution in time if set True
selected_channels = None  # [0,1,2,3,7,8,9,10,11,15] # If empty means all channels are selected, other option: [0,4,5]
sync_method = "event_date"  # options are "default" and "event_date"

# Classification based
low_freq = 1.0
baseline_epoc = int(0.2 * freq)
epoc_window = int(0.8 * freq)
butter_filt_order = 4
cv_folds = 10
xdawn_filters = 4
mean_baseline = False
mean_per_chan = False
car_pre = False  # True
epoc_window = int(0.8 * freq)
high_freq = 15.0
car_post = False
mean_baseline_riemann = True

# Plot-based
plot_GAERP_allchan = False
plot_GAERP_onechan = False
plot_ERP_onechan = False
chan_toPlot = 4

total_channels = 16 if selected_channels is None else len(selected_channels)
if selected_channels:
    channels = [channels[i] for i in selected_channels]

prob_th = 0.5
num_sims = 1  # only used for cross-fold validation

maze = np.array([
    [1., 0., 1., 1., 1., 1., 1., 0.],
    [1., 1., 1., 0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 1., 1., 0., 0.],
    [1., 1., 1., 1., 0., 0., 0., 1.],
    [1., 0., 0., 0., 1., 1., 1., 1.],
    [1., 0., 1., 1., 1., 0., 0., 1.],
    [1., 1., 1., 0., 1., 1., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 1.]
])
'''
LEFT = 2
UP = 0
RIGHT = 1
DOWN = 3
'''
opt_maze = np.array([
    [3.,  4.,  1.,  1.,  1.,  3.,  2., 0],
    [1.,  1.,  0.,  4.,  4.,  3.,  4., 0],
    [4.,  4.,  4.,  3.,  2.,  2.,  4., 0],
    [3.,  2.,  2.,  2.,  4.,  4.,  3., 3],
    [3.,  4.,  4.,  4.,  1.,  1.,  1., 3],
    [3.,  1.,  1.,  1.,  0.,  2.,  3., 3],
    [1.,  1.,  0.,  4.,  0.,  2.,  1., 3],
    [2.,  2.,  1.,  4.,  2.,  2.,  1., 0]
])

## Helper Functions

# function to convert missing values to 0
convert = lambda x: float(x.strip() or float('NaN'))
print("Baseline Epoch: {}".format(baseline_epoc))


def convert_codes(x):
    if ':' in x:
        return x.split(':', 1)[1]
    else:
        return (x.strip() or float('NaN'))


# function to convert stimulations
def to_byte(value, length):
    for x in range(length):
        yield value % 256
        value //= 256


# function to bandpass filter
def bandpass(sig, band, fs, butter_filt_order):
    B, A = butter(butter_filt_order, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


# In[6]:


def loadData(DataFolder, label_file):
    raw_file = label_file.split('_')[0] + '.csv'
    raw_EEG = np.loadtxt(open(os.path.join(DataFolder, raw_file), "rb"), delimiter=",", skiprows=1,
                         usecols=(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
    stim = pd.read_csv(os.path.join(DataFolder, label_file))
    return raw_EEG, stim


def pre_process(raw_EEG):
    # Step 1: Subtract Mean per electrode
    sig = raw_EEG[:, 1:]
    sig = sig[:, selected_channels] if selected_channels is not None else sig
    if mean_per_chan:
        sig_mean = np.mean(sig, axis=0)
        sig = sig - sig_mean

    # Step 2: Bandpass in the given frequency range
    sigF = bandpass(sig, [low_freq, high_freq], freq, butter_filt_order)

    # Step 3: Removing Average of all channels
    if car_pre:
        sig_mean = np.mean(sigF, axis=1)
        sigF = sigF - np.reshape(sig_mean, [len(sig_mean), 1]) * np.ones([1, total_channels])

    return sigF


def post_process(X):
    if car_post:
        temp = np.repeat(np.mean(X, axis=2), X.shape[2], axis=1)
        temp = np.reshape(temp, X.shape)
        X = X - temp
    X = X.transpose((0, 2, 1))
    return X


# Arrange EEG data for training/testing
def arrange_data(EEG, sigF, stim):
    X = []
    Y = []
    info = []
    for stim_id, stim_code in enumerate(stim['label']):
        if (not math.isnan(stim_code)) and stim_code != 33552 and stim_code != 33553 and stim_code != 33554:
            if stim_code == stim_nonErrp or stim_code == stim_Errp:
                if sync_method == "default":
                    time_instant = stim['time1'][stim_id] + time_delay
                else:
                    time_instant = stim['time2'][stim_id] + time_delay
                time_instant = time_instant
                idx = bisect.bisect(EEG[:, 0], time_instant)
                X_temp = sigF[idx - 1 - baseline_epoc:idx + epoc_window, :]
                if mean_baseline:
                    X_mean = np.mean(sigF[idx - baseline_epoc:idx, :], 0)
                    X_temp = X_temp - X_mean
                check = False
                # Removing nan values and corrupted data
                if np.isnan(X_temp).any():
                    check = True
                for i in range(total_channels):
                    X_diff = [abs(t - s) for s, t in zip(X_temp[:, i], X_temp[1:, i])]
                    if max(X_diff) > th_1 or max(X_temp[:, i]) - min(X_temp[:, i]) > th_2:
                        check = True
                if not check:
                    X.append(X_temp)
                    Y.append(stim_code)
                    a0, x0, y0 = stim['action'][stim_id], stim['agent_prev_x'][stim_id] - 1, stim['agent_prev_y'][stim_id] - 1
                    if (x0 == 0) and (y0 == 6) and (a0 == 1):
                        print("Catched {}, {}, {}, y: {}".format(x0, y0, a0, stim_code))
                    info.append([a0, x0, y0])
    return np.array(X), np.array(Y), np.array(info)


# In[7]:


# Pre-processing Data from Train/Test Directory

def dir_list(folder_path):
    dir_list = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    dir_list.append('')
    return dir_list


data_dict = {}
train_folder = os.path.join(data_dir, train_dir)
test_folder = os.path.join(data_dir, test_dir)
list_of_dir = dir_list(train_folder)
if flag_test:
    assert list_of_dir == dir_list(test_folder), "Error: Test directory does not have same structure as train"
total_dirs = len(list_of_dir)

# define buckets mapping from state (action, x, y) to EEG segments
buckets = [[] for _ in range(8*8*4)]

for dir_idx in range(total_dirs):
    paths = [["train", os.path.join(train_folder, list_of_dir[dir_idx])],
             ["test", os.path.join(test_folder, list_of_dir[dir_idx])]]
    if not flag_test:
        paths.pop(1)
    for items in paths:
        train_test = items[0]
        folder = items[1]
        list_of_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and '_labels' in f]
        # print list_of_files
        for curr_file in list_of_files:

            # Step 1: Load Raw EEG data (with filtered stimulations)
            EEG, stim = loadData(folder, curr_file)

            # Step 2: Pre-process the raw EEG data before cutting in time-windows
            sig = pre_process(EEG)

            # Step 3: Arrange Data
            X_curr, Y_curr, info = arrange_data(EEG, sig, stim)

            # Making sure either the stimulations are present or raw data during stimulation is not discarded
            if Y_curr.shape[0] > 0:
                # Step 4: Post-process the Signals
                X_curr = post_process(X_curr)

                # Step 5: Arrange data in dictionary
                key = 'data_' + list_of_dir[dir_idx] + '$' + train_test
                st_idx = 0
                if key in data_dict:
                    [X, Y] = data_dict[key]
                    st_idx = X.shape[0]
                    X = np.concatenate((X, X_curr), axis=0)
                    Y = np.concatenate((Y, Y_curr), axis=0)
                    data_dict[key] = [X, Y]
                else:
                    data_dict[key] = [X_curr, Y_curr]

                for k in range(X_curr.shape[0]):
                    b_idx = info[k][0] * 64 + info[k][1] * 8 + info[k][2]
                    # print(b_idx, k+st_idx)
                    buckets[int(b_idx)].append(k + st_idx)

exps = set()
for key in data_dict.keys():
    exps.add(key.split("$")[0])
print(exps)

# elements in the 'exps' set holds the folder names
# [X_train, Y_train] = data_dict[exp+"$train"] loads the X_train and Y_train for training samples
# All samples in all files inside one folder are concatenated when storing in data_dict

# Size of X = [#of stim x #chan x # of timestamps]


# Different Classification Pipelines after pre-processing

def pipeline_riemann(X_tr, X_te, Y_tr):
    X_train_baseline_mean = np.mean(X_tr[:, :, :baseline_epoc], 2)
    X_test_baseline_mean = np.mean(X_te[:, :, :baseline_epoc], 2)

    B, A = butter(butter_filt_order, np.array([1.0, 10.0]) / (freq / 2), btype='bandpass')

    X_tr = lfilter(B, A, X_tr, axis=2)
    X_te = lfilter(B, A, X_te, axis=2)

    X_tr = X_tr[:, :, baseline_epoc:]
    X_te = X_te[:, :, baseline_epoc:]
    if mean_baseline_riemann:
        # X_train_baseline_mean = np.reshape(X_train_baseline_mean, [,1]
        X_tr = X_tr - np.repeat(X_train_baseline_mean[:, :, np.newaxis], X_tr.shape[2], axis=2)
        X_te = X_te - np.repeat(X_test_baseline_mean[:, :, np.newaxis], X_tr.shape[2], axis=2)

    filt_1 = pyriemann.estimation.XdawnCovariances(nfilter=xdawn_filters, estimator='lwf', xdawn_estimator='lwf')
    filt_2 = pyriemann.channelselection.ElectrodeSelection(nelec=8, metric='riemann')
    filt_3 = pyriemann.tangentspace.TangentSpace(metric='riemann', tsupdate=False)  # 36-sized vector for nelec=8
    filt_4 = Normalizer(norm='l1')
    filt_5 = SGDClassifier(loss='squared_hinge', penalty='elasticnet', l1_ratio=0.5, alpha=0.002, max_iter=1000,
                           tol=0.001)

    clf = make_pipeline(filt_1, filt_2, filt_3, filt_4)
    clf.fit(X_tr, Y_tr)
    train_features = clf.transform(X_tr)
    test_features = clf.transform(X_te)

    clf_isotonic = CalibratedClassifierCV(filt_5, cv=2, method='isotonic')
    clf_isotonic.fit(train_features, Y_tr)
    train_scores = clf_isotonic.predict_proba(train_features)
    test_scores = clf_isotonic.predict_proba(test_features)

    return train_scores, test_scores, clf, clf_isotonic


# In[10]:


def balance_dataset(X_tr, Y_tr):
    rusU = RandomUnderSampler(return_indices=True)
    X_new = np.reshape(X_tr, [X_tr.shape[0], X_tr.shape[1] * X_tr.shape[2]])
    X_resU, Y_resU, id_rusU = rusU.fit_resample(X_new, Y_tr)
    X_new2 = np.reshape(X_resU, [X_resU.shape[0], X_tr.shape[1], X_tr.shape[2]])
    X_sim = X_new2
    Y_sim = Y_resU
    return X_sim, Y_sim

# In[11]:
# Acutal Prediction


def predict(X_tr, X_te, Y_tr, Y_te):
    train_preds = []
    test_preds = []

    train_riemann_prob, test_riemann_prob, clf, clf_isotonic = pipeline_riemann(X_tr, X_te, Y_tr)

    # here index 1 and value 1 - corresponds to the Non-ErrP
    train_prob = train_riemann_prob[:, 1]
    test_prob = test_riemann_prob[:, 1]

    train_preds, test_preds = np.zeros([train_prob.shape[0]]), np.zeros([test_prob.shape[0]])
    for idx in range(train_preds.shape[0]):
        if train_prob[idx] >= prob_th:
            train_preds[idx] = 1
        elif train_prob[idx] < 1 - prob_th:
            train_preds[idx] = 0
        else:
            train_preds[idx] = -1
    for idx in range(test_preds.shape[0]):
        if test_prob[idx] >= prob_th:
            test_preds[idx] = 1
        elif test_prob[idx] < 1 - prob_th:
            test_preds[idx] = 0
        else:
            test_preds[idx] = -1

    return confusion_matrix(Y_tr, train_preds, labels=[-1, 0, 1]), \
           confusion_matrix(Y_te, test_preds,labels=[-1, 0, 1]), test_preds, clf, clf_isotonic


# In[12]:
# Cross Validation: Returns train and test confusion matrix for X and Y

def predict_CV(X, Y):
    kfold = KFold(n_splits=cv_folds, shuffle=True)
    cmat_train = np.zeros([3, 3])
    cmat_test = np.zeros([3, 3])
    clf_set, clf_isotonic_set = [], []
    print(X.shape, Y.shape)
    for train_index, test_index in kfold.split(X):
        X_tr, X_te = X[train_index], X[test_index]
        Y_tr, Y_te = Y[train_index], Y[test_index]

        X_tr, Y_tr = balance_dataset(X_tr, Y_tr)

        # print "Training ErrPs: ", np.argwhere(Y_train==0).shape[0]*100.0/(X_train.shape[0]), "%"
        # print "Validation ErrPs: ", np.argwhere(Y_test==0).shape[0]*100.0/(X_test.shape[0]), "%"
        cm_train, cm_test, _, clf, clf_isotonic = predict(X_tr, X_te, Y_tr, Y_te)
        cmat_train = cmat_train + cm_train
        cmat_test = cmat_test + cm_test
        clf_set.append(clf)
        clf_isotonic_set.append(clf_isotonic)

    return cmat_train, cmat_test, clf_set, clf_isotonic_set


# In[13]:


# Computing Accuracies

result_dict = {}
cmat_train = np.zeros([3, 3])
cmat_test = np.zeros([3, 3])

errp_clf_set, errp_clf_isotonic_set = None, None
for exp in exps:
    Y_pred = None
    Y_test = None
    if flag_test:
        [X_train, Y_train] = data_dict[exp + "$train"]
        index_shuffle = range(X_train.shape[0])
        shuffle(index_shuffle)
        X_train = X_train[index_shuffle, :, :]
        Y_train = Y_train[index_shuffle]
        [X_test, Y_test] = data_dict[exp + "$test"]
        X_train, Y_train = balance_dataset(X_train, Y_train)
        cmat_train, cmat_test, Y_pred = predict(X_train, X_test, Y_train, Y_test)
    else:
        [X_train, Y_train] = data_dict[exp + "$train"]
        for ind in range(num_sims):
            a, b, errp_clf_set, errp_clf_isotonic_set = predict_CV(X_train, Y_train)
            cmat_train = cmat_train + a
            cmat_test = cmat_test + b
        cmat_train = cmat_train / num_sims
        cmat_test = cmat_test / num_sims

    assert not exp in result_dict.keys()
    train_errp_split = (sum(cmat_train, 1) * 1.0 / sum(cmat_train))[1]
    test_errp_split = (sum(cmat_test, 1) * 1.0 / sum(cmat_test))[1]
    result_dict[exp] = [cmat_train, cmat_test, X_train.shape[0], train_errp_split, test_errp_split, Y_test, Y_pred]


print_results_1 = []
print_results_2 = []
for key in result_dict.keys():
    [cmat_train, cmat_test, num_samples, tr_sp, te_sp, Y_test, Y_pred] = result_dict[key]
    train_acc = cmat_train[1:, 1:].diagonal() * 1.0 / cmat_train[1:, 1:].sum(axis=1)
    test_acc = cmat_test[1:, 1:].diagonal() * 1.0 / cmat_test[1:, 1:].sum(axis=1)
    test_perDN = cmat_train[1:, 0] * 1.0 / sum(cmat_train[1:, :], axis=1)  # percentage of Don't KNOW
    print_results_1.append(
        [key, mean(test_acc), test_acc[0], test_acc[1], mean(test_perDN), test_perDN[0], test_perDN[1]])
    print_results_2.append([key, num_samples, tr_sp, te_sp, mean(train_acc), train_acc[0], train_acc[1]])

    # Plot error distribution in time
    if plt_err_dist and flag_test:
        decision = (Y_pred == Y_test)
        x = [ind_x for ind_x in range(len(Y_test))]
        x = np.array(x)
        correct_idx = np.argwhere(decision == True)
        incorrect_idx = np.argwhere(decision == False)
        plt.plot(x[correct_idx], Y_test[correct_idx], 'b.', label='Correct')
        plt.plot(x[incorrect_idx], Y_test[incorrect_idx], 'r.', label='Incorrect')
        plt.xlabel('timesteps')
        yticks(np.array([-5, 0, 1, 5]), ('', 'ErrP', 'Non-Errp', ''))
        plt.legend()
        vert_line = 100
        while vert_line < len(Y_test):
            plt.axvline(x=vert_line)
            vert_line = vert_line + 100
        plt.title(key)
        plt.show()

print(tabulate(print_results_1,
               headers=['Name', 'Avg. Acc', 'ErrP Acc', 'Non_ErrP Acc', 'Avg. % DN', '% DN (from ErrP)',
                        '% DN (from Non-ErrP)']))
print("")
print(tabulate(print_results_2, headers=['Name', '#Samples', 'Tr_split', 'Te_split', 'Train Avg. Acc', 'Train ErrP Acc',
                                         'Train Non_ErrP Acc']))

'''
===================== RL part =====================
variables used in this part:
    X_train, Y_train, errp_clf_set, d_th_set
'''

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5  # The current rat cell will be painteg by gray 0.5
LEFT = 2
UP = 0
RIGHT = 1
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1


# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))

class Qmaze(object):
    def __init__(self, maze, rat=(0, 0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)  # target cell where the "cheese" is
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:  # invalid action, no change in rat position
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [LEFT, UP, RIGHT, DOWN]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(UP)
        elif row == nrows - 1:
            actions.remove(DOWN)

        if col == 0:
            actions.remove(LEFT)
        elif col == ncols - 1:
            actions.remove(RIGHT)

        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(UP)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(DOWN)

        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(LEFT)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(RIGHT)

        return actions


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]  # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        actions = np.zeros((data_size, 1))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            actions[i, 0] = float(action)
            # There should be no target values for actions not taken.
            targets[i] = target_model.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(target_model.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, actions, targets


num_lastlayer = num_actions * 2
E_W_ = np.random.normal(0, 1, (num_lastlayer, num_actions))
E_W = np.zeros((num_lastlayer, num_actions))
Cov_W = np.stack([np.eye(num_lastlayer) for _ in range(num_actions)])


def BayesReg(phiphiT, phiY, alpha, batch_size, experience):
    # global policy_net, target_net, device
    phiphiT *= (1 - alpha)
    # Forgetting parameter alpha suggest how much of the moment from the past can be used, we set alpha to 1
    # which means do not use the past moment
    phiY *= (1 - alpha)
    envstate, action, target = experience.get_data(
        data_size=batch_size)  # sample a minibatch of size one from replay buffer
    action = action.reshape(-1)
    for i in range(num_actions):
        state_a = envstate[action == i]
        y_a = target[:, i][action == i]
        phi_s = phi_func([state_a])[0]
        phiphiT[i] += np.dot(phi_s.T, phi_s)
        phiY[i, :] += np.dot(phi_s.T, y_a)
        inv = np.linalg.inv(phiphiT[i] / sigma_ep + 1 / sigma * np.eye(num_lastlayer))
        E_W[:, i] = np.dot(inv, phiY[i]) / sigma_ep
        Cov_W[i, :, :] = sigma * inv
    return phiphiT, phiY, E_W, Cov_W


# Thompson sampling, sample model W form the posterior.
def sample_W(E_W, U):
    C = 5
    for i in range(num_actions):
        sam = np.random.normal(0, 1, (num_lastlayer))
        E_W_[:, i] = E_W[:, i] + np.dot(U[i], sam)  # * C * np.sqrt(np.log(e + 1) / (e + 1))
    return E_W_


# ErrP detection function
def errp_detection(action, x, y):
    global errp_clf_set, errp_clf_isotonic_set, buckets, X_train, Y_train
    b_idx = action * 64 + x * 8 + y
    if buckets[b_idx] == []:
        # No ErrP
        return True, True
    errp_res = 0.
    for x_sel in buckets[b_idx]:
        X_t, Y_t = X_train[x_sel], Y_train[x_sel]
        X_t = X_t[np.newaxis, :, baseline_epoc:]
        y_hat_sum = 0.
        for clf, clf_isotonic in zip(errp_clf_set, errp_clf_isotonic_set):
            X_feature = clf.transform(X_t)
            y_hat = clf_isotonic.predict_proba(X_feature)
            y_hat_sum += y_hat[0, 1]
        # y_hat[0, 1] == 1 denotes Non-ErrP
        errp_res += float(y_hat_sum) / cv_folds

    # True denotes Non-ErrP
    return (errp_res/cv_folds) >= 0.35, bool(Y_t)


def action_selection(envstate, x, y, E_W, Cov_W, win_rate):
    global model, n_s, phi_func
    C = .5
    u = np.zeros(num_actions)
    phi_s = phi_func([envstate])[0].reshape(1, -1)
    n_s_all = np.sum(n_s[x, y, :])
    for a in range(num_actions):
        w_vec = E_W[:, a].reshape(-1, 1)
        if win_rate > 0.5:
            eps = 1.
        else:
            eps = np.random.normal(0, 1)
        u[a] = phi_s.dot(w_vec)[0, 0] + phi_s.dot(Cov_W).dot(phi_s.T)[0, 0] * eps
    return np.argmax(u)


def qtrain(model, maze, **opt):
    global epsilon, E_W, E_W_, Cov_W, phiphiT, phiY
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 256)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.maze.size // 2  # history window size
    win_rate = 0.0
    n_step = 0
    T_sampling = 1
    T_W_updating = 1
    T_Q_updating = 100
    T_target_update = 1
    epsilon = 0.1
    print(experience.max_memory)
    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()

        n_episodes = 0
        errp_set, errp_corr = [], []
        a, b = 0, 0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            x, y, _ = qmaze.state
            if np.random.rand() < 0.:
                action = random.choice(valid_actions)
            else:
                action = action_selection(envstate, x, y, E_W, Cov_W, win_rate)
                # action = np.argmax(model.predict(prev_envstate)[0])

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            n_s[x, y, action] += 1
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False
                # res = errp_detection(action, x, y)
                # ErrP_res, real_res = res[0], res[1]
                # errp_set.append(ErrP_res)
                # errp_corr.append(ErrP_res == real_res)
                # if not ErrP_res:  # and (action != opt_maze[x, y]):
                #     reward = -0.75  # ErrP detected
                # if (not ErrP_res) and real_res:
                #    b += 1
                # if (not ErrP_res) and (action == opt_maze[x, y]):
                #     print("x {}, y {}, a {}, label {}".format(x, y, action, real_res))
                # if real_res and (buckets[int(action*64+x*8+y)] != []) and (action != opt_maze[x, y]):
                #     print("Wrong x {}, y {}, a {}".format(x, y, action))
                # if action == opt_maze[x, y]:
                #     a += 1

            # Code for simulated ErrP
            # rand_num = np.random.uniform()
            # # correct action then turn as ErrP with probability 1-p
            # if (opt_maze[x,y] == action):
            #     if (rand_num < 1 - p_err_sim):
            #     reward = -0.75
            # # wrong action then turn as ErrP with probability p
            # else:
            #     if (rand_num < 1 p_err_sim):
            #     reward = -0.75


            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1
            n_step += 1

            # Updating the posterior of weights
            if (n_step % T_W_updating == 0) and (len(experience.memory) >= update_batch_size):
                phiphiT, phiY, E_W, Cov_W = BayesReg(phiphiT, phiY, update_alpha, update_batch_size, experience)

            # Sampling weights
            # if n_step % T_sampling == 0:
            #     E_W_ = sample_W(E_W, Cov_W)
            #     model.layers[-1].set_weights([E_W_])

        # probability of ErrP labels leading to wrong decisions
        if a > 0: print("{:.4f}".format(float(b)/a))
        # Train neural network model
        if (epoch % 1 == 0) and (len(experience.memory) >= data_size):
            inputs, _, targets = experience.get_data(data_size=1000)
            h = model.fit(
                inputs,
                targets,
                epochs=30,
                batch_size=32,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)
            # target_model.set_weights(model.get_weights())

        if epoch % T_target_update == 0:
            target_model.set_weights(model.get_weights())

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / float(hsize)
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        win_rate_set.append(win_rate)
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # print(errp_set)
        # print(errp_corr)
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if (sum(win_history[-hsize:]) == hsize) and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


class LastLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(LastLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=False)
        super(LastLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_lastlayer))
    model.add(PReLU())
    model.add(LastLayer(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


update_alpha = 0.95
update_batch_size = 64
sigma_ep = .1
sigma = 1.

for exp_idx in range(5):
    # init array for number of visits
    n_s = np.ones((maze.shape[0], maze.shape[1], num_actions))

    phiphiT = np.zeros((num_actions, num_lastlayer, num_lastlayer))
    phiY = np.zeros((num_actions, num_lastlayer))

    qmaze = Qmaze(maze)
    show(qmaze)

    model = build_model(maze)
    target_model = build_model(maze)
    model.layers[-1].set_weights([E_W_])
    target_model.set_weights(model.get_weights())
    phi_func = K.function([model.input], [model.layers[-2].output])
    win_rate_set = []

    qtrain(model, maze, epochs=1000, max_memory=1000, data_size=64)

    np.savez('noErrP_'+str(exp_idx), res=np.array(win_rate_set))
    print("Experiment {} finished.".format(exp_idx))
