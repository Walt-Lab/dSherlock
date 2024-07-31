"""
auxiliary functions for the analysis of dSHERLOCK data

author: Anton Thieme <anton@thiemenet.de>
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import fmin_slsqp
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema, find_peaks
import scienceplots


def batch_load_and_combine(experiment_path, filename):
    uid_offset = 0
    df_list = []

    for file_or_folder in os.listdir(experiment_path):

        path = os.path.join(experiment_path, file_or_folder)

        if os.path.isdir(path):

            for file in os.listdir(path):

                if file == filename:
                    df = pd.read_csv(os.path.join(path, file))
                    df['set'] = path[path.rfind("\\") + 1 + 8:]
                    df['UID'] = df['UID'] + uid_offset
                    df_list.append(df)

                    uid_offset = df['UID'].max()

    df = pd.concat(df_list)
    df.reset_index(inplace=True)

    return df


def clean(df, z_roxInitial_exclude=2.5, z_fqInitial_exclude=2.5, frac_edge_exclude=0.05, dropna=True):
    df['z_rox'] = df.groupby('set', group_keys=False)['feat_rox_initial'].apply(stats.zscore)
    df['z_rox'] = df.groupby('set', group_keys=False)['z_rox'].apply(np.abs)
    df = df.drop(df[df['z_rox'] > z_roxInitial_exclude].index, axis=0)

    df['z_fq'] = df.groupby('set', group_keys=False)['feat_fq_initial'].apply(stats.zscore)
    df['z_fq'] = df.groupby('set', group_keys=False)['z_fq'].apply(np.abs)
    df = df.drop(df[df['z_fq'] > z_fqInitial_exclude].index, axis=0)

    for set in df['set'].unique():
        min_x = min(df.loc[df['set'] == set]['x'])
        max_x = max(df.loc[df['set'] == set]['x'])
        min_y = min(df.loc[df['set'] == set]['y'])
        max_y = max(df.loc[df['set'] == set]['y'])
        x_len = max_x - min_x
        y_len = max_y - min_y

        new_x_min = min_x + frac_edge_exclude * x_len
        new_x_max = max_x - frac_edge_exclude * x_len
        new_y_min = min_y + frac_edge_exclude * y_len
        new_y_max = max_y - frac_edge_exclude * y_len

        df = df.drop(df.loc[(df['set'] == set) & (
                (df['x'] < new_x_min) | (df['x'] > new_x_max) | (df['y'] < new_y_min) | (
                df['y'] > new_y_max))].index)

    if dropna:
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='any')

    return df


def testTrainSplit(df, n_class_members):
    df = pd.concat(
        [df.loc[df['cluster'] == 'MT'].sample(n_class_members),
         df.loc[df['cluster'] == 'WT'].sample(n_class_members)])

    uid_full = np.array(df['UID'])

    # uid_train_neg = np.array(df.loc[(df['cluster'] == 'NEG')]['UID'].sample(n=len_train, random_state=55))
    uid_train_mt = np.array(df.loc[(df['cluster'] == 'MT')]['UID'].sample(n=n_class_members - 1000, random_state=55))
    uid_train_wt = np.array(df.loc[(df['cluster'] == 'WT')]['UID'].sample(n=n_class_members - 1000, random_state=55))

    uid_train = np.concatenate([uid_train_mt, uid_train_wt])
    uid_test = uid_full[~np.isin(uid_full, uid_train)]

    return uid_train, uid_test


def get_concentration_estimate(frac, dilution_factor):
    molecules_per_liter = - 1e6 * dilution_factor / 0.000755 * math.log(1 - frac, math.e)
    molar = molecules_per_liter / (6.022 * 1e23)
    femtomolar = molar * 1e15
    # print("Estimated Concentration:  {:.2f}fM".format(femtomolar))

    return femtomolar


def estimateGaussian(X):
    m = X.shape[0]
    # compute mean of X
    sum_ = np.sum(X, axis=0)
    mu = (sum_ / m)
    # compute variance of X
    var = np.var(X, axis=0)
    return mu, var


def multivariateGaussian(X, mu, sigma):
    k = len(mu)
    sigma = np.diag(sigma)
    X = X - mu.T
    p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(sigma) ** 0.5)) * np.exp(
        -0.5 * np.sum(X @ np.linalg.pinv(sigma) * X, axis=1))
    return p


def labelNeg_thresh(df, feat, neg_sub, neg_add, neg_max):
    probs = pd.DataFrame({'UID': [], 'set': [], 'prob': []})
    mus = []

    for set in df['set'].unique():
        y, x, _ = plt.hist(x=df.loc[(df['set'] == set) & (df[feat] < neg_max)][feat], bins=1000)

        NEG_mean = x[np.where(y == y.max())][0]

        half_width = NEG_mean - x[np.where(y > 3)][0]

        print("Range: {} - {}".format(NEG_mean, NEG_mean + 1.4 * half_width))

        plt.clf()

        y, x, _ = plt.hist(
            x=df.loc[(df['set'] == set) & (df[feat] < NEG_mean + 2 * half_width) & (df[feat] > NEG_mean)][feat],
            bins=10)

        extrema_ind = find_peaks(-y)
        try:
            thresh = x[extrema_ind[0][0]]
        except:
            thresh = x[len(x) - 1]

        X = np.array(df.loc[(df['set'] == set) & (df[feat] < thresh)][[feat]])
        mu, var = estimateGaussian(X)
        mus.append(mu)
        print('#####')
        print(mu)
        print(var)

        print(set)

        plt.plot(np.linspace(1, 10000, 1000),
                 1000000 * stats.norm.pdf(np.linspace(1, 10000, 1000), mu[0], np.sqrt(var[0])), color='red')
        plt.hist(x=df.loc[df['set'] == set][feat], bins=100, color='blue')
        # plt.vlines(x=thresh, ymin=0, ymax=1000, color='red')
        plt.show()

        X_full = np.array(df.loc[(df['set'] == set)][[feat]])

        X_prob = multivariateGaussian(X_full, mu, np.sqrt(var))
        print(X_prob)

        probs = pd.concat([probs, pd.DataFrame({'UID': df.loc[df['set'] == set]['UID'], 'set': set, 'prob': X_prob})])

    df = df.merge(probs, on=['UID', 'set'])
    df.loc[df['prob'] > 0, 'cluster'] = 'NEG'

    for ii, set in enumerate(df['set'].unique()):
        df.loc[(df['set'] == set) & (df[feat] < mus[ii][0]), 'cluster'] = 'NEG'

    return df, mus


def normalize(df, feat):
    print(feat)

    # plt.style.use(['science', 'no-latex', 'nature'])
    # plt.rcParams['svg.fonttype'] = 'none'

    corrected = pd.DataFrame({'UID': [], 'set': [], '{}_corr'.format(feat): []})

    mus = []
    vars = []
    legend_elements = []

    for set in df['set'].unique():
        print(set)
        X = np.array(df.loc[(df['set'] == set) & (df['cluster'] == 'NEG')][[feat]])
        mu, var = estimateGaussian(X)
        mus.append(mu[0])
        vars.append(var[0])

        # fig, ax = plt.subplots()
        n, bins, patches = plt.hist(x=df.loc[df['set'] == set][feat], bins=100, color='blue')
        legend_elements.append(Patch(facecolor=patches[0].get_facecolor(), label=set))
        plt.plot(np.linspace(bins[0], bins[len(bins) - 1], 10000),
                 (1 / max(
                     stats.norm.pdf(np.linspace(bins[0], bins[len(bins) - 1], 10000), mu[0], np.sqrt(var[0])))) * max(
                     n) * stats.norm.pdf(np.linspace(bins[0], bins[len(bins) - 1], 10000), mu[0], np.sqrt(var[0])),
                 color='orange')

        # plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\Normalization_Pre_{}_{}.svg'.format(set, feat), format='svg', dpi=450, transparent=True)
        plt.clf()

        X_full = df.loc[(df['set'] == set)][[feat]]
        X_full["{}_corr".format(feat)] = (X_full[feat] - mu[0]) / math.sqrt(var[0])

        corrected = pd.concat([corrected, pd.DataFrame({'UID': df.loc[df['set'] == set]['UID'], 'set': set,
                                                        '{}_corr'.format(feat): X_full['{}_corr'.format(feat)]})])

    df = df.merge(corrected, on=['UID', 'set'])

    # plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\Normalization_Post_{}_{}.svg'.format(set, feat), format='svg', dpi=450, transparent=True)
    # plt.clf()

    return df


def processKinetics(df, forest, scaler, pca, features_norm, features_orig, neg_prob_cutoff=None):
    # remove outliers
    df = clean(df, z_roxInitial_exclude=1.5, z_fqInitial_exclude=2.5, frac_edge_exclude=0.05, dropna=True)

    # label negative cluster by gaussian fitting
    df, _ = labelNeg_thresh(df, 'feat_fq_delta_max', 2500, 2500, 10000)
    df.drop('prob', axis=1, inplace=True)

    print(df)
    print(df.isna().any())

    # decompose into PCs and predict clusters
    X_CONT = np.array(df.loc[df['cluster'] != 'NEG'][[feat for feat in features_norm]])
    X_DISC = np.array(df.loc[df['cluster'] != 'NEG'][[feat for feat in features_orig]])

    # apply scaling and pca to continuous features
    X_CONT_SCALED = scaler.transform(X_CONT)
    X_PC = pca.transform(X_CONT_SCALED)

    # make predictions
    df.loc[df['cluster'] != 'NEG', 'cluster'] = forest.predict(np.concatenate([X_CONT, X_PC, X_DISC], axis=1))

    return df, X_PC


