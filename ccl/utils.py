""" Utilities
"""


import numpy as np
import scipy as sp
import os
import shutil
import tarfile
import scipy.stats as ss
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from subfunc.showdata import *
from subfunc.munkres import Munkres


# =============================================================
# =============================================================
def w_to_directed(w):
    """ Convert w to directed graph
    Args:
        w: [node x node x dim]
    Returns:
        wdir: directed w, NaN if not determined
    """

    num_node, _, num_dim = w.shape
    wdir = w.copy()
    for d in range(num_dim):
        for i in range(num_node):
            for j in range(i + 1, num_node):
                if np.abs(wdir[i, j, d]) > np.abs(wdir[j, i, d]):
                    wdir[j, i, d] = 0
                elif np.abs(wdir[i, j, d]) < np.abs(wdir[j, i, d]):
                    wdir[i, j, d] = 0
                elif (np.abs(wdir[i, j, d]) == np.abs(wdir[j, i, d])) and (wdir[i, j, d] != 0):
                    # cannot determine the direction
                    wdir[i, j, d] = np.nan
                    wdir[j, i, d] = np.nan

    return wdir


# =============================================================
# =============================================================
def w_threshold(w, thresh_ratio=0):
    """ Apply threshold to w
    Args:
        w: [node x node x dim]
        thresh_ratio: Threshold ratio compared to the maximum absolute value
    Returns:
        wthresh: thresholded w
    """

    num_node, _, num_dim = w.shape
    wthresh = np.zeros_like(w)
    for d in range(num_dim):
        wd = w[:, :, d].copy()
        thval = np.max(np.abs(wd)) * thresh_ratio
        wd[np.abs(wd) <= thval] = 0
        wthresh[:, :, d] = wd

    return wthresh


# =============================================================
# =============================================================
def eval_dag(wtrue, west, conn_list):
    """ Evaluate estimated causal sturcture
    Args:
        wtrue: [node x node x dim]
        west: [node x node x dim]
        conn_list: list of edges
    Returns:
        F1: [dim, dim]
        precision: [dim, dim]
        recall: [dim, dim]
        FPR: [dim, dim]
        sort_idx
    """

    num_node, _, num_dim = wtrue.shape

    # evaluate across dimensions
    pre_mat = np.zeros([num_dim, num_dim])
    rec_mat = np.zeros([num_dim, num_dim])
    f1_mat = np.zeros([num_dim, num_dim])
    fpr_mat = np.zeros([num_dim, num_dim])
    for d1 in range(num_dim):
        for d2 in range(num_dim):
            cause_true = wtrue[:, :, d1].copy()
            cause_est = west[:, :, d2].copy()
            cause_est_t = cause_est.copy().T

            # decide direction of nans favorably
            for i in range(num_node):
                for j in range(i + 1, num_node):
                    if np.isnan(cause_est[i, j]):
                        if cause_true[i, j] > 0:
                            cause_est[i, j] = 1
                            cause_est[j, i] = 0
                            cause_est_t[i, j] = 1
                            cause_est_t[j, i] = 0
                        else:
                            cause_est[i, j] = 0
                            cause_est[j, i] = 1
                            cause_est_t[i, j] = 0
                            cause_est_t[j, i] = 1

            # vectorize & binarize
            cause_true = cause_true[conn_list[0], conn_list[1]] != 0
            cause_est = cause_est[conn_list[0], conn_list[1]] != 0
            cause_est_t = cause_est_t[conn_list[0], conn_list[1]] != 0

            precision = precision_score(cause_true, cause_est, zero_division=0)
            precision_t = precision_score(cause_true, cause_est_t, zero_division=0)

            recall = recall_score(cause_true, cause_est, zero_division=0)
            recall_t = recall_score(cause_true, cause_est_t, zero_division=0)

            f1 = f1_score(cause_true, cause_est, zero_division=0)
            f1_t = f1_score(cause_true, cause_est_t, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(cause_true, cause_est).flatten()
            fpr = fp / (fp + tn)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(cause_true, cause_est_t).flatten()
            fpr_t = fp_t / (fp_t + tn_t)

            # decide trasponse or not based on f1
            if f1 >= f1_t:
                pre_mat[d1, d2] = precision
                rec_mat[d1, d2] = recall
                f1_mat[d1, d2] = f1
                fpr_mat[d1, d2] = fpr
            else:
                pre_mat[d1, d2] = precision_t
                rec_mat[d1, d2] = recall_t
                f1_mat[d1, d2] = f1_t
                fpr_mat[d1, d2] = fpr_t

    # sorting
    munk = Munkres()
    indexes = munk.compute(-f1_mat)
    sort_idx = [idx[1] for idx in indexes]

    return f1_mat, pre_mat, rec_mat, fpr_mat, sort_idx


# =============================================================
# =============================================================
def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
         method: correlation method ('Pearson' or 'Spearman')
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
     """

    print('Calculating correlation...')

    x = x.copy().T
    y = y.copy().T
    dimx = x.shape[0]
    dimy = y.shape[0]

    # calculate correlation
    if method == 'Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dimy, dimy:]
    elif method == 'Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dimy, dimy:]
    else:
        raise ValueError
    if np.max(np.isnan(corr)):
        raise ValueError

    # sort
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dimy, dtype=int)
    for i in range(dimy):
        sort_idx[i] = indexes[i][1]
    sort_idx_other = np.setdiff1d(np.arange(0, dimx), sort_idx)
    sort_idx = np.concatenate([sort_idx, sort_idx_other])

    x_sort = x[sort_idx, :]

    # re-calculate correlation
    if method == 'Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dimy, dimy:]
    elif method == 'Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dimy, dimy:]
    else:
        raise ValueError

    return corr_sort, sort_idx, x_sort


# ===============================================================
# ===============================================================
def unzip(loadfile, unzipfolder, necessary_word='/storage'):
    """unzip trained model (loadfile) to unzipfolder
    """

    print('load: %s...' % loadfile)
    if loadfile.find(".tar.gz") > -1:
        if unzipfolder.find(necessary_word) > -1:
            if os.path.exists(unzipfolder):
                print('delete savefolder: %s...' % unzipfolder)
                shutil.rmtree(unzipfolder)  # remove folder
            archive = tarfile.open(loadfile)
            archive.extractall(unzipfolder)
            archive.close()
        else:
            assert False, "unzip folder doesn't include necessary word"
    else:
        if os.path.exists(unzipfolder):
            print('delete savefolder: %s...' % unzipfolder)
            shutil.rmtree(unzipfolder)  # remove folder
        os.makedirs(unzipfolder)
        src_files = os.listdir(loadfile)
        for fn in src_files:
            full_file_name = os.path.join(loadfile, fn)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, unzipfolder + '/')

    if not os.path.exists(unzipfolder):
        raise ValueError
