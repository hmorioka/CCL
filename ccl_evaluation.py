""" Evaluation
    Main script for evaluating the model trained by ccl_training.py
"""


import os
import numpy as np
import pickle
import torch
from matplotlib.colors import ListedColormap
import colorcet as cc
from sklearn.metrics import accuracy_score

from subfunc.generate_dataset import generate_dataset
from subfunc.preprocessing import pca
from ccl import ccl, utils
from cclalt import ccl as cclalt
from subfunc.showdata import *


# parameters ==================================================
# =============================================================

eval_dir_base = './storage'

eval_dir = os.path.join(eval_dir_base, 'model')

parm_path = os.path.join(eval_dir, 'parm.pkl')
save_path = eval_dir.replace('.tar.gz', '') + '.pkl'

thresh_ratio = np.arange(0, 1.05, 0.05)
load_ema = True  # recommended unless the number of iterations was not enough
num_data_pred = 1024  # number of data points for evaluating predictions
device = 'cpu'


# =============================================================
# =============================================================
if eval_dir.find('.tar.gz') >= 0:
    unzipfolder = './storage/temp_unzip'
    utils.unzip(eval_dir, unzipfolder)
    eval_dir = unzipfolder
    parm_path = os.path.join(unzipfolder, 'parm.pkl')

model_path = os.path.join(eval_dir, 'model.pt')

# load parameter file
with open(parm_path, 'rb') as f:
    model_parm = pickle.load(f)

num_node = model_parm['num_node']
num_dim = model_parm['num_dim']
num_data = model_parm['num_data']
q_lambda1 = model_parm['q_lambda1']
q_lambda2 = model_parm['q_lambda2']
qbar_lambda1 = model_parm['qbar_lambda1']
qbar_lambda2 = model_parm['qbar_lambda2']
num_neighbor = model_parm['num_neighbor']
num_layer = model_parm['num_layer']
list_hidden_nodes = model_parm['list_hidden_nodes']
method = model_parm['method']
pair = model_parm['pair']
pca_parm = model_parm['pca_parm']
num_latent = model_parm['num_latent'] if 'num_latent' in model_parm else None
random_seed = model_parm['random_seed']


# generate sensor signal --------------------------------------
x, s, A1, A2, B1, B2 = generate_dataset(num_node=num_node,
                                        num_dim=num_dim,
                                        num_data=num_data,
                                        num_layer=num_layer,
                                        q_lambda1=q_lambda1,
                                        q_lambda2=q_lambda2,
                                        qbar_lambda1=qbar_lambda1,
                                        qbar_lambda2=qbar_lambda2,
                                        num_neighbor=num_neighbor,
                                        num_latent=num_latent,
                                        random_seed=random_seed)

# preprocessing
if pca_parm is not None:
    xshape = x.shape
    x = x.transpose([0, 2, 1]).reshape([-1, x.shape[1]]).T
    x, _ = pca(x, params=pca_parm)
    x = x.T.reshape([xshape[0], xshape[2], xshape[1]]).transpose([0, 2, 1])

# for memory limit
if num_data_pred is not None:
    x_pred = x[:num_data_pred, :, :].copy()
else:
    x_pred = x.copy()

# build model ------------------------------------------------
# -------------------------------------------------------------
num_node = x.shape[2]
conn_list = np.triu_indices(num_node, k=1)
conn_list_inv = conn_list[::-1]
conn_list = (np.concatenate([conn_list[0], conn_list_inv[0]]), np.concatenate([conn_list[1], conn_list_inv[1]]))
num_conn = len(conn_list[0])
num_conn_half = int(len(conn_list[0]) / 2)

# define network
if method == 'ccl':
    model = ccl.Net(num_node=num_node,
                    h_sizes=[num_dim] + list_hidden_nodes,
                    conn=conn_list)
elif method == 'cclalt':
    model = cclalt.Net(num_node=num_node,
                       h_sizes=[num_dim] + list_hidden_nodes)
else:
    raise ValueError
model = model.to(device)
model.eval()

# load parameters
print('Load trainable parameters from %s...' % model_path)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
if load_ema:
    model.load_state_dict(checkpoint['ema_state_dict'])

# feedforward
if method == 'ccl':
    # feedforward for h
    xconn = np.stack([np.transpose(x, [0, 2, 1]), np.transpose(x, [0, 2, 1])], axis=2)  # [data, node, 2, dim]
    xconn = xconn.reshape([-1, 2, num_dim])
    x_torch = torch.from_numpy(xconn.astype(np.float32)).to(device)
    _, h = model(x_torch, calc_logit=False)
    h = h[:, 0, :].reshape([num_data, num_node, num_dim]).permute(0, 2, 1)  # [data, dim, node]

    # feedforward for predictions
    xconn = np.stack([np.transpose(x_pred[:, :, conn_list[0]], [0, 2, 1]), np.transpose(x_pred[:, :, conn_list[1]], [0, 2, 1])], axis=2)  # [data, conn, 2, dim]
    xconn = xconn.reshape([-1, 2, num_dim])
    x_torch = torch.from_numpy(xconn.astype(np.float32)).to(device)
    logits, hconn = model(x_torch)
    predicted = torch.argmax(logits, dim=1).reshape([num_data_pred, num_conn])
    y = torch.arange(num_conn)[None, :].repeat(num_data_pred, 1)

elif method == 'cclalt':
    # feedforward for h
    xconn = np.repeat(np.transpose(x, [0, 2, 1]).reshape([-1, x.shape[1]])[:, None, :], 2, axis=1)
    x_torch = torch.from_numpy(xconn.astype(np.float32)).to(device)
    _, h = model(x_torch, calc_logit=False)
    h = h[:, 0, :].reshape([num_data, num_node, num_dim]).permute(0, 2, 1)  # [data, dim, node]

    # feedforward for predictions
    x0 = x_pred[:, :, pair]
    xast = x0.copy()
    xast[:, :, 1] = xast[np.random.permutation(x0.shape[0]), :, 1]
    x_batch = np.concatenate([x0, xast], axis=0).transpose([0, 2, 1])
    x_torch = torch.from_numpy(x_batch.astype(np.float32)).to(device)
    logits, _ = model(x_torch)
    predicted = logits > 0.0
    y = torch.cat([torch.ones([x0.shape[0]]), torch.zeros([x0.shape[0]])])

else:
    raise ValueError

# convert to numpy
pred_val = predicted.cpu().numpy()
label_val = np.squeeze(y.detach().cpu().numpy())
h_val = h.detach().cpu().numpy()

# evaluate outputs --------------------------------------------
# -------------------------------------------------------------

# classification accuracy
accu = accuracy_score(pred_val.reshape(-1), label_val.reshape(-1))

# correlation
corrmat, sort_idx, _ = utils.correlation(h_val.transpose(0, 2, 1).reshape([-1, h_val.shape[1]]),
                                         s.transpose(0, 2, 1).reshape([-1, s.shape[1]]),
                                         'Pearson')
meanabscorr = np.mean(np.abs(np.diag(corrmat)))

# causal structure
if method == 'ccl':
    wtrue = A1 * A2
    west = - model.adjacency_matrix()
    west_dir = utils.w_to_directed(west)
    #
    precision_ave = np.zeros(len(thresh_ratio))
    recall_ave = np.zeros(len(thresh_ratio))
    f1_ave = np.zeros(len(thresh_ratio))
    fpr_ave = np.zeros(len(thresh_ratio))
    for i in range(len(thresh_ratio)):
        west_thresh = utils.w_threshold(west_dir, thresh_ratio=thresh_ratio[i])
        f1, pre, rec, fpr, sort_idx_w = utils.eval_dag(wtrue, west_thresh, conn_list=conn_list)
        f1_ave[i] = np.mean(np.diag(f1[:, sort_idx_w]))
        precision_ave[i] = np.mean(np.diag(pre[:, sort_idx_w]))
        recall_ave[i] = np.mean(np.diag(rec[:, sort_idx_w]))
        fpr_ave[i] = np.mean(np.diag(fpr[:, sort_idx_w]))
else:
    precision_ave = None
    recall_ave = None
    f1_ave = None
    fpr_ave = None

# display results
print('Result...')
print('    accuracy  : %7.4f [percent]' % (accu * 100))
print(' correlation  : %7.4f' % meanabscorr)
if method == 'ccl':
    print('   precision  : (max) %7.4f (th=%g)' % (np.max(precision_ave), thresh_ratio[np.argmax(precision_ave)]))
    print('      recall  : (max) %7.4f (th=%g)' % (np.max(recall_ave), thresh_ratio[np.argmax(recall_ave)]))
    print('          F1  : (max) %7.4f (th=%g)' % (np.max(f1_ave), thresh_ratio[np.argmax(f1_ave)]))

# save results
result = {'accu': accu if 'accu' in locals() else None,
          'corrmat': corrmat if 'corrmat' in locals() else None,
          'meanabscorr': meanabscorr,
          'sort_idx': sort_idx,
          'precision': precision_ave,
          'recall': recall_ave,
          'f1': f1_ave,
          'fpr': fpr_ave,
          'num_node': num_node,
          'num_dim': num_dim,
          'thresh_ratio': thresh_ratio,
          'modelpath': model_path}

print('Save results...')
with open(save_path, 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

# visualization: correlation s vs h
showmat(corrmat,
        yticklabel=np.arange(0, corrmat.shape[0]),
        xticklabel=sort_idx.astype(np.int32),
        ylabel='True',
        xlabel='Estimated')

# visualization: causal structure
if method == 'ccl':
    for i in range(num_dim):
        plt.figure(figsize=(8 * 3, 6))
        plt.subplot(1, 3, 1)
        wdisp = wtrue[:, :, i]
        plt.imshow(wdisp, interpolation='none', aspect='equal', cmap=ListedColormap(cc.coolwarm))
        plt.colorbar()
        plt.clim([-np.max(np.abs(wdisp)), np.max(np.abs(wdisp))])
        plt.title('True')

        plt.subplot(1, 3, 2)
        wdisp = west[:, :, sort_idx[i]]
        plt.imshow(wdisp, interpolation='none', aspect='equal', cmap=ListedColormap(cc.coolwarm))
        plt.colorbar()
        plt.clim([-np.max(np.abs(wdisp)), np.max(np.abs(wdisp))])
        plt.title('Estimated (unthresholded)')

        plt.subplot(1, 3, 3)
        wdisp = utils.w_threshold(west_dir, thresh_ratio=thresh_ratio[np.argmax(f1_ave)])[:, :, sort_idx[i]]
        plt.imshow(wdisp, interpolation='none', aspect='equal', cmap=ListedColormap(cc.coolwarm))
        plt.colorbar()
        plt.clim([-np.max(np.abs(wdisp)), np.max(np.abs(wdisp))])
        plt.title('Estimated (thresholded)')

print('done.')

