""" Training
    Main script for training the model
"""


import os
import pickle
import shutil
import tarfile

from subfunc.generate_dataset import generate_dataset
from subfunc.preprocessing import pca
from ccl.ccl_train import train
from cclalt.ccl_train import train as train_alt
from subfunc.showdata import *


# parameters ==================================================
# =============================================================

# data generation ---------------------------------------------
num_layer = 3  # number of layers of mixing-MLP (L)
num_dim = 10  # number of components or modalities (d)
num_data = 2**12  # number of data points (n)
random_seed = 0  # random seed

# simulation1
num_node = 30  # number of nodes (p)
num_latent = None  # number of latent confounders within p

# # simulation2
# num_node = 60  # number of nodes (p)
# num_latent = 30  # number of latent confounders within p

# pairwise gauss (acyclic)
q_lambda1 = [0.7, 1]  # std of q
q_lambda2 = [1, 1]
qbar_lambda1 = [1, 1]  # std of marginal distribution (root node)
qbar_lambda2 = [-0.0, 0.0]  # mean of marginal distribution (root node)
num_neighbor = 3  # number of parents

# MLP ---------------------------------------------------------
list_hidden_nodes = [2 * num_dim]*(num_layer - 1) + [num_dim]
# list of the number of nodes of each hidden layer of feature-MLP
# [layer1, layer2, ..., layer(num_layer)]

method = 'ccl'  # CCL
# method, pair = 'cclalt', [0, 1]  # CCLalt

# training ----------------------------------------------------
initial_learning_rate = 0.1  # initial learning rate (default:0.1)
momentum = 0.9  # momentum parameter of SGD
max_steps = int(8e5)  # number of iterations (mini-batches)
decay_steps = [int(5e5), int(7e5)]  # decay steps (tf.train.exponential_decay)
decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
batch_size = 512  # mini-batch size
moving_average_decay = 0.999  # moving average decay of variables to be saved
checkpoint_steps = int(1e7)  # interval to save checkpoint
summary_steps = 2000  # interval to save summary
apply_pca = True  # apply PCA for preprocessing or not
weight_decay = 1e-5  # weight decay
device = None  # gpu id (or None)


# other -------------------------------------------------------
# # Note: save folder must be under ./storage
train_dir_base = './storage'

train_dir = os.path.join(train_dir_base, 'model')  # save directory (Caution!! this folder will be removed at first)

train_parm_path = os.path.join(train_dir, 'parm.pkl')  # file name to save parameters


# =============================================================
# =============================================================

# prepare save folder -----------------------------------------
if train_dir.find('/storage/') > -1:
    if os.path.exists(train_dir):
        print('delete savefolder: %s...' % train_dir)
        shutil.rmtree(train_dir)  # remove folder
    print('make savefolder: %s...' % train_dir)
    os.makedirs(train_dir)  # make folder
else:
    assert False, 'savefolder looks wrong'

# generate sensor signal --------------------------------------
x, _, _, _, _, _ = generate_dataset(num_node=num_node,
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
if apply_pca:
    xshape = x.shape
    x = x.transpose([0, 2, 1]).reshape([-1, x.shape[1]]).T
    x, pca_parm = pca(x)
    x = x.T.reshape([xshape[0], xshape[2], xshape[1]]).transpose([0, 2, 1])
else:
    pca_parm = None

# train model  ------------------------------------------------
if method == 'ccl':
    train(x,
          list_hidden_nodes=list_hidden_nodes,
          initial_learning_rate=initial_learning_rate,
          momentum=momentum,
          max_steps=max_steps,
          decay_steps=decay_steps,
          decay_factor=decay_factor,
          batch_size=batch_size,
          train_dir=train_dir,
          weight_decay=weight_decay,
          checkpoint_steps=checkpoint_steps,
          moving_average_decay=moving_average_decay,
          summary_steps=summary_steps,
          device=device,
          random_seed=random_seed)
elif method == 'cclalt':
    train_alt(x,
              pair=pair,
              list_hidden_nodes=list_hidden_nodes,
              initial_learning_rate=initial_learning_rate,
              momentum=momentum,
              max_steps=max_steps,
              decay_steps=decay_steps,
              decay_factor=decay_factor,
              batch_size=batch_size,
              train_dir=train_dir,
              weight_decay=weight_decay,
              checkpoint_steps=checkpoint_steps,
              moving_average_decay=moving_average_decay,
              summary_steps=summary_steps,
              device=device,
              random_seed=random_seed)


# save parameters necessary for evaluation --------------------
model_parm = {'random_seed': random_seed,
              'num_node': num_node,
              'num_dim': num_dim,
              'num_data': num_data,
              'q_lambda1': q_lambda1,
              'q_lambda2': q_lambda2,
              'qbar_lambda1': qbar_lambda1,
              'qbar_lambda2': qbar_lambda2,
              'num_neighbor': num_neighbor,
              'num_layer': num_layer,
              'list_hidden_nodes': list_hidden_nodes,
              'method': method,
              'pair': pair if 'pair' in locals() else None,
              'moving_average_decay': moving_average_decay,
              'pca_parm': pca_parm,
              'num_latent': num_latent if 'num_latent' in locals() else None}

print('Save parameters...')
with open(train_parm_path, 'wb') as f:
    pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)

# save as tarfile
tarname = train_dir + ".tar.gz"
archive = tarfile.open(tarname, mode="w:gz")
archive.add(train_dir, arcname="./")
archive.close()

print('done.')
