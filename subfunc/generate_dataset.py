"""Data generation"""

import sys
import numpy as np
from subfunc.showdata import *


# =============================================================
# =============================================================
def generate_dataset(num_node,
                     num_dim,
                     num_data,
                     num_layer,
                     q_lambda1=None,
                     q_lambda2=None,
                     qbar_lambda1=None,
                     qbar_lambda2=None,
                     num_neighbor=None,
                     negative_slope=0.2,
                     x_limit=1e2,
                     num_latent=None,
                     random_seed=0):
    """Generate artificial data.
    Args:
        num_node: number of nodes
        num_dim: number of dimension
        num_data: number of data
        num_layer: number of layers of mixing-MLP
        q_lambda1: range of q_lambda1
        q_lambda2: range of q_lambda2
        qbar_lambda1: range of qbar_lambda1
        qbar_lambda2: range of qbar_lambda2
        num_neighbor: number of neighbors
        negative_slope: negative slope of leaky ReLU
        x_limit: if x exceed this range, re-generate it
        num_latent: number of observable nodes (None: fully observable)
        random_seed: (option) random seed
    Returns:
        x: observed signals [data, node, modality]
        s: latent components  [data, node, comp]
        lam1: lambda1 [node, node, comp]
        lam2: lambda2 [node, node, comp]
        lambar1: lambda_bar1 [node, comp]
        lambar2: lambda_bar2 [node, comp]
    """

    stable_flag = False
    cnt = 0
    while not stable_flag:
        # change random seed
        random_seed = random_seed + num_data + num_layer*100 + cnt*10000

        # generate MLP parameters
        mlplayer = gen_mlp_parms(num_dim,
                                 num_layer,
                                 negative_slope=negative_slope,
                                 random_seed=random_seed)

        # generate network
        lam1, lam2, lambar1, lambar2 = gen_net_parms(num_node,
                                                     num_dim,
                                                     q_lambda1_range=q_lambda1,
                                                     q_lambda2_range=q_lambda2,
                                                     qbar_lambda1_range=qbar_lambda1,
                                                     qbar_lambda2_range=qbar_lambda2,
                                                     num_neighbor=num_neighbor,
                                                     random_seed=random_seed)

        # generate data
        x, s = gen_x(q_lam1=lam1,
                     q_lam2=lam2,
                     qbar_std=lambar1,
                     qbar_mean=lambar2,
                     num_data=num_data,
                     mlplayer=mlplayer,
                     negative_slope=negative_slope,
                     random_seed=random_seed)

        # check stability
        x_max = np.max(np.abs(x))
        if x_max < x_limit:
            stable_flag = True

        cnt = cnt + 1

    # mask latent confounders
    if num_latent is not None:
        num_observe = num_node - num_latent
        assert num_node % num_observe == 0
        pick_interval = int(num_node / num_observe)
        x = x[:, :, pick_interval - 1::pick_interval]
        s = s[:, :, pick_interval - 1::pick_interval]
        lam1 = lam1[pick_interval - 1::pick_interval, :, :][:, pick_interval - 1::pick_interval, :]
        lam2 = lam2[pick_interval - 1::pick_interval, :, :][:, pick_interval - 1::pick_interval, :]
        lambar1 = lambar1[pick_interval - 1::pick_interval, :]
        lambar2 = lambar2[pick_interval - 1::pick_interval, :]

    return x, s, lam1, lam2, lambar1, lambar2


# =============================================================
# =============================================================
def gen_net_parms(num_node,
                  num_dim,
                  q_lambda1_range=None,
                  q_lambda2_range=None,
                  qbar_lambda1_range=None,
                  qbar_lambda2_range=None,
                  q_lambda1=None,
                  q_lambda2=None,
                  qbar_lambda1=None,
                  qbar_lambda2=None,
                  num_neighbor=3,
                  max_interval=4,
                  norm_by_num_parents=True,
                  random_seed=0):
    """Generate graph.
    Args:
        num_node: number of nodes
        num_dim: number of dimension
        q_lambda1_range: range of cross-lambda1
        q_lambda2_range: range of cross-lambda2
        qbar_lambda1_range: range of std
        qbar_lambda2_range: range of mean
        q_lambda1: (option) if given, simply output it
        q_lambda2: (option) if given, simply output it
        qbar_lambda1: (option) if given, simply output it
        qbar_lambda2: (option) if given, simply output it
        num_neighbor: (option) number of neighbors
        max_interval: (option) maximum interval between parents
        norm_by_num_parents: normalize by number of parents, or not
        random_seed: (option) random seed
    Returns:
        q_lambda1: lambda1 of cross potential (corresponding to inverse of std) [node, node, dim]
        q_lambda2: lambda2 of cross potential [node, node, dim]
        qbar_lambda1: std of marginal distribution (only for node-1) [node, dim]
        qbar_lambda2: mean of marginal distribution (only for node-1) [node, dim]
    """

    print("Generating graph (DAG)...")

    if q_lambda1_range is None:
        q_lambda1_range = [0.7, 1]
    if q_lambda2_range is None:
        q_lambda2_range = [0, 0]
    if qbar_lambda1_range is None:
        qbar_lambda1_range = [1, 1]
    if qbar_lambda2_range is None:
        qbar_lambda2_range = [0, 0]

    # initialize random generator
    np.random.seed(random_seed)

    # generate modulation
    if q_lambda1 is None:
        q_lambda1 = np.zeros([num_node, num_node, num_dim])
        for i in range(num_dim):
            # choose sources randomly for each node (with enough interval, and no common parents)
            num_source = num_neighbor
            for j in range(num_node):
                redundant = True
                while redundant:
                    palist = []
                    for k in range(num_source):
                        pacands = np.arange(max(0, j - (k + 1) * max_interval), min(min(palist), j - k * max_interval) if palist else j - k * max_interval)
                        pacands = np.setdiff1d(pacands, palist)
                        if k != 0:
                            papaids = np.where(np.sum(q_lambda1[:, palist, i], axis=1) > 0)
                            pacands = np.setdiff1d(pacands, papaids)
                        paid = pacands[np.random.permutation(len(pacands))][0] if len(pacands) != 0 else []
                        q_lambda1[paid, j, i] = 1
                        if not (isinstance(paid, list) and len(paid) == 0):
                            palist.append(paid)
                    # check redundancy
                    if (j > 0) and (np.min(np.sum(np.abs(q_lambda1[:, :j, i] - q_lambda1[:, j, i][:, None]), axis=0)) == 0):
                        q_lambda1[:, j, i] = 0
                    else:
                        redundant = False
        # scaling
        q_lambda1 = q_lambda1 * np.random.uniform(q_lambda1_range[0], q_lambda1_range[1], size=q_lambda1.shape)
        # std to lambda (for std-modulation)
        q_lambda1[q_lambda1 != 0] = 1 / (2 * q_lambda1[q_lambda1 != 0]**2)  # gauss
        if norm_by_num_parents:
            num_parents = np.sum(q_lambda1 > 0, axis=0)
            num_parents[num_parents == 0] = 1
            q_lambda1 = q_lambda1 / num_parents[None, :, :]
    if q_lambda2 is None:
        q_lambda2 = q_lambda1 != 0
        q_lambda2 = q_lambda2 * np.random.uniform(q_lambda2_range[0], q_lambda2_range[1], size=q_lambda2.shape)
        if norm_by_num_parents:
            q_lambda2 = q_lambda2 * num_parents[None, :, :]
    if qbar_lambda1 is None:
        qbar_lambda1 = np.random.uniform(qbar_lambda1_range[0], qbar_lambda1_range[1], [num_node, num_dim])
    if qbar_lambda2 is None:
        qbar_lambda2 = np.random.uniform(qbar_lambda2_range[0], qbar_lambda2_range[1], [num_node, num_dim])

    return q_lambda1, q_lambda2, qbar_lambda1, qbar_lambda2


# =============================================================
# =============================================================
def gen_x(q_lam1,
          q_lam2,
          qbar_std,
          qbar_mean,
          num_data,
          mlplayer,
          nonlinearity='ReLU',
          negative_slope=None,
          random_seed=0):
    """Generate latent components and observations
    Args:
        q_lam1: lambda1 of cross potential (corresponding to inverse of std) [node, node, dim]
        q_lam2: lambda2 of cross potential [node, node, dim]
        qbar_std: std of marginal distribution (only for node-1) [node, dim]
        qbar_mean: mean of marginal distribution (only for node-1) [node, dim]
        num_data: number of data
        mlplayer: parameters of mixing layers (gen_mlp_parms)
        nonlinearity: nonlinearity of cross potential
        negative_slope: negative slope of leaky ReLU
        random_seed: (option) random seed
    Returns:
        x: observations. 3D ndarray [num_data, num_dim, num_node]
        s: latent components. 3D ndarray [num_data, num_dim, num_node]
    """

    # initialize random generator
    np.random.seed(random_seed)

    num_node, _, num_dim = q_lam1.shape

    assert nonlinearity in {'ReLU'}

    smat = np.zeros([num_data, num_dim, num_node])
    siglist = np.zeros([num_data, num_dim, num_node])
    mulist = np.zeros([num_data, num_dim, num_node])
    for t in range(num_data):
        if (t + 1) % 1000 == 0:
            sys.stdout.write('\rGenerating s... %d/%d' % (t + 1, num_data))
            sys.stdout.flush()

        # initialize s
        st = np.zeros([num_node, num_dim])

        # generate for each node
        for n in range(num_node):
            if n == 0:
                # the first node
                st[n, :] = np.random.normal(qbar_mean[n, :], qbar_std[n, :])
            else:
                a1 = np.sum(q_lam1[:, n, :], axis=0)
                if nonlinearity == 'ReLU':
                    st_relu = st.copy()
                    st_relu[st_relu < 0] = 0
                    a2 = np.sum(2 * q_lam1[:, n, :] * q_lam2[:, n, :] * st_relu, axis=0)

                sig = 1 / np.sqrt(2 * a1)
                mu = - a2 / (2 * a1)
                sn = np.random.normal(mu, sig)

                st[n, :] = sn
                siglist[t, :, n] = sig
                mulist[t, :, n] = mu

        smat[t, :, :] = st.T

    sys.stdout.write('\r\n')

    s = smat  # [data, dim, node]

    # apply MLP
    if len(mlplayer) > 0:
        sperm = np.transpose(s, [0, 2, 1]).reshape([-1, num_dim])  # [data*node, dim]
        x = apply_mlp(sperm, mlplayer, negative_slope=negative_slope)
        x = x.reshape([num_data, num_node, num_dim]).transpose(0, 2, 1)  # [data, dim, node]
    else:
        x = s.copy()

    return x, s


# =============================================================
# =============================================================
def gen_mlp_parms(num_dim,
                  num_layer,
                  iter4condthresh=10000,
                  cond_thresh_ratio=0.25,
                  layer_name_base='ip',
                  negative_slope=None,
                  random_seed=0):
    """Generate MLP and Apply it to source signal.
    Args:
        num_dim: number of dimensions
        num_layer: number of layers
        iter4condthresh: (option) number of random iteration to decide the threshold of condition number of mixing matrices
        cond_thresh_ratio: (option) percentile of condition number to decide its threshold
        layer_name_base: (option) layer name
        negative_slope: negative slope of leakyReLU (for properly scaling weights)
        random_seed: (option) random seed
    Returns:
        mixlayer: parameters of mixing layers
    """

    print("Generating gnn parameters...")

    # initialize random generator
    np.random.seed(random_seed)

    # generate W
    def genw(num_in, num_out, nonlin=True):
        wf = np.random.uniform(-1, 1, [num_out, num_in])
        if nonlin:
            wf = wf * np.sqrt(6/((1 + negative_slope**2)*num_in))
        else:
            wf = wf * np.sqrt(6/(num_in*2))
        return wf

    # Determine condThresh
    condlist = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        w = genw(num_dim, num_dim)
        condlist[i] = np.linalg.cond(w)
    condlist.sort()
    cond_thresh = condlist[int(iter4condthresh * cond_thresh_ratio)]
    print("    cond thresh: {0:f}".format(cond_thresh))

    mixlayer = []
    for ln in range(num_layer):
        condw = cond_thresh + 1
        while condw > cond_thresh:
            if ln == 0:  # 1st layer
                w = genw(num_dim, num_dim, nonlin=(num_layer != 1))
            elif ln == num_layer-1:  # last layer
                w = genw(num_dim, num_dim, nonlin=False)
            else:
                w = genw(num_dim, num_dim, nonlin=True)
            condw = np.linalg.cond(w)
        print("    L{0:d}: cond={1:f}".format(ln, condw))
        b = np.zeros(w.shape[-1]).reshape([-1, 1])
        # storege
        layername = layer_name_base + str(ln)
        mixlayer.append({"name": layername, "W": w.copy(), "b": b.copy()})

    return mixlayer


# =============================================================
# =============================================================
def apply_mlp(x,
              mlplayer,
              nonlinear_type='ReLU',
              negative_slope=None):
    """Generate MLP and Apply it to source signal.
    Args:
        x: input signals. 2D ndarray [num_comp, num_data]
        mlplayer: parameters of MLP generated by gen_mlp_parms
        nonlinear_type: (option) type of nonlinearity
        negative_slope: (option) parameter of leaky-ReLU
    Returns:
        y: mixed signals. 2D ndarray [num_comp, num_data]
    """

    num_layer = len(mlplayer)

    # generate mixed signal
    y = x.copy()
    for ln in range(num_layer):

        # apply bias and mixing matrix
        y = y + mlplayer[ln]['b'].reshape([1, -1])
        y = np.dot(y, mlplayer[ln]['W'].T)

        # apply nonlinearity
        if ln != num_layer-1:  # no nolinearity for the last layer
            if nonlinear_type == "ReLU":  # leaky-ReLU
                y[y < 0] = negative_slope * y[y < 0]
            else:
                raise ValueError

    return y

