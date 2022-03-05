# ######################################################### #
# ######################################################### #
# THE PROCEDURE PERFORMED HERE TO SEEK TO THE BEST MODEL
# IS A VERY DUMMY ONE !!!
# NO STRATEGY HERE, I MEAN IT IS JUST BRAINLESS STRATEGY:
#      * Bunch of models (no use of correlation or mutual information!)
#      * training and evaluating, hoping for the best
#
# <<< /!\ DO NOT PERFROM MODEL SELECTION THIS WAY             >>>
# <<<     it is just to get XP school attached to the project >>>
# The only small interesting thing here is the multiprocessing ...
# ######################################################### #
# ######################################################### #

# ######################################################### #
#                    LIBRARIES IMPORT                       #
# ######################################################### #

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('QtAgg')

# Pour le multiprocessing
import concurrent.futures
from multiprocessing import cpu_count


path = os.path.join(os.path.dirname(__file__), '..', 'ex00')
sys.path.insert(1, path)
from polynomial_model_extended import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from ridge_v2 import MyRidge
from data_spliter import data_spliter
from scaler import MyStandardScaler


# ######################################################### #
#                        CONSTANTES                         #
# ######################################################### #

n_cpu = cpu_count()
cpu_use = int(2 * n_cpu / 3)

lst_check_feat = ["weight", "prod_distance", "time_delivery"]
lst_dataset = lst_check_feat + ["target"]

col2idx = {'w': 0,
           'p': 1,
           't': 2,
           'w2': 3,
           'p2': 4,
           't2': 5,
           'w3': 6,
           'p3': 7,
           't3': 8,
           'w4': 9,
           'p4': 10,
           't4': 11,
           'wp': 12,
           'w2p': 13,
           'wp2': 14,
           'w2p2': 15,
           'wt': 16,
           'w2t': 17,
           'wt2': 18,
           'w2t2': 19,
           'pt': 20,
           'p2t': 21,
           'pt2': 22,
           'p2t2': 23}

nb_steps = 10000

# ######################################################### #
#                  FUNCTION DEFINITIONS                     #
# ######################################################### #


def data_idx(cols):
    lst = []
    for c in cols:
        lst.append(col2idx[c])
    return np.array(lst, dtype=int)


def loss_report(x: np.array, y: np.array, lst_models: list):
    lst_loss = []

    print('\n' + '#' * 6 + ' Score report ' + '#' * 5 + '| MSE + regularization:')
    for model in lst_models:
        pred = model.predict_(x[:, data_idx(model._vars_)])
        loss = model.loss_(pred, y)
        lst_loss.append(loss)
        print(model._tag_.ljust(25) + f"| {loss:.5f}")

    return lst_loss


# ######################################################### #
#                             MAIN                          #
# ######################################################### #
if __name__ == "__main__":
    print("CPU USE: ", cpu_use)
    # Importation of the dataset + basic checking:
    try:
        data = pd.read_csv("space_avocado.csv", index_col=0, dtype=np.float64)
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()

    if any([c not in lst_dataset for c in data.columns]):
        print("At least a missing expected columns.", file=sys.stderr)
        sys.exit()

    if any([dt.kind not in ['i', 'f'] for dt in data.dtypes]):
        s = "At least one column is not of expected kind dtype."
        print(s, file=sys.stderr)
        sys.exit()

    data.rename(columns={'weight': 'w',
                         'prod_distance': 'p',
                         'time_delivery': 't'},
                inplace=True)
    # Data augmentation: the subject specifies we need to use
    # add_polynomial_features method:
    # But the use of numpy.polynomila.polynomial.polyvander2d would be wiser.
    w_1to4 = add_polynomial_features(data.w.values.reshape(-1, 1), 4)
    p_1to4 = add_polynomial_features(data.p.values.reshape(-1, 1), 4)
    t_1to4 = add_polynomial_features(data.t.values.reshape(-1, 1), 4)

    for ii in range(1, 4):
        data[f'w{ii + 1}'] = w_1to4[:, ii]
        data[f'p{ii + 1}'] = p_1to4[:, ii]
        data[f't{ii + 1}'] = t_1to4[:, ii]

    data['wp'] = data['w'] * data['p']
    data['w2p'] = (data['w'] ** 2) * data['p']
    data['wp2'] = data['w'] * (data['p'] ** 2)
    data['w2p2'] = (data['w'] ** 2) * (data['p'] ** 2)

    data['wt'] = data['w'] * data['t']
    data['w2t'] = (data['w'] ** 2) * data['t']
    data['wt2'] = data['w'] * (data['t'] ** 2)
    data['w2t2'] = (data['w'] ** 2) * (data['t'] ** 2)

    data['pt'] = data['p'] * data['t']
    data['p2t'] = (data['p'] ** 2) * data['t']
    data['pt2'] = data['p'] * (data['t'] ** 2)
    data['p2t2'] = (data['p'] ** 2) * (data['t'] ** 2)

    cols = data.columns.values
    cols = cols[cols != 'target']
    Xs = data_spliter(data[cols].values,
                      data["target"].values.reshape(-1, 1), 0.8)
    x_train, x_test, y_train, y_test = Xs

    # We split the test set in 2: a real test set and cross validation set.
    # We should do it on the training set, especially if we would do k-fold CV,
    # but it will not be the case here
    sep = int(np.floor(0.5 * x_test.shape[0]))
    x_cross, y_cross = x_test.copy()[:sep], y_test.copy()[:sep]
    x_test, y_test = x_test.copy()[sep:], y_test.copy()[sep:]

    scaler_x = MyStandardScaler()
    scaler_y = MyStandardScaler()
    scaler_x.fit(x_train)
    scaler_y.fit(y_train)

    x_train_tr = scaler_x.transform(x_train)
    y_train_tr = scaler_y.transform(y_train)
    x_cross_tr = scaler_x.transform(x_cross)
    y_cross_tr = scaler_y.transform(y_cross)
    x_test_tr = scaler_x.transform(x_test)
    y_test_tr = scaler_y.transform(y_test)

    # ###################################################################### #
    #                            First Bath of models                        #
    # ###################################################################### #
    # Simple models:
    ridges_w = [MyRidge(np.random.rand(2, 1),alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_p = [MyRidge(np.random.rand(2, 1),alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_t = [MyRidge(np.random.rand(2, 1),alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    ridges_wp = [MyRidge(np.random.rand(3, 1),alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_wt = [MyRidge(np.random.rand(3, 1),alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_pt = [MyRidge(np.random.rand(3, 1),alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    ridges_wpt = [MyRidge(np.random.rand(4, 1),alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    simple_models = [ridges_w, ridges_p, ridges_t, ridges_wp, ridges_wt, ridges_pt, ridges_wpt]
    lst_vars1 = [['w'], ['p'], ['t'],
                 ['w', 'p'], ['w', 't'], ['p', 't'], ['w', 'p', 't']]

    batch1_future = []
    batch1_trained = []
    nb = len(simple_models) * 6
    s_state = ['[ ]'] * nb
    count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_use) as executor:
        # Starting/Exectuting processes
        for ii, models, vars in zip(range(nb), simple_models, lst_vars1):
            for jj, model in zip(range(6), models):
                print(f"Batch simple: starting model {count + 1} / {nb}",
                      end='\r',
                      flush=True)
            
                # model._tag_, model._idx_ = f"batch_1_model_{ii + 1}_reg_{2 * jj:0>#2d}", count
                model._tag_, model._idx_ = f"b1_mdl{ii + 1}_reg{2 * jj:0>#2d}", count
                model._vars_ = vars
                batch1_future.append(executor.submit(model.fit_,
                                    x_train_tr[:, data_idx(vars)],
                                    y_train_tr))
                count += 1
        print('\n')
        # Action when process are completed
        # (printing the state string to have an idea of the remaining train)
        for task in concurrent.futures.as_completed(batch1_future):
            if task.result() is not None:
                batch1_trained.append(task.result())
                s_state[task.result()._idx_] = '[✔]'
                print('Simple batch: ' + ' '.join(s_state),
                      end='\r',
                      flush=True)

    # ###################################################################### #
    #                           Second Bath of models                        #
    # ###################################################################### #
    # 'intermediate' models
    ridges_w2 = [MyRidge(np.random.rand(3, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_w3 = [MyRidge(np.random.rand(4, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_w4 = [MyRidge(np.random.rand(5, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    ridges_p2 = [MyRidge(np.random.rand(3, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_p3 = [MyRidge(np.random.rand(4, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_p4 = [MyRidge(np.random.rand(5, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    ridges_t2 = [MyRidge(np.random.rand(3, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_t3 = [MyRidge(np.random.rand(4, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_t4 = [MyRidge(np.random.rand(5, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    ridges_w_p_2 = [MyRidge(np.random.rand(5, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_w_p_3 = [MyRidge(np.random.rand(7, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_w_p_4 = [MyRidge(np.random.rand(9, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    ridges_w_t_2 = [MyRidge(np.random.rand(5, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_w_t_3 = [MyRidge(np.random.rand(7, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_w_t_4 = [MyRidge(np.random.rand(9, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    ridges_p_t_2 = [MyRidge(np.random.rand(5, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_p_t_3 = [MyRidge(np.random.rand(7, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]
    ridges_p_t_4 = [MyRidge(np.random.rand(9, 1), alpha=1e-2, max_iter=nb_steps, lambda_ = 0.2 * ii) for ii in range(6)]

    intermediate_models = [ridges_w2, ridges_w3, ridges_w4, ridges_p2, ridges_p3, ridges_p4, ridges_t2,
                           ridges_t3, ridges_t4, ridges_w_p_2, ridges_w_p_3, ridges_w_p_4,
                           ridges_w_t_2, ridges_w_t_3, ridges_w_t_4, ridges_p_t_2, ridges_p_t_3,
                           ridges_p_t_4]

    lst_vars2 = [['w', 'w2'],
                 ['w', 'w2', 'w3'],
                 ['w', 'w2', 'w3', 'w4'],
                 ['p', 'p2'],
                 ['p', 'p2', 'p3'],
                 ['p', 'p2', 'p3', 'p4'],
                 ['t', 't2'],
                 ['t', 't2', 't3'],
                 ['t', 't2', 't3', 't4'],
                 ['w', 'w2', 'p', 'p2'],
                 ['w', 'w2', 'w3', 'p', 'p2', 'p3'],
                 ['w', 'w2', 'w3', 'w4', 'p', 'p2', 'p3', 'p4'],
                 ['w', 'w2', 't', 't2'],
                 ['w', 'w2', 'w3', 't', 't2', 't3'],
                 ['w', 'w2', 'w3', 'w4', 't', 't2', 't3', 't4'],
                 ['p', 'p2', 't', 't2'],
                 ['p', 'p2', 'p3', 't', 't2', 't3'],
                 ['p', 'p2', 'p3', 'p4', 't', 't2', 't3', 't4']]

    # I could do a wrapping function taking only the x_train_tr, y_train_tr
    # and models and performing the full multi process training
    batch2_future = []
    batch2_trained = []
    nb = len(intermediate_models) * 6
    count = 0
    completed = 0
    print('\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_use) as executor:
        for ii, models, vars in zip(range(nb), intermediate_models, lst_vars2):
            for jj, model in zip(range(6), models):
                print(f"Batch intermediate: starting model {count + 1} / {nb}",
                    end='\r',
                    flush=True)
                # model._tag_, model._idx_ = f"batch_2_model_{ii + 1}_reg_{2 * jj:0>#2d}", count
                model._tag_, model._idx_ = f"b2_mdl{ii + 1}_reg{2 * jj:0>#2d}", count
                model._vars_ = vars
                batch2_future.append(executor.submit(model.fit_,
                                                    x_train_tr[:, data_idx(vars)],
                                                    y_train_tr))
                count += 1
        print('\n')
        for task in concurrent.futures.as_completed(batch2_future):
            if task.result() is not None:
                completed += 1
                batch2_trained.append(task.result())
                print(f'Intermediate batch completed: {completed} / {nb}',
                      end='\r',
                      flush=True)

    # ###################################################################### #
    #                            Third Bath of models                        #
    # ###################################################################### #
    # 'sophisticate' models
    # models with ∑w^n n in (1,...,4) and ∑p^n n in (1,...,4) plus
    # the terms ∑t^m. 1st: t^1 - 2nd: t^1 + p^2 - 3rd: p^1 + t^2 + t^3
    ridges_WP_t1 = [MyRidge(np.random.rand(10, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WP_t2 = [MyRidge(np.random.rand(11, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WP_t3 = [MyRidge(np.random.rand(12, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WP_t4 = [MyRidge(np.random.rand(13, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]

    # models with ∑w^n n in (1,...,4) and ∑t^n n in (1,...,4) plus
    # the terms ∑p^m. 1st: p^1 - 2nd: p^1 + p^2 - 3rd: p^1 + p^2 + p^3
    ridges_WT_p1 = [MyRidge(np.random.rand(10, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WT_p2 = [MyRidge(np.random.rand(11, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WT_p3 = [MyRidge(np.random.rand(12, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]

    # models with ∑p^n n in (1,...,4) and ∑t^n n in (1,...,4) plus
    # the terms ∑w^m. 1st: w^1 - 2nd: w^1 + w^2 - 3rd: w^1 + w^2 + w^3
    ridges_PT_w1 = [MyRidge(np.random.rand(10, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_PT_w2 = [MyRidge(np.random.rand(11, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_PT_w3 = [MyRidge(np.random.rand(12, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]

    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # one cross term: 'wp', w^2p, wp^2, w^2p^2
    # and finally all the cross terms 'wp' + w^2p + wp^2 + w^2p^2
    ridges_WPT_wp = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_w2p = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_wp2 = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_w2p2 = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_WP = [MyRidge(np.random.rand(17, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]

    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # one cross term: 'wt', w^2t, wt^2, w^2t^2
    # and finally all the cross terms 'wt' + w^2t + wt^2 + w^2t^2
    ridges_WPT_wt = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_w2t = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_wt2 = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_w2t2 = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_WT = [MyRidge(np.random.rand(17, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]

    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # one cross term: 'pt', p^2t, pt^2, p^2t^2
    # and finally all the cross terms 'pt' + p^2t + pt^2 + p^2t^2
    ridges_WPT_pt = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_p2t = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_pt2 = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_p2t2 = [MyRidge(np.random.rand(14, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]
    ridges_WPT_PT = [MyRidge(np.random.rand(17, 1), alpha=1e-2, max_iter=nb_steps, lambda_=0.2 * ii) for ii in range(6)]

    # models with ∑(w^n+ p^n + t^n) n in (1,...,4) plus
    # all the cross term of wp, wt and pt
    ridges_WPT_WP_WT_PT = MyRidge(np.random.rand(25, 1), alpha=1e-2, max_iter=nb_steps)

    complex_models = [ridges_WP_t1, ridges_WP_t2, ridges_WP_t3, ridges_WP_t4,
                      ridges_WT_p1, ridges_WT_p2, ridges_WT_p3,
                      ridges_PT_w1, ridges_PT_w2, ridges_PT_w3,
                      ridges_WPT_wp, ridges_WPT_w2p, ridges_WPT_wp2, ridges_WPT_w2p2,
                      ridges_WPT_WP, ridges_WPT_wt, ridges_WPT_w2t, ridges_WPT_wt2,
                      ridges_WPT_w2t2, ridges_WPT_WT, ridges_WPT_pt, ridges_WPT_p2t,
                      ridges_WPT_pt2, ridges_WPT_p2t2, ridges_WPT_PT]

    lst_W = ['w', 'w2', 'w3', 'w4']
    lst_T = ['t', 't2', 't3', 't4']
    lst_P = ['p', 'p2', 'p3', 'p4']

    lst_WP = lst_W + lst_P
    lst_WT = lst_W + lst_T
    lst_PT = lst_P + lst_T
    lst_WPT = lst_W + lst_P + lst_T

    # My apologize, it might be painful to read: It is the list of var string
    lst_vars3 = [lst_WP + [lst_T[0]], lst_WP + lst_T[0:2], lst_WP + lst_T[:3],
                 lst_WPT, lst_WT + [lst_P[0]], lst_WT + lst_P[0:2], lst_WT + lst_P[:3],
                 lst_PT + [lst_W[0]], lst_PT + lst_W[0:2], lst_PT + lst_W[:3],
                 lst_WPT + ['wp'], lst_WPT + ['w2p'], lst_WPT + ['wp2'], lst_WPT + ['w2p2'],
                 lst_WPT + ['wp', 'w2p', 'wp2', 'w2p2'],
                 lst_WPT + ['wt'], lst_WPT + ['w2t'], lst_WPT + ['wt2'], lst_WPT + ['w2t2'],
                 lst_WPT + ['wt', 'w2t', 'wt2', 'w2t2'],
                 lst_WPT + ['pt'], lst_WPT + ['p2t'], lst_WPT + ['pt2'], lst_WPT + ['p2t2'],
                 lst_WPT + ['pt', 'p2t', 'pt2', 'p2t2'],
                 lst_WPT + ['wp', 'w2p', 'wp2', 'w2p2'] + ['wt', 'w2t', 'wt2', 'w2t2']
                 + ['pt', 'p2t', 'pt2', 'p2t2']]

    # Yep I could have done this function ...
    batch3_trained = []
    batch3_future = []
    nb = len(complex_models) * 6
    count = 0
    completed = 0
    print('\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_use) as executor:
        for ii, models, vars in zip(range(nb), complex_models, lst_vars3):
            for jj, model in zip(range(6), models):
                print(f"Batch 'complex': starting model {count + 1} / {nb}",
                    end='\r',
                    flush=True)
                model._tag_, model._idx_ = f"b3_mdl{ii + 1}_reg{2 * jj:0>#2d}", count
                model._vars_ = vars
                batch3_future.append(executor.submit(model.fit_,
                                                    x_train_tr[:, data_idx(vars)],
                                                    y_train_tr))
                count += 1
        print('\n')
        for task in concurrent.futures.as_completed(batch3_future):
            if task.result() is not None:
                completed += 1
                batch3_trained.append(task.result())
                print(f"complex batch completed: {completed} / {nb}",
                      end='\r',
                      flush=True)

    print('\n')
    lst_loss = loss_report(x_cross_tr, y_cross_tr,
                           batch1_trained + batch2_trained + batch3_trained)

    min = lst_loss[0]
    min_loss_model = batch1_trained[0]
    # Looking for the model with the minimal loss
    # Converting thetas array into a serializable object
    for loss, model in zip(lst_loss, batch1_trained + batch2_trained + batch3_trained):
        if loss < min:
            min = loss
            min_loss_model = model
        model._loss_ = loss
    print(f"model with the lowest loss: {min_loss_model._tag_}({min})")

    dcts_models = {}
    for model in batch1_trained + batch2_trained + batch3_trained:
        dcts_models[model._tag_] = model.__dict__

    # Saving the models
    with open("models_test.pickle", "wb") as outfile:
        pickle.dump(dcts_models, outfile)

    fig, axes = plt.subplots(2, 1, figsize=(25, 14))
    plt.title("MSE vs explored models")

    nb_models = len(batch1_trained + batch2_trained + batch3_trained)
    half = nb_models // 2
    lst_tags = [model._tag_ for model in batch1_trained + batch2_trained + batch3_trained]
    axes[0].scatter(lst_tags[:half], lst_loss[:half])
    axes[1].scatter(lst_tags[half:], lst_loss[half:])
    
    fig.tight_layout(pad=14.0)
    axes[0].set_xlabel("models")
    axes[0].set_ylabel("MSE")
    axes[1].set_xlabel("models")
    axes[1].set_ylabel("MSE")
    
    plt.setp(axes[0].get_xticklabels(), rotation=90)
    plt.setp(axes[1].get_xticklabels(), rotation=90)

    axes[0].set_ylim([0, 0.9])
    axes[1].set_ylim([0, 0.5])
    axes[0].grid()
    axes[1].grid()
    #plt.subplots_adjust(bottom=0.22)
    plt.savefig("loss_vs_models_scoring_on_CV_sets.svg", dpi=500)
    plt.show()
