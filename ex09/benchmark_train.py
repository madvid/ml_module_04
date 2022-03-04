import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if mpl.get_backend() == 'agg':
    mpl.use('QtAgg')

path = os.path.join(os.path.dirname(__file__), '..', 'ex08')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLogR

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from data_spliter import data_spliter
from scaler import MyStandardScaler, MyMinMaxScaler
from logistic_related import *
from other_metrics import f1_score_
from confusion_matrix import confusion_matrix_

# ########################################################################## #
#  _____________________________ CONSTANTES ________________________________ #
# ########################################################################## #
file_x = "solar_system_census.csv"
file_y = "solar_system_census_planets.csv"

dct_labels = {"Venus": 0,
              "Earth": 1,
              "Mars": 2,
              "Asteroids' Belt": 3}

s_nb = ['0', '1', '2', '3']

# ########################################################################## #
#  ______________________________ FUNCTIONS ________________________________ #
# ########################################################################## #

# >>> Usage <<<
def usage():
    s = "usgage: python benchmark.py"
    print(s)


# >>> Preprocessing related methods <<<
def labelbinarizer(y, target):
    y_ = np.zeros(y.shape, dtype='int8')
    y_[y == target] = 1
    return y_


def binarize(y, threshold=0.0, copy=True):
    """Cheap mimic of the binarize method from sklearn.preprocessing module
    """
    if copy:
        y_ = np.zeros(y.shape, dtype=np.int8)
        y_[y >= threshold] = 1
        return y_
    y[y >= threshold] = 1
    y[y < threshold] = 0

def multiclass_f1_score_(y, y_hat, labels):
    """
    Compute the f1 score for a multi label classification situation.
    Args:
        y:a [numpy.array] for the correct labels
        y_hat: [numpy.array] for the predicted labels
        pos_label: [str | int ...], classes on which multi label f1_score 
            is calculated
    Return:
        The f1 score as a float.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    multiclass_f1 = 0
    for l in labels:
        multiclass_f1 += f1_score_(y, y_hat, l) * np.sum(y == l) / y.shape[0]
    return multiclass_f1


# ########################################################################## #
#  ________________________________ MAIN ___________________________________ #
# ########################################################################## #

if __name__ == "__main__":
    # Importing data:
    try:
        x = pd.read_csv(file_x, index_col=0)
        y = pd.read_csv(file_y, index_col=0)
    except:
        s = "Issue while reading one of the dataset."
        print(s, file=sys.stderr)
        sys.exit()

    try:
        # casting the y data
        # 2 reasons: minimizing the memory space
        #            if casting fails it means y is not numeric only
        y = y.to_numpy(dtype=np.int8)
    except:
        s = "Something wrong when casting data to integer."
        print(s, file=sys.stderr)
        sys.exit()

    if x.shape[0] != y.shape[0]:
        s = f"Unmatching number of lines between {file_x} and {file_y}"
        print(s, file=sys.stderr)
        sys.exit()

    # A bit of data augmentation
    x['height2'] = x['height'] ** 2
    x['height3'] = x['height'] ** 3
    x['weight2'] = x['weight'] ** 2
    x['weight3'] = x['weight'] ** 3
    x['bone_density2'] = x['bone_density'] ** 2
    x['bone_density3'] = x['bone_density'] ** 3


    # Spliting the data into a training a test set
    x_tmp, x_test, y_tmp, y_test = data_spliter(x.values, y, 0.85)
    
    
    # We split the training set in 2: a real training set and cross validation set.
    x_train, y_train = x_tmp.copy()[:-y_test.shape[0]], y_tmp.copy()[:-y_test.shape[0]]
    x_cross, y_cross = x_tmp.copy()[-y_test.shape[0]:], y_tmp.copy()[-y_test.shape[0]:]
    print(f'train set size: {y_train.shape} -- cv set size: {y_cross.shape} -- test set size: {y_test.shape}')

    # Preprocessing (simple standardistation of the features)
    scaler_x = MyStandardScaler()
    scaler_x.fit(x_train)

    x_train_tr = scaler_x.transform(x_train)
    x_cross_tr = scaler_x.transform(x_cross)
    x_test_tr = scaler_x.transform(x_test)

    # Instanciation and training of the models
    thetas_Venus = np.random.rand(x.shape[1] + 1, 1)
    models_Venus = [MyLogR(thetas_Venus, lambda_=l_ii * 0.2) for l_ii in range(6)]
    thetas_Earth = np.random.rand(x.shape[1] + 1, 1)
    models_Earth = [MyLogR(thetas_Earth, lambda_=l_ii * 0.2) for l_ii in range(6)]
    thetas_Mars = np.random.rand(x.shape[1] + 1, 1)
    models_Mars = [MyLogR(thetas_Mars, lambda_=l_ii * 0.2) for l_ii in range(6)]
    thetas_AstroBelt = np.random.rand(x.shape[1] + 1, 1)
    models_AstroBelt = [MyLogR(thetas_AstroBelt, lambda_=l_ii * 0.2) for l_ii in range(6)]

    # Preparation of the lists for the prediction of the models afeter training
    preds_Venus = preds_Earth = preds_Mars = preds_AstroBelt = []
    for ii in range(6):
        # Setting learning rate and number of iterations for each models
        dct_params = {'alpha': 1e-1, 'max_iter': 2000}
        models_Venus[ii].set_params_(dct_params)
        models_Venus[ii]._tag_ = f'Venus_reg_{ii * 0.2:.2f}'
        models_Earth[ii].set_params_(dct_params)
        models_Earth[ii]._tag_ = f'Earth_reg_{ii * 0.2:.2f}'
        models_Mars[ii].set_params_(dct_params)
        models_Mars[ii]._tag_ = f'Mars_reg_{ii * 0.2:.2f}'
        models_AstroBelt[ii].set_params_(dct_params)
        models_AstroBelt[ii]._tag_ = f'AstroBelt_reg_{ii * 0.2:.2f}'

        # Training of the models
        print(f"model Venus:".ljust(20) + f"{ii} / 6", file=sys.stderr, end='\r')
        models_Venus[ii].fit_(x_train_tr, labelbinarizer(y_train, 0))
        print(f"model Earth:".ljust(20) + f"{ii} / 6", file=sys.stderr, end='\r')
        models_Earth[ii].fit_(x_train_tr, labelbinarizer(y_train, 1))
        print(f"model Mars:".ljust(20) + f"{ii} / 6", file=sys.stderr, end='\r')
        models_Mars[ii].fit_(x_train_tr, labelbinarizer(y_train, 2))
        print(f"model AstroBelt:".ljust(20) + f"{ii} / 6", file=sys.stderr, end='\r')
        models_AstroBelt[ii].fit_(x_train_tr, labelbinarizer(y_train, 3))
        # Prediction and binarization of the probabilities for all the models
        if ii == 0:
            preds_Venus = models_Venus[ii].predict_(x_cross_tr)
            preds_Earth = models_Earth[ii].predict_(x_cross_tr)
            preds_Mars = models_Mars[ii].predict_(x_cross_tr)
            preds_AstroBelt = models_AstroBelt[ii].predict_(x_cross_tr)
        else:
            preds_Venus = np.hstack((preds_Venus, models_Venus[ii].predict_(x_cross_tr)))
            preds_Earth = np.hstack((preds_Earth, models_Earth[ii].predict_(x_cross_tr)))
            preds_Mars = np.hstack((preds_Mars, models_Mars[ii].predict_(x_cross_tr)))
            preds_AstroBelt = np.hstack((preds_AstroBelt, models_AstroBelt[ii].predict_(x_cross_tr)))

    # stacking and calculating the one vs all prediction
    preds_reg00 = np.c_[preds_Venus[:,0], preds_Earth[:,0], preds_Mars[:,0], preds_AstroBelt[:,0]]
    preds_reg02 = np.c_[preds_Venus[:,1], preds_Earth[:,1], preds_Mars[:,1], preds_AstroBelt[:,1]]
    preds_reg04 = np.c_[preds_Venus[:,2], preds_Earth[:,2], preds_Mars[:,2], preds_AstroBelt[:,2]]
    preds_reg06 = np.c_[preds_Venus[:,3], preds_Earth[:,3], preds_Mars[:,3], preds_AstroBelt[:,3]]
    preds_reg08 = np.c_[preds_Venus[:,4], preds_Earth[:,4], preds_Mars[:,4], preds_AstroBelt[:,4]]
    preds_reg10 = np.c_[preds_Venus[:,5], preds_Earth[:,5], preds_Mars[:,5], preds_AstroBelt[:,5]]
    oneVsAll_pred_reg00 = np.argmax(preds_reg00, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg02 = np.argmax(preds_reg02, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg04 = np.argmax(preds_reg04, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg06 = np.argmax(preds_reg06, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg08 = np.argmax(preds_reg08, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg10 = np.argmax(preds_reg10, axis=1).reshape(-1, 1)

    # Calcul of the fraction of correct prediction
    f1_reg00 = multiclass_f1_score_(y_cross.astype(int), oneVsAll_pred_reg00, [1, 2, 3, 4])
    f1_reg02 = multiclass_f1_score_(y_cross.astype(int), oneVsAll_pred_reg02, [1, 2, 3, 4])
    f1_reg04 = multiclass_f1_score_(y_cross.astype(int), oneVsAll_pred_reg04, [1, 2, 3, 4])
    f1_reg06 = multiclass_f1_score_(y_cross.astype(int), oneVsAll_pred_reg06, [1, 2, 3, 4])
    f1_reg08 = multiclass_f1_score_(y_cross.astype(int), oneVsAll_pred_reg08, [1, 2, 3, 4])
    f1_reg10 = multiclass_f1_score_(y_cross.astype(int), oneVsAll_pred_reg10, [1, 2, 3, 4])
    s = "f1-score of regularized models with lambda_="
    print(s + '0.0: ' + f'{f1_reg00:.4f}')
    print(s + '0.2: ' + f'{f1_reg02:.4f}')
    print(s + '0.4: ' + f'{f1_reg04:.4f}')
    print(s + '0.6: ' + f'{f1_reg06:.4f}')
    print(s + '0.8: ' + f'{f1_reg08:.4f}')
    print(s + '1.0: ' + f'{f1_reg10:.4f}')
    
    # If someone want to take a look to the loss, because the f1-score is the same despite
    # change in regulariaztion factor.
    # s = 'value of the loss from model '
    # y_= y_cross.astype(int)
    # print(s + 'models_Earth[0] =', models_Earth[0]._loss_(y_, preds_Earth[:,0].reshape(-1,1)))
    # print(s + 'models_Earth[1] =', models_Earth[1]._loss_(y_, preds_Earth[:,1].reshape(-1,1)))
    # print(s + 'models_Earth[2] =', models_Earth[2]._loss_(y_, preds_Earth[:,2].reshape(-1,1)))
    # print(s + 'models_Earth[3] =', models_Earth[3]._loss_(y_, preds_Earth[:,3].reshape(-1,1)))
    # print(s + 'models_Earth[4] =', models_Earth[4]._loss_(y_, preds_Earth[:,4].reshape(-1,1)))
    # print(s + 'models_Earth[5] =', models_Earth[5]._loss_(y_, preds_Earth[:,5].reshape(-1,1)))

    # print(s + 'models_Venus[0] =', models_Venus[0]._loss_(y_, preds_Venus[:,0].reshape(-1,1)))
    # print(s + 'models_Venus[1] =', models_Venus[1]._loss_(y_, preds_Venus[:,1].reshape(-1,1)))
    # print(s + 'models_Venus[2] =', models_Venus[2]._loss_(y_, preds_Venus[:,2].reshape(-1,1)))
    # print(s + 'models_Venus[3] =', models_Venus[3]._loss_(y_, preds_Venus[:,3].reshape(-1,1)))
    # print(s + 'models_Venus[4] =', models_Venus[4]._loss_(y_, preds_Venus[:,4].reshape(-1,1)))
    # print(s + 'models_Venus[5] =', models_Venus[5]._loss_(y_, preds_Venus[:,5].reshape(-1,1)))

    # print(s + 'models_Mars[0] =', models_Mars[0]._loss_(y_, preds_Mars[:,0].reshape(-1,1)))
    # print(s + 'models_Mars[1] =', models_Mars[1]._loss_(y_, preds_Mars[:,1].reshape(-1,1)))
    # print(s + 'models_Mars[2] =', models_Mars[2]._loss_(y_, preds_Mars[:,2].reshape(-1,1)))
    # print(s + 'models_Mars[3] =', models_Mars[3]._loss_(y_, preds_Mars[:,3].reshape(-1,1)))
    # print(s + 'models_Mars[4] =', models_Mars[4]._loss_(y_, preds_Mars[:,4].reshape(-1,1)))
    # print(s + 'models_Mars[5] =', models_Mars[5]._loss_(y_, preds_Mars[:,5].reshape(-1,1)))

    # print(s + 'models_AstroBelt[0] =', models_AstroBelt[0]._loss_(y_, preds_AstroBelt[:,0].reshape(-1,1)))
    # print(s + 'models_AstroBelt[1] =', models_AstroBelt[1]._loss_(y_, preds_AstroBelt[:,1].reshape(-1,1)))
    # print(s + 'models_AstroBelt[2] =', models_AstroBelt[2]._loss_(y_, preds_AstroBelt[:,2].reshape(-1,1)))
    # print(s + 'models_AstroBelt[3] =', models_AstroBelt[3]._loss_(y_, preds_AstroBelt[:,3].reshape(-1,1)))
    # print(s + 'models_AstroBelt[4] =', models_AstroBelt[4]._loss_(y_, preds_AstroBelt[:,4].reshape(-1,1)))
    # print(s + 'models_AstroBelt[5] =', models_AstroBelt[5]._loss_(y_, preds_AstroBelt[:,5].reshape(-1,1)))

    # If someone want to take a look to the theta values, because the f1-score
    # is the same despite change in regulariaztion factor.
    # print(s + 'models_Earth[0] =', models_Earth[0].theta.reshape(1,-1))
    # print(s + 'models_Earth[1] =', models_Earth[1].theta.reshape(1,-1))
    # print(s + 'models_Earth[2] =', models_Earth[2].theta.reshape(1,-1))
    # print(s + 'models_Earth[3] =', models_Earth[3].theta.reshape(1,-1))
    # print(s + 'models_Earth[4] =', models_Earth[4].theta.reshape(1,-1))
    # print(s + 'models_Earth[5] =', models_Earth[5].theta.reshape(1,-1))

    # print(s + 'models_Venus[0] =', models_Venus[0].theta.reshape(1,-1))
    # print(s + 'models_Venus[1] =', models_Venus[1].theta.reshape(1,-1))
    # print(s + 'models_Venus[2] =', models_Venus[2].theta.reshape(1,-1))
    # print(s + 'models_Venus[3] =', models_Venus[3].theta.reshape(1,-1))
    # print(s + 'models_Venus[4] =', models_Venus[4].theta.reshape(1,-1))
    # print(s + 'models_Venus[5] =', models_Venus[5].theta.reshape(1,-1))

    # print(s + 'models_Mars[0] =', models_Mars[0].theta.reshape(1,-1))
    # print(s + 'models_Mars[1] =', models_Mars[1].theta.reshape(1,-1))
    # print(s + 'models_Mars[2] =', models_Mars[2].theta.reshape(1,-1))
    # print(s + 'models_Mars[3] =', models_Mars[3].theta.reshape(1,-1))
    # print(s + 'models_Mars[4] =', models_Mars[4].theta.reshape(1,-1))
    # print(s + 'models_Mars[5] =', models_Mars[5].theta.reshape(1,-1))

    # print(s + 'models_AstroBelt[0] =', models_AstroBelt[0].theta.reshape(1, -1))
    # print(s + 'models_AstroBelt[1] =', models_AstroBelt[1].theta.reshape(1, -1))
    # print(s + 'models_AstroBelt[2] =', models_AstroBelt[2].theta.reshape(1, -1))
    # print(s + 'models_AstroBelt[3] =', models_AstroBelt[3].theta.reshape(1, -1))
    # print(s + 'models_AstroBelt[4] =', models_AstroBelt[4].theta.reshape(1, -1))
    # print(s + 'models_AstroBelt[5] =', models_AstroBelt[5].theta.reshape(1, -1))

    # Confusion matrices for each OneVsAll:
    # print('One -vs-All regularization = 0.0')
    # print(confusion_matrix_(y_cross.astype(int), oneVsAll_pred_reg00, df_option=True))
    # print('\nOne -vs-All regularization = 0.2')
    # print(confusion_matrix_(y_cross.astype(int), oneVsAll_pred_reg02, df_option=True))
    # print('\nOne -vs-All regularization = 0.4')
    # print(confusion_matrix_(y_cross.astype(int), oneVsAll_pred_reg04, df_option=True))
    # print('\nOne -vs-All regularization = 0.6')
    # print(confusion_matrix_(y_cross.astype(int), oneVsAll_pred_reg06, df_option=True))
    # print('\nOne -vs-All regularization = 0.8')
    # print(confusion_matrix_(y_cross.astype(int), oneVsAll_pred_reg08, df_option=True))
    # print('\nOne -vs-All regularization = 1.0')
    # print(confusion_matrix_(y_cross.astype(int), oneVsAll_pred_reg10, df_option=True))

    # Preparation for a pickle file:
    model_OvA_reg00 = {'Venus': models_Venus[0],
                       'Earth': models_Earth[0],
                       'Mars': models_Mars[0],
                       'AstroBelt': models_AstroBelt[0],
                       'f1_score': f1_reg00}
    model_OvA_reg02 = {'Venus': models_Venus[1],
                       'Earth': models_Earth[1],
                       'Mars': models_Mars[1],
                       'AstroBelt': models_AstroBelt[1],
                       'f1_score': f1_reg02}
    model_OvA_reg04 = {'Venus': models_Venus[2],
                       'Earth': models_Earth[2],
                       'Mars': models_Mars[2],
                       'AstroBelt': models_AstroBelt[2],
                       'f1_score': f1_reg04}
    model_OvA_reg06 = {'Venus': models_Venus[3],
                       'Earth': models_Earth[3],
                       'Mars': models_Mars[3],
                       'AstroBelt': models_AstroBelt[3],
                       'f1_score': f1_reg06}
    model_OvA_reg08 = {'Venus': models_Venus[4],
                       'Earth': models_Earth[4],
                       'Mars': models_Mars[4],
                       'AstroBelt': models_AstroBelt[4],
                       'f1_score': f1_reg08}
    model_OvA_reg10 = {'Venus': models_Venus[5],
                       'Earth': models_Earth[5],
                       'Mars': models_Mars[5],
                       'AstroBelt': models_AstroBelt[5],
                       'f1_score': f1_reg10}
    
    dcts_models = {'regularization 00': model_OvA_reg00,
                   'regularization 02': model_OvA_reg02,
                   'regularization 04': model_OvA_reg04, 
                   'regularization 06': model_OvA_reg06, 
                   'regularization 08': model_OvA_reg08, 
                   'regularization 10': model_OvA_reg10}

    # Saving the models
    with open("models.pickle", "wb") as outfile:
        pickle.dump(dcts_models, outfile)


    # Plotting of the data and the predictions
    colors = ['#0066ff', '#00cc00', '#ff8c1a', '#ac00e6']
    cmap = mpl.colors.ListedColormap(colors, name='from_list', N=None)
    fig, axes = plt.subplots(1, 3, figsize=(25, 15))
    # Fromating of scatter points for the expected and predicted
    kws_expected = {'s': 300,
                    'linewidth': 0.1,
                    'alpha': 0.5,
                    'marker': 'o',
                    'facecolor': None,
                    'edgecolor': 'face',
                    'cmap': cmap,
                    'c': y_test}

    kws_predicted = {'s': 50,
                     'marker': 'o',
                     'cmap': cmap,
                     'c': oneVsAll_pred_reg00}

    axes[0].scatter(x_test[:, 0], x_test[:, 1],
                    label='expected', **kws_expected)
    axes[1].scatter(x_test[:, 1], x_test[:, 2],
                    label='expected', **kws_expected)
    axes[2].scatter(x_test[:, 2], x_test[:, 0],
                    label='expected', **kws_expected)
    axes[0].scatter(x_test[:, 0], x_test[:, 1],
                    label='predicted', **kws_predicted)
    axes[1].scatter(x_test[:, 1], x_test[:, 2],
                    label='predicted', **kws_predicted)
    axes[2].scatter(x_test[:, 2], x_test[:, 0],
                    label='predicted', **kws_predicted)

    scalarmapable = mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin=0,
                                                               vmax=4),
                                          cmap)
    cbar = fig.colorbar(scalarmapable,
                        orientation='horizontal',
                        label='Citizenship',
                        ticks=[0.5, 1.5, 2.5, 3.5],
                        ax=axes[:],
                        aspect=60, shrink=0.6)
    cbar.ax.set_xticklabels(['Venus', 'Earth', 'Mars', 'Asteroids\nBelt'])

    axes[0].set_xlabel(x.columns[0])
    axes[0].set_ylabel(x.columns[1])
    axes[1].set_xlabel(x.columns[1])
    axes[1].set_ylabel(x.columns[2])
    axes[2].set_xlabel(x.columns[2])
    axes[2].set_ylabel(x.columns[0])

    axes[0].legend(), axes[1].legend(), axes[2].legend()
    axes[0].grid(), axes[1].grid(), axes[2].grid()
    title = 'fraction of correct predictions = '
    #fig.suptitle(title + f'{oneVsAll_pred_reg00:0.4f}', fontsize=14)
    plt.show()
