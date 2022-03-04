import sys
import os
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
from scaler import MyStandardScaler
from logistic_related import *
from other_metrics import f1_score_

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
    x_train, x_test, y_train, y_test = data_spliter(x.values, y, 0.8)

    # Preprocessing (simple standardistation of the features)
    scaler_x = MyStandardScaler()
    scaler_x.fit(x_train)
    x_train_tr = scaler_x.transform(x_train)
    x_test_tr = scaler_x.transform(x_test)

    # Instanciation and training of the models
    models_Venus = [MyLogR(np.random.rand(x.shape[1] + 1, 1), lambda_=l_ii * 0.2) for l_ii in range(6)]
    models_Earth = [MyLogR(np.random.rand(x.shape[1] + 1, 1), lambda_=l_ii * 0.2) for l_ii in range(6)]
    models_Mars = [MyLogR(np.random.rand(x.shape[1] + 1, 1), lambda_=l_ii * 0.2) for l_ii in range(6)]
    models_AstroBelt = [MyLogR(np.random.rand(x.shape[1] + 1, 1), lambda_=l_ii * 0.2) for l_ii in range(6)]

    # Preparation of the lists for the prediction of the models afeter training
    preds_Venus = []
    preds_Earth = []
    preds_Mars = []
    preds_AstroBelt = []
    dct_params = {'alpha': 1e-2, 'max_iter': 10000}
    for ii in range(6):
        # Setting learning rate and number of iterations for each models
        models_Venus[ii].set_params_(dct_params)
        models_Earth[ii].set_params_(dct_params)
        models_Mars[ii].set_params_(dct_params)
        models_AstroBelt[ii].set_params_(dct_params)
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
        preds_Venus.append(models_Venus[ii].predict_(x_test_tr))
        preds_Earth.append(models_Earth[ii].predict_(x_test_tr))
        preds_Mars.append(models_Mars[ii].predict_(x_test_tr))
        preds_AstroBelt.append(models_AstroBelt[ii].predict_(x_test_tr))

    # stacking and calculating the one vs all prediction
    preds_reg00 = np.hstack((preds_Venus[0], preds_Earth[0], preds_Mars[0], preds_AstroBelt[0]))
    preds_reg02 = np.hstack((preds_Venus[1], preds_Earth[1], preds_Mars[1], preds_AstroBelt[1]))
    preds_reg04 = np.hstack((preds_Venus[2], preds_Earth[2], preds_Mars[2], preds_AstroBelt[2]))
    preds_reg06 = np.hstack((preds_Venus[3], preds_Earth[3], preds_Mars[3], preds_AstroBelt[3]))
    preds_reg08 = np.hstack((preds_Venus[4], preds_Earth[4], preds_Mars[4], preds_AstroBelt[4]))
    preds_reg10 = np.hstack((preds_Venus[5], preds_Earth[5], preds_Mars[5], preds_AstroBelt[5]))
    oneVsAll_pred_reg00 = np.argmax(preds_reg00, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg02 = np.argmax(preds_reg02, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg04 = np.argmax(preds_reg04, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg06 = np.argmax(preds_reg06, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg08 = np.argmax(preds_reg08, axis=1).reshape(-1, 1)
    oneVsAll_pred_reg10 = np.argmax(preds_reg10, axis=1).reshape(-1, 1)

    # Calcul of the fraction of correct prediction
    print(y_test.astype(int).dtype)
    print(oneVsAll_pred_reg00.dtype)
    f1_reg00 = f1_score_(y_test.astype(int), oneVsAll_pred_reg00.astype(int))
    f1_reg02 = f1_score_(y_test.astype(int), oneVsAll_pred_reg02.astype(int))
    f1_reg04 = f1_score_(y_test.astype(int), oneVsAll_pred_reg04.astype(int))
    f1_reg06 = f1_score_(y_test.astype(int), oneVsAll_pred_reg06.astype(int))
    f1_reg08 = f1_score_(y_test.astype(int), oneVsAll_pred_reg08.astype(int))
    f1_reg10 = f1_score_(y_test.astype(int), oneVsAll_pred_reg10.astype(int))
    s = "f1-score of "
    print(s + 'regularized models with lambda_=0.0: ' + f'{f1_reg00:.4f}')
    print(s + 'regularized models with lambda_=0.2: ' + f'{f1_reg02:.4f}')
    print(s + 'regularized models with lambda_=0.4: ' + f'{f1_reg04:.4f}')
    print(s + 'regularized models with lambda_=0.6: ' + f'{f1_reg06:.4f}')
    print(s + 'regularized models with lambda_=0.8: ' + f'{f1_reg08:.4f}')
    print(s + 'regularized models with lambda_=1.0: ' + f'{f1_reg10:.4f}')

    # Plotting of the data and the predictions
    colors = ['#0066ff', '#00cc00', '#ff8c1a', '#ac00e6']
    cmap = mpl.colors.ListedColormap(colors, name='from_list', N=None)
    fig, axes = plt.subplots(1, 3, figsize=(25, 15))
    # Fromating of scatter points for the expected and predicted
    kws_expected = {'s': 300,
                    'linewidth': 0.2,
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
    fig.suptitle(title + f'{correct_pred_reg00:0.4f}', fontsize=14)
    plt.show()
