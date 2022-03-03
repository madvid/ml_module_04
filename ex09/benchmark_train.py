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
    monolr_Venus = [MyLogR(np.random.rand(x.shape[1] + 1, 1)) for l_ii in range(6)]
    monolr_Earth = [MyLogR(np.random.rand(x.shape[1] + 1, 1)) for l_ii in range(6)]
    monolr_Mars = [MyLogR(np.random.rand(x.shape[1] + 1, 1)) for l_ii in range(6)]
    monolr_AstroBelt = [MyLogR(np.random.rand(x.shape[1] + 1, 1)) for l_ii in range(6)]

    monolr_Venus.fit_(x_train_tr, labelbinarizer(y_train, 0))
    monolr_Earth.fit_(x_train_tr, labelbinarizer(y_train, 1))
    monolr_Mars.fit_(x_train_tr, labelbinarizer(y_train, 2))
    monolr_AstroBelt.fit_(x_train_tr, labelbinarizer(y_train, 3))

    # Prediction and binarization of the probabilities
    pred_Venus = monolr_Venus.predict_(x_test_tr)
    pred_Earth = monolr_Earth.predict_(x_test_tr)
    pred_Mars = monolr_Mars.predict_(x_test_tr)
    pred_AstroBelt = monolr_AstroBelt.predict_(x_test_tr)

    # stacking and calculating the one vs all prediction
    preds = np.hstack((pred_Venus, pred_Earth, pred_Mars, pred_AstroBelt))
    oneVsAll_pred = np.argmax(preds, axis=1).reshape(-1, 1)

    # Calcul of the fraction of correct prediction
    correct_pred = np.sum(oneVsAll_pred == y_test) / y_test.shape[0]
    s = "Fraction of the corrected prediction (accuracy): "
    print(s + f'{correct_pred:.4f}')

    # Plotting of the data and the predictions
    colors = ['#0066ff', '#00cc00', '#ff8c1a', '#ac00e6']
    cmap = mpl.colors.ListedColormap(colors, name='from_list', N=None)
    fig, axes = plt.subplots(1, 3, figsize=(13, 10))
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
                     'c': oneVsAll_pred}

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
    fig.suptitle(title + f'{correct_pred:0.4f}', fontsize=14)
    plt.show()
