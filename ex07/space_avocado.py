import numpy as np
import pandas as pd
import pickle
import sys
import os
import matplotlib.pyplot as plt
# mpl.use('QtAgg')

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from scaler import MyStandardScaler

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

# ######################################################### #
#                        CONSTANTES                         #
# ######################################################### #

lst_check_feat = ["weight", "prod_distance", "time_delivery"]
lst_dataset = lst_check_feat + ["target"]

# ######################################################### #
#                  FUNCTION DEFINITIONS                     #
# ######################################################### #


def find_lowest_loss(data_models: dict) -> dict:
    min_loss = np.inf
    dct_target = None
    for _, dct in data_models.items():
        if dct['_loss_'] < min_loss:
            min_loss = dct['_loss_']
            dct_target = dct
    return dct_target


def retrieve_tags_losses(data_models: dict):
    lst_tags = []
    lst_losses = []
    for _, dct in data_models.items():
        lst_tags.append(dct['_tag_'])
        lst_losses.append(dct['_loss_'])
    return lst_tags, lst_losses


def expand_poly_cross_term(data: pd.DataFrame):
    data['w2'] = data['w'] ** 2
    data['w3'] = data['w'] ** 3
    data['w4'] = data['w'] ** 4
    data['t2'] = data['t'] ** 2
    data['t3'] = data['t'] ** 3
    data['t4'] = data['t'] ** 4
    data['p2'] = data['p'] ** 2
    data['p3'] = data['p'] ** 3


# ######################################################### #
#                             MAIN                          #
# ######################################################### #
if __name__ == "__main__":
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

    # Renaming columns for convenience
    data.rename(columns={'weight': 'w',
                         'prod_distance': 'p',
                         'time_delivery': 't'},
                inplace=True)

    # Retrieving the models' data from the pickle file
    data_models = None
    try:
        with open("models.pickle", 'rb') as file:
            data_models = pickle.load(file)
    except FileNotFoundError:
        print("Pickle file was not found, check the name of the file.",
              file=sys.stderr)
        sys.exit()
    except:
        print("Something wrong happens.")
        sys.exit()

    # Testing the keys wihin the models.
    lst_keys = ['thetas',
                'alpha',
                'max_iter',
                '_tag_',
                '_idx_',
                '_vars_',
                '_loss_']
    for _, dct in data_models.items():
        if any([key not in lst_keys for key in dct.keys()]):
            print("Extra unknown key among the dictionnaries.")
            print(dct.keys())
            sys.exit()
        if any([key not in dct.keys() for key in lst_keys]):
            print("Missing expected key among the dictionnaries.")
            sys.exit()

    # ##################################################### #
    # Plotting losses vs model tags.
    # ##################################################### #
    tags, losses = retrieve_tags_losses(data_models)
    _, axe = plt.subplots(1, 1, figsize=(15, 8))
    plt.scatter(tags, losses)
    axe.set_xlabel("models")
    axe.set_ylabel("MSE")
    plt.title("MSE vs explored models")
    plt.xticks(rotation=90)
    plt.grid()
    plt.subplots_adjust(bottom=0.22)
    plt.show()

    # ##################################################### #
    # Getting the best model and training a new one identical
    # but with the CV set include in training set
    # ##################################################### #
    dct_target = find_lowest_loss(data_models)

    model_base = MyLR(dct_target['thetas'])
    model_base._tag_ = dct_target['_tag_']
    model_base._vars_ = dct_target['_vars_']
    print(f" vars = {model_base._vars_}-- tag = {model_base._tag_}")

    m = model_base.thetas.shape
    model = MyLR(np.random.rand(*m), alpha=1e-2, max_iter=1000000)

    # ##################################################### #
    # Preparing the data: generating the polynomial features
    # standardization of train and test sets
    # ##################################################### #
    expand_poly_cross_term(data)
    cols = data.columns.values
    cols = cols[cols != 'target']
    Xs = data_spliter(data[cols].values,
                      data["target"].values.reshape(-1, 1), 0.8)
    x_train, x_test, y_train, y_test = Xs

    scaler_x = MyStandardScaler()
    scaler_y = MyStandardScaler()
    scaler_x.fit(x_train)
    scaler_y.fit(y_train)
    x_train_tr = scaler_x.transform(x_train)
    y_train_tr = scaler_y.transform(y_train)
    x_test_tr = scaler_x.transform(x_test)
    y_test_tr = scaler_y.transform(y_test)

    Xtr = scaler_x.transform(data[cols].values)
    Ytr = scaler_y.transform(data['target'].values)

    model.fit_(x_train_tr, y_train_tr)
    pred_trained = model.predict_(Xtr)

    # ##################################################### #
    # Plotting the resuts (base model from the exploration
    # + the retrained model)
    # ##################################################### #
    mse_trained = model._loss_(model.predict_(x_test_tr), y_test_tr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    # 'weight' aka w is located at col 0
    # 'prod_distance' aka p is located at col 1
    # 'time_deivery' aka t is located at col 2
    axes[0].scatter(Xtr[:, 0], Ytr, label='ground true', s=4)
    axes[0].scatter(Xtr[:, 0], pred_trained, label='trained', s=2)
    axes[0].grid()
    axes[0].set_xlabel("standardized weight")
    axes[0].set_ylabel("standardized target")

    axes[1].scatter(Xtr[:, 1], Ytr, label='ground true', s=4)
    axes[1].scatter(Xtr[:, 1], pred_trained, label='trained', s=2)
    axes[1].grid()
    axes[1].legend()
    axes[1].set_xlabel("standardized prod_distance")

    axes[2].scatter(Xtr[:, 2], Ytr, label='ground true', s=4)
    axes[2].scatter(Xtr[:, 2], pred_trained, label='trained', s=2)
    axes[2].grid()
    axes[2].set_xlabel("standardized time_delivery")

    title = f"MSE score: {mse_trained:.5f}"
    axes[1].set_title(title)
    plt.show()
