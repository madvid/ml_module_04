import numpy as np
import pandas as pd
import pickle
import sys
import os
import matplotlib.pyplot as plt
from benchmark_train import data_idx
# mpl.use('QtAgg')


path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from scaler import MyStandardScaler
from data_spliter import data_spliter
from ridge_v2 import MyRidge

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


def expand_poly_term(data: pd.DataFrame):
    data['w2'] = data['w'] ** 2
    data['w3'] = data['w'] ** 3
    data['w4'] = data['w'] ** 4
    data['t2'] = data['t'] ** 2
    data['t3'] = data['t'] ** 3
    data['t4'] = data['t'] ** 4
    data['p2'] = data['p'] ** 2
    data['p3'] = data['p'] ** 3
    data['p4'] = data['p'] ** 4


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
                'lambda_',
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
    fig, axes = plt.subplots(2, 1, figsize=(25, 14))
    
    half = len(tags) // 2
    axes[0].scatter(tags[:half], losses[:half])
    axes[1].scatter(tags[half:], losses[half:])
    fig.tight_layout(pad=14.0)
    axes[0].set_xlabel("models")
    axes[0].set_ylabel("MSE")
    axes[1].set_xlabel("models")
    axes[1].set_ylabel("MSE")
    axes[0].set_ylim([0, 0.9])
    axes[1].set_ylim([0, 0.5])
    axes[0].grid()
    axes[1].grid()

    plt.title("MSE vs explored models")
    plt.setp(axes[0].get_xticklabels(), rotation=90)
    plt.setp(axes[1].get_xticklabels(), rotation=90)
    #plt.subplots_adjust(bottom=0.22)
    plt.show()   

    # ##################################################### #
    # Getting the best model and training a new one identical
    # but with the CV set include in training set
    # ##################################################### #
    dct_target = find_lowest_loss(data_models)

    model_best_reg00 = MyRidge(dct_target['thetas'])
    model_best_reg02 = MyRidge(dct_target['thetas'])
    model_best_reg04 = MyRidge(dct_target['thetas'])
    model_best_reg06 = MyRidge(dct_target['thetas'])
    model_best_reg08 = MyRidge(dct_target['thetas'])
    model_best_reg10 = MyRidge(dct_target['thetas'])
    
    model_best_regs = [model_best_reg00,
                       model_best_reg02,
                       model_best_reg04,
                       model_best_reg06,
                       model_best_reg08,
                       model_best_reg10]

    for ii, model_b in enumerate(model_best_regs):
        model_b.set_params_(dct_target)
        model_b.set_params_({"lambda_": ii * 0.2})
    
    # model_base._tag_ = dct_target['_tag_']
    # model_base._vars_ = dct_target['_vars_']
    print(f"{model_best_reg00._vars_ = }",
          f"-- {model_best_reg00._tag_ = }",
          f"-- {model_best_reg00.alpha = }",
          f"-- {model_best_reg00.lambda_ = }")

    # ##################################################### #
    # Preparing the data: generating the polynomial features
    # standardization of train and test sets
    # ##################################################### #
    expand_poly_term(data)
    full_model = MyRidge(np.random.rand(data.shape[1], 1), alpha=1e-2, max_iter=100000)
    
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

    preds_best_regs = {}
    for model in model_best_regs:
        model.fit_(x_train_tr[:, data_idx(dct_target['_vars_'])], y_train_tr)
        preds_best_regs[f'reg{model.lambda_:.2f}'] = model.predict_(x_test_tr[:, data_idx(dct_target['_vars_'])])
    
    full_model.fit_(x_train_tr, y_train_tr)
    #pred_full_trained = full_model.predict_(Xtr)
    preds_best_regs['full'] = full_model.predict_(x_test_tr)
    # ##################################################### #
    # Plotting the resuts (base model from the exploration
    # + the retrained model)
    # ##################################################### #
    mse_best_regs = {}
    for model in model_best_regs:
        y_hat = preds_best_regs[f'reg{model.lambda_:.2f}']
        mse_best_regs[f'reg{model.lambda_:.2f}'] = model.loss_(y_test_tr, y_hat)
    
    mse_best_regs['full'] = full_model._loss_( y_test_tr, preds_best_regs['full'])

    print(' ' * 15 + '| MSE')
    for k, val in mse_best_regs.items():
        print(f"[{k}]".ljust(15) + '| ' + f"{val:4f}")

    fig, axes = plt.subplots(1, 3, figsize=(25, 8))
    # 'weight' aka w is located at col 0
    # 'prod_distance' aka p is located at col 1
    # 'time_deivery' aka t is located at col 2
    xlabel_axes = ["standardized weight", "standardized prod_distance", "standardized time_delivery"]
    for ax_i in range(3):
        axes[ax_i].scatter(x_test_tr[:, ax_i], y_test_tr, label='ground true', s=32)
        size = 26
        for k, pred in preds_best_regs.items():
            axes[ax_i].scatter(x_test_tr[:, ax_i], pred, label=k, s=size)
            size -= 3 
        
        axes[ax_i].grid()
        axes[ax_i].set_xlabel(xlabel_axes[ax_i])
    axes[0].legend()
    axes[0].set_ylabel("standardized target")

    # axes[0].scatter(Xtr[:, 0], Ytr, label='ground true', s=4)
    # axes[0].scatter(Xtr[:, 0], pred_trained, label='trained', s=2)
    # axes[0].scatter(Xtr[:, 0], pred_full_trained, label='full model', s=2)
    # axes[0].grid()
    # axes[0].set_xlabel("standardized weight")
    # axes[0].set_ylabel("standardized target")
# 
    # axes[1].scatter(Xtr[:, 1], Ytr, label='ground true', s=4)
    # axes[1].scatter(Xtr[:, 1], pred_trained, label='trained', s=2)
    # axes[1].scatter(Xtr[:, 1], pred_full_trained, label='full model', s=2)
    # axes[1].grid()
    # axes[1].legend()
    # axes[1].set_xlabel("standardized prod_distance")
# 
    # axes[2].scatter(Xtr[:, 2], Ytr, label='ground true', s=4)
    # axes[2].scatter(Xtr[:, 2], pred_trained, label='trained', s=2)
    # axes[2].scatter(Xtr[:, 2], pred_full_trained, label='full model', s=2)
    # axes[2].grid()
    # axes[2].set_xlabel("standardized time_delivery")

    #title = f"MSE score model base: {mse_base:.5f} -- MSE score model full: {mse_full:.5f}"
    #axes[1].set_title(title)
    plt.show()
