import numpy as np
import pandas as pd
import pickle
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
# mpl.use('QtAgg')

path = os.path.join(os.path.dirname(__file__), '..', 'ex08')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLogR

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from scaler import MyStandardScaler
from data_spliter import data_spliter
from other_metrics import f1_score_, recall_score_, precision_score_

# ######################################################### #
#                        CONSTANTES                         #
# ######################################################### #

lst_check_feat = ["weight", "prod_distance", "time_delivery"]
lst_dataset = lst_check_feat + ["target"]

file_x = 'solar_system_census.csv'
file_y = 'solar_system_census_planets.csv'

dct1_keys = ['regularization 00',
                'regularization 02',
                'regularization 04',
                'regularization 06',
                'regularization 08',
                'regularization 10']
dct2_keys = ['Venus', 'Mars', 'Earth', 'AstroBelt', 'f1_score']
dct1_vals = {'Venus': MyLogR,
                'Mars': MyLogR,
                'Earth': MyLogR,
                'AstroBelt': MyLogR,
                'f1_score': float}

# ######################################################### #
#                  FUNCTION DEFINITIONS                     #
# ######################################################### #
def find_best_f1(data_models: dict) -> dict:
    min_f1 = 0
    dct_target = None
    for k, dct in data_models.items():
        if dct['f1_score'] > min_f1:
            min_f1 = dct['f1_score']
            dct_target = dct
    return dct_target


def retrieve_tags_f1(data_models: dict):
    lst_tags = []
    lst_f1 = []
    for k, dct_val in data_models.items():
        lst_tags.append(str(k))
        lst_f1.append(dct_val['f1_score'])
    return lst_tags, lst_f1


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


def multiclass_recall_score_(y, y_hat, labels):
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
    multiclass_recall = 0
    for l in labels:
        multiclass_recall += recall_score_(y, y_hat, l) * np.sum(y == l) / y.shape[0]
    return multiclass_recall

def multiclass_precision_score_(y, y_hat, labels):
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
    multiclass_prec = 0
    for l in labels:
        multiclass_prec += precision_score_(y, y_hat, l) * np.sum(y == l) / y.shape[0]
    return multiclass_prec

# ######################################################### #
#                             MAIN                          #
# ######################################################### #
if __name__ == "__main__":
    # Importation of the dataset + basic checking:
    try:
        df_x = pd.read_csv(file_x, index_col=0)
        df_y = pd.read_csv(file_y, index_col=0)
    except:
        s = "Issue while reading one of the dataset."
        print(s, file=sys.stderr)
        sys.exit()

    try:
        # casting the y data
        # 2 reasons: minimizing the memory space
        #            if casting fails it means y is not numeric only
        y = df_y.to_numpy(dtype=np.int8)
    except:
        s = "Something wrong when casting data to integer."
        print(s, file=sys.stderr)
        sys.exit()

    if df_x.shape[0] != y.shape[0]:
        s = f"Unmatching number of lines between {file_x} and {file_y}"
        print(s, file=sys.stderr)
        sys.exit()

    # A bit of data augmentation
    df_x['height2'] = df_x['height'] ** 2
    df_x['height3'] = df_x['height'] ** 3
    df_x['weight2'] = df_x['weight'] ** 2
    df_x['weight3'] = df_x['weight'] ** 3
    df_x['bone_density2'] = df_x['bone_density'] ** 2
    df_x['bone_density3'] = df_x['bone_density'] ** 3

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
    try:
        for key, dct_vals in data_models.items():
            if key not in dct1_keys:
                print(key)
                s = "Unknown key among the dictionnaries."
                print(s, file=sys.stderr)
                sys.exit()
            for k, v in dct_vals.items():
                if k not in dct2_keys:
                    print(key)
                    s = "Unknown key among the dictionnaries."
                    print(s, file=sys.stderr)
                    sys.exit()
                if not isinstance(v, dct1_vals[k]):
                    s = "Unknown values among the dictionnaries."
                    print(s, file=sys.stderr)
                    sys.exit()
    except:
        s = 'Unexpected issue during verification of models.pickle'
        print(s, file=sys.stderr)
        sys.exit()
    
    # ################################################################# #
    # Displaying the calculated f1 score (during benchmark) vs model tags.
    # ################################################################# #
    tags, f1_scores = retrieve_tags_f1(data_models)
    print(' ' * 6 + "Models" + ' '* 5 + '| f1')
    for tag, f1 in zip(tags, f1_scores):
        print(f"{tag}:", f1)
    _, axe = plt.subplots(1, 1, figsize=(15, 8))
    sns.barplot(x=tags, y=f1_scores, ax=axe)
    axe.set_xlabel("models")
    axe.set_ylabel("F1 score")
    plt.title("F1 vs explored One-vs-All")
    plt.xticks(rotation=90)
    axe.grid()
    plt.subplots_adjust(bottom=0.22)
    plt.show()

    # ############################################################## #
    # Getting the best model and training a new one from scratch
    # Plus ...
    # ############################################################## #
    dct_target = find_best_f1(data_models)

    m = dct_target['Earth'].theta.shape
    model_Earth = MyLogR(np.random.rand(*m))
    model_Mars = MyLogR(np.random.rand(*m))
    model_Venus = MyLogR(np.random.rand(*m))
    model_AstroBelt = MyLogR(np.random.rand(*m))
    
    model_Earth.set_params_(dct_target['Earth'].get_params_())
    model_Venus.set_params_(dct_target['Venus'].get_params_())
    model_Mars.set_params_(dct_target['Mars'].get_params_())
    model_AstroBelt.set_params_(dct_target['AstroBelt'].get_params_())
    
    model_Earth.set_params_({'theta': np.random.rand(*m)} )
    model_Venus.set_params_({'theta': np.random.rand(*m)})
    model_Mars.set_params_({'theta': np.random.rand(*m)})
    model_AstroBelt.set_params_({'theta': np.random.rand(*m)})

    # ##################################################### #
    # Preparing the data: generating the polynomial features
    # standardization of train and test sets
    # ##################################################### #
    x = df_x.values
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    print("\nInformation about spliting:")
    print(f" x_train shape: {x_train.shape}\n x_test shape: {x_test.shape}\n")
    
    scaler_x = MyStandardScaler()
    scaler_x.fit(x_train)
    x_train_tr = scaler_x.transform(x_train)
    x_test_tr = scaler_x.transform(x_test)
    x_tr = scaler_x.transform(x)
    # ##################################################### #
    # Training of the best model (with the 'best' regularization factor)
    # and prediction and dsiplay of metrics from the retrieved models
    # ##################################################### #
    for ii, model in enumerate((model_Venus, model_Earth, model_Mars, model_AstroBelt)):
        # Training of the models
        print(f"Training the model {model._tag_}")
        model.fit_(x_train_tr, labelbinarizer(y_train, ii))
        # Prediction and binarization of the probabilities for all the models
        if ii == 0:
            preds = model.predict_(x_test_tr)
        else:
            preds = np.hstack((preds.copy(), model.predict_(x_test_tr)))

    # Calculating the one vs all prediction
    preds_OvA_best = np.argmax(preds, axis=1).reshape(-1, 1)

    # ##################################################### #
    # Printing the precision, recall and f1 of each One-vs-All
    # models
    # ##################################################### #
    print(' ' * 24 + 'prec  |recall | f1' )
    for key, OvA in data_models.items():
        p_Venus = OvA['Venus'].predict_(x_test_tr)
        p_Earth = OvA['Earth'].predict_(x_test_tr)
        p_Mars = OvA['Mars'].predict_(x_test_tr)
        p_AstroBelt = OvA['AstroBelt'].predict_(x_test_tr)
        preds = np.hstack((p_Venus, p_Earth, p_Mars, p_AstroBelt))
        preds_ = np.argmax(preds, axis=1).reshape(-1, 1).copy()
        f1 = multiclass_f1_score_(y_test, preds_, labels=[0, 1, 2, 3])
        prec = multiclass_precision_score_(y_test, preds_, labels=[0, 1, 2, 3])
        recall = multiclass_recall_score_(y_test, preds_, labels=[0, 1, 2, 3])
        print(f"[{key:20s}]: {prec:.3f} | {recall:.3f} | {f1:.3f}")


    f1 = multiclass_f1_score_(y_test, preds_OvA_best, labels=[0, 1, 2, 3])
    prec = multiclass_precision_score_(y_test, preds_OvA_best, labels=[0, 1, 2, 3])
    recall = multiclass_recall_score_(y_test, preds_OvA_best, labels=[0, 1, 2, 3])
    print(f"[{'selected':20s}]: {prec:.3f} | {recall:.3f} | {f1:.3f}")

    # ##################################################### #
    # Plotting the resuts (base model from the exploration
    # + the retrained model)
    # ##################################################### #
    
    for ii, model in enumerate((model_Venus, model_Earth, model_Mars, model_AstroBelt)):
        # Prediction and binarization of the probabilities for all the models
        if ii == 0:
            preds = model.predict_(x_tr)
        else:
            preds = np.hstack((preds.copy(), model.predict_(x_tr)))
    preds_OvA_best = np.argmax(preds, axis=1).reshape(-1, 1)

    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')
    colors = ["purple", "royalblue", "red", "gold"]
    height = df_x['height'].values.reshape(-1,1)
    weight = df_x['weight'].values.reshape(-1,1)
    bone_density = df_x['bone_density'].values.reshape(-1,1)
    leg = {0:'Venus', 1:'Earth', 2:'Mars', 3:'Asteroid'}
    for label in range(4):
        index_y = np.where(np.all(y == label, axis = 1))
    
        ax.scatter(height[index_y],
                   weight[index_y],
                   bone_density[index_y],
                   facecolors='none',
                   edgecolors=colors[label],
                   s=90, label = f"truth origin {leg[label]}")
        index_hat = np.where(np.all(preds_OvA_best == label, axis = 1))
        ax.scatter(height[index_hat],
                   weight[index_hat],
                   bone_density[index_hat],
                   color=colors[label],
                   s=80, label = f"predicted origin {leg[label]}")
    
    ax.set_xlabel('height')
    ax.set_ylabel('weight')
    ax.set_zlabel('bone_density')
    ax.set_title("Best Model Predictions Vs Truth: "
                 + f"f1={f1:.3f} - P={prec:.3f} - R={recall:.3f}")
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.show()
