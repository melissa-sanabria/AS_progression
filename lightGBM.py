import pickle
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import utils
import lightgbm as lgb

## Read data
features = pickle.load(open("features-22.pkl", "rb"))
labels = pickle.load(open("../labels.pkl", "rb"))
patients = pickle.load(open("../patients.pkl", "rb"))
feature_names = pickle.load(open("../feature_names.pkl", "rb"))


indices_file = "../split_indices.csv"
indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]

n_max_visits = features.shape[1]

# lgb_1v_output_path = "lightGBM-1v_allvisits/"
lgb_1v_output_path = "lightGBM-1v/"
if not os.path.exists(lgb_1v_output_path):
    os.makedirs(lgb_1v_output_path)

predictions_values_per_visit = {}
for v in range(n_max_visits):
    predictions_values_per_visit[v] = [[] for i in range(10)]
predictions_values_per_visit[-1] = [[] for i in range(10)]

real_values_per_visit = {}
for v in range(n_max_visits):
    real_values_per_visit[v] = [[] for i in range(10)]
real_values_per_visit[-1] = [[] for i in range(10)]

predictions_raw_values_per_visit = {}
for v in range(n_max_visits):
    predictions_raw_values_per_visit[v] = [[] for i in range(10)]
predictions_raw_values_per_visit[-1] = [[] for i in range(10)]

scaler = MinMaxScaler(feature_range=(-1, 1))

for i in range(10):
    these_train_indices = np.where(indices[:, i] == 1)[0]
    these_test_indices = np.where(indices[:, i] == 2)[0]
    these_val_indices = np.where(indices[:, i] == 3)[0]

    X_train = features[these_train_indices]
    y_train = labels[these_train_indices]
    X_train, y_train, patient_visit_train = utils.get_data(X_train, y_train, these_train_indices)

    X_train = scaler.fit_transform(X_train)
    X_train = np.nan_to_num(X_train, nan=-1)

    X_test = features[these_test_indices]
    y_test = labels[these_test_indices]
    X_test, y_test, patient_visit_test = utils.get_data(X_test, y_test, these_test_indices)

    X_test = scaler.transform(X_test)
    X_test = np.nan_to_num(X_test, nan=-1)

    X_val = features[these_val_indices]
    y_val = labels[these_val_indices]
    X_val, y_val, patient_visit_val = utils.get_data(X_val, y_val, these_val_indices)

    X_val = scaler.transform(X_val)
    X_val = np.nan_to_num(X_val, nan=-1)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    val_data = lgb.Dataset(X_val, label=y_val)

    clf = lgb.LGBMClassifier()
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'binary_error'
    param['verbosity'] = -1
    bst = lgb.train(param, train_data, 50, valid_sets=val_data, early_stopping_rounds=50, verbose_eval=False)
    predictions_raw = bst.predict(X_test)
    predictions = np.round(predictions_raw)

    for v in range(n_max_visits):
        v_ids = np.where(patient_visit_test[:, 1] == v)[0]
        if len(v_ids) > 0:
            predictions_values_per_visit[v][i] += predictions[v_ids].tolist()
            predictions_raw_values_per_visit[v][i] += predictions_raw[v_ids].tolist()
            real_values_per_visit[v][i] += y_test[v_ids].tolist()

            if v == 0:
                for patients_idx in np.unique(patient_visit_test[:, 0]):
                    this_patient_samples = np.where(patient_visit_test[:, 0] == patients_idx)[0]
                    predictions_values_per_visit[-1][i].append(predictions[this_patient_samples[-1]])
                    predictions_raw_values_per_visit[-1][i].append(predictions_raw[this_patient_samples[-1]])
                    real_values_per_visit[-1][i].append(y_test[this_patient_samples[-1]])

pickle.dump(predictions_values_per_visit, open(lgb_1v_output_path + "predictions.pkl", "wb"))
pickle.dump(predictions_raw_values_per_visit, open(lgb_1v_output_path + "predictions_raw.pkl", "wb"))
pickle.dump(real_values_per_visit, open(lgb_1v_output_path + "real_values.pkl", "wb"))

## lightGBM 2v
print("lightGBM 2v")
lgb_2v_output_path = "lightGBM-2v/"
if not os.path.exists(lgb_2v_output_path):
    os.makedirs(lgb_2v_output_path)

predictions_values_per_visit = {}
for v in range(n_max_visits):
    predictions_values_per_visit[v] = [[] for i in range(10)]
predictions_values_per_visit[-1] = [[] for i in range(10)]

real_values_per_visit = {}
for v in range(n_max_visits):
    real_values_per_visit[v] = [[] for i in range(10)]
real_values_per_visit[-1] = [[] for i in range(10)]

predictions_raw_values_per_visit = {}
for v in range(n_max_visits):
    predictions_raw_values_per_visit[v] = [[] for i in range(10)]
predictions_raw_values_per_visit[-1] = [[] for i in range(10)]

for i in range(10):
    these_train_indices = np.where(indices[:, i] == 1)[0]
    these_test_indices = np.where(indices[:, i] == 2)[0]
    these_val_indices = np.where(indices[:, i] == 3)[0]

    X_train = features[these_train_indices]
    y_train = labels[these_train_indices]
    X_train, y_train, patient_visit_train = utils.get_data_2v(X_train, y_train, these_train_indices)

    X_train = scaler.fit_transform(X_train)
    X_train = np.nan_to_num(X_train, nan=-1)

    X_test = features[these_test_indices]
    y_test = labels[these_test_indices]
    X_test, y_test, patient_visit_test = utils.get_data_2v(X_test, y_test, these_test_indices)

    X_test = scaler.transform(X_test)
    X_test = np.nan_to_num(X_test, nan=-1)

    X_val = features[these_val_indices]
    y_val = labels[these_val_indices]
    X_val, y_val, patient_visit_val = utils.get_data_2v(X_val, y_val, these_val_indices)

    X_val = scaler.transform(X_val)
    X_val = np.nan_to_num(X_val, nan=-1)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    val_data = lgb.Dataset(X_val, label=y_val)

    clf = lgb.LGBMClassifier()
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'binary_error'
    param['verbosity'] = -1
    bst = lgb.train(param, train_data, 50, valid_sets=val_data, early_stopping_rounds=50, verbose_eval=False)

    predictions_raw = bst.predict(X_test)
    predictions = np.round(predictions_raw)

    for v in range(1, n_max_visits):
        v_ids = np.where(patient_visit_test[:, 1] == v)[0]
        if len(v_ids) > 0:
            predictions_values_per_visit[v][i] += predictions[v_ids].tolist()
            predictions_raw_values_per_visit[v][i] += predictions_raw[v_ids].tolist()
            real_values_per_visit[v][i] += y_test[v_ids].tolist()

            if v == 1:
                for patients_idx in np.unique(patient_visit_test[:, 0]):
                    this_patient_samples = np.where(patient_visit_test[:, 0] == patients_idx)[0]
                    predictions_values_per_visit[-1][i].append(predictions[this_patient_samples[-1]])
                    predictions_raw_values_per_visit[-1][i].append(predictions_raw[this_patient_samples[-1]])
                    real_values_per_visit[-1][i].append(y_test[this_patient_samples[-1]])

pickle.dump(predictions_values_per_visit, open(lgb_2v_output_path + "predictions.pkl", "wb"))
pickle.dump(predictions_raw_values_per_visit, open(lgb_2v_output_path + "predictions_raw.pkl", "wb"))
pickle.dump(real_values_per_visit, open(lgb_2v_output_path + "real_values.pkl", "wb"))
