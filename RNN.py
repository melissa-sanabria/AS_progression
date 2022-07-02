import pickle
import numpy as np
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import MinMaxScaler

## Read data

features = pickle.load(open("features-22.pkl", "rb"))
labels = pickle.load(open("../labels.pkl", "rb"))
patients = pickle.load(open("../patients.pkl", "rb"))
feature_names = pickle.load(open("../feature_names.pkl", "rb"))

indices_file = "../split_indices.csv"
indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]

n_max_visits = features.shape[1]

### RNN
rnn_output_path = "RNN/"
if not os.path.exists(rnn_output_path):
    os.makedirs(rnn_output_path)

rnn_weights_path = rnn_output_path + "weights/"
if not os.path.exists(rnn_weights_path):
    os.makedirs(rnn_weights_path)

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

print("RNN")
for i in range(10):
    these_train_indices = np.where(indices[:, i] == 1)[0]
    these_test_indices = np.where(indices[:, i] == 2)[0]
    these_val_indices = np.where(indices[:, i] == 3)[0]

    X_train = features[these_train_indices]
    y_train = np.expand_dims(labels[these_train_indices], axis=-1)
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_train = np.nan_to_num(X_train, nan=-1)

    X_test = features[these_test_indices]
    y_test = np.expand_dims(labels[these_test_indices], axis=-1)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_test = np.nan_to_num(X_test, nan=-1)

    X_val = features[these_val_indices]
    y_val = np.expand_dims(labels[these_val_indices], axis=-1)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_val = np.nan_to_num(X_val, nan=-1)

    es = EarlyStopping(monitor='val_loss', verbose=0, patience=150, restore_best_weights=True)

    input = Input(shape=(None, X_train.shape[-1]))
    lstm = Masking(mask_value=-1)(input)
    lstm = GRU(16, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(lstm)
    lstm = Dense(1, activation="sigmoid")(lstm)

    model = Model(input, lstm)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, callbacks=[es], verbose=0)
    model.save(rnn_weights_path + str(i) + ".h5")

    predictions_raw = model.predict(X_test)[:, :, 0]
    predictions = np.round(predictions_raw)
    y_test = y_test[:, :, 0]
    for p_idx in range(len(predictions)):
        for v in range(n_max_visits):
            if X_test[p_idx, v].max() != -1:
                predictions_values_per_visit[v][i].append(predictions[p_idx, v])
                predictions_raw_values_per_visit[v][i].append(predictions_raw[p_idx, v])
                real_values_per_visit[v][i].append(y_test[p_idx, v])
            else:
                predictions_values_per_visit[-1][i].append(predictions[p_idx, v - 1])
                predictions_raw_values_per_visit[-1][i].append(predictions_raw[p_idx, v - 1])
                real_values_per_visit[-1][i].append(y_test[p_idx, v - 1])
                break
            if v == (n_max_visits - 1):
                predictions_values_per_visit[- 1][i].append(predictions[p_idx, v])
                predictions_raw_values_per_visit[- 1][i].append(predictions_raw[p_idx, v])
                real_values_per_visit[- 1][i].append(y_test[p_idx, v])


pickle.dump(predictions_values_per_visit, open(rnn_output_path + "predictions.pkl", "wb"))
pickle.dump(predictions_raw_values_per_visit, open(rnn_output_path + "predictions_raw.pkl", "wb"))
pickle.dump(real_values_per_visit, open(rnn_output_path + "real_values.pkl", "wb"))

