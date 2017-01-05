import numpy as np
import pandas as pd
from hyperband import run_hyperband
from sklearn.metrics import r2_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense


# Function to create model, required for KerasRegressor
def create_model(dens1=1400, dens2=300, dropout=0.2):
    model = Sequential()
    model.add(Dense(input_dim=N, output_dim=N))
    model.add(PReLU())
    model.add(Dropout(dropout))
    model.add(Dense(dens1))
    model.add(PReLU())
    model.add(Dense(dens2))
    model.add(PReLU())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['mean_squared_error'])
    return model


data = pd.read_csv("../age_prediction/FINAL_TRAIN_DATA.csv")
print(data.shape)
scaler = StandardScaler()
N = data.shape[1] - 1

X = scaler.fit_transform(data.ix[:, :-1])
y = data.ix[:, -1]
skmodel = KerasRegressor(build_fn=create_model, verbose=2)

params_opt_dic = {'dens1': [1000, 2000], 'dens2': [500, 1000]}

best_params = run_hyperband(skmodel, params_opt_dic, X, y, val_metrics=r2_score)
