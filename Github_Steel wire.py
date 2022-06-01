import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Layer, Input, Concatenate
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K 

panel = pd.read_csv("C:/.../Steel wire.csv") # Please see the data
S = panel['Stress']*0.00689476
N = panel['N']
lgN = np.log10(N)
censor_indx = panel['Censor']


train_data = S
train_target = lgN
test_data = np.arange(1*min(S.unique()), 1.05*max(S.unique()), 5)
S_pred = test_data
test_data_min = test_data.min(axis=0)
test_data_max = test_data.max(axis=0)
test_data_range = test_data_max - test_data_min
train_data = (train_data - test_data_min) / test_data_range
test_data = (test_data - test_data_min) / test_data_range


def mdn_cost(mu, sigma, y, indx):
    dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y)*indx-dist.log_survival_function(y)*(tf.constant([[1.0]])-indx))
  

class NonPos(Constraint):
    """Constrains the weights to be non-positive.
    """
    def __call__(self, w):
        return w * K.cast(K.less_equal(w, 0.), K.floatx()) 

epochs = 500
batch_size = 1
learning_rate = 0.001
InputLayer = Input(shape=(1,))
Layer_1_1 = Dense(5,activation="tanh", kernel_constraint=NonPos(), bias_constraint=NonPos())(InputLayer)
Layer_1_2 = Dense(5,activation="tanh", kernel_constraint=NonPos())(InputLayer)
mu0 = Dense(1, activation="linear", kernel_constraint=keras.constraints.NonNeg())(Layer_1_1)
sigma = Dense(1, activation=lambda x: tf.nn.elu(x) + 1, kernel_constraint=keras.constraints.NonNeg())(Layer_1_2)
merged = Concatenate(axis=1)([mu0,sigma])
mu = Dense(1, activation="linear",kernel_constraint=keras.constraints.NonNeg())(merged)
y_real = Input(shape=(1,))
indx = Input(shape=(1,))
lossF = mdn_cost(mu,sigma,y_real,indx)
model = Model(inputs=[InputLayer, y_real, indx], outputs=[mu, sigma])
model.add_loss(lossF)
adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=adamOptimizer,metrics=['mse'])     

history_cache = model.fit([train_data, train_target, censor_indx], #notice we are using an input to pass the real values due to the inner workings of keras
                          verbose=1, # write =1 if you wish to see the progress for each epoch
                          epochs=epochs,
                          batch_size=batch_size)
print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))
mu_pred, sigma_pred = model.predict(list((test_data, test_data, test_data))) # the model expects a list of arrays as it has 2 inputs

plt.figure(figsize=(3, 3))
axes = plt.gca()
plt.xlabel('log(N)')
plt.ylabel('S: stress (MPa)')
plt.scatter(lgN[censor_indx == 1], S[censor_indx == 1], 12, marker="o",c="C0", label='Failures')
plt.scatter(lgN[censor_indx == 0], S[censor_indx == 0], 30, marker=">", edgecolors='r', facecolors='none', label='Runouts')
plt.plot(mu_pred,S_pred,c='m', label='Mean')
plt.plot(mu_pred-1.96*sigma_pred,S_pred,c='k',linestyle='--', label='95% CI')
plt.plot(mu_pred+1.96*sigma_pred,S_pred,c='k',linestyle='--')
plt.legend(loc='upper right',frameon=False)
plt.show()
