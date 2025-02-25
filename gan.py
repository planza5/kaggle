from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

import otra_forma.io_utiil as io

X, s = io.load_dataframes()


# Codifica 'race_group'

xtrain = X.drop['ID','race_group']

generator = Sequential()
generator.add(Dense(64, input_dim=X.shape[1], activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))
generator.add(Dropout(0.2))
generator.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
generator.add(Dense(1, activation='linear'))






