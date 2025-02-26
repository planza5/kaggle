import tensorflow as tf
import pandas as pd
from keras import Sequential, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
import cnn_lib

def create_stratification_column(df, columns=['efs', 'race_group']):
    temp_df = df[columns].astype(str)
    strat_col = temp_df[columns[0]]

    for col in columns[1:]:
        strat_col = strat_col + "_" + temp_df[col]

    return strat_col

def stratified_c_index_loss(y_true, y_pred):
    """
    Función de pérdida basada en C-Index suavizado, estratificado por race_group.
    y_true[:, 0] = efs
    y_true[:, 1] = efs_time
    y_true[:, 2] = race_group
    y_pred = risk score predicho
    """
    event = tf.cast(y_true[:, 0], dtype=tf.float32)  # efs
    time = tf.cast(y_true[:, 1], dtype=tf.float32)  # efs_time
    race_group = tf.cast(y_true[:, 2], dtype=tf.int32)  # grupo racial
    risk_score = y_pred

    def compute_race_loss(race):
        mask = tf.equal(race_group, race)

        risk_race = tf.boolean_mask(risk_score, mask)
        time_race = tf.boolean_mask(time, mask)
        event_race = tf.boolean_mask(event, mask)

        valid_pairs = tf.logical_and(
            tf.expand_dims(event_race, 1) > 0,
            tf.expand_dims(time_race, 1) < tf.expand_dims(time_race, 0)
        )
        valid_pairs = tf.cast(valid_pairs, dtype=tf.float32)

        risk_diff = tf.expand_dims(risk_race, 1) - tf.expand_dims(risk_race, 0)
        concordant_pairs = tf.nn.sigmoid(risk_diff)

        numerator = tf.reduce_sum(concordant_pairs * valid_pairs)
        denominator = tf.reduce_sum(valid_pairs) + K.epsilon()

        c_index_race = numerator / denominator
        return c_index_race

    unique_races = tf.unique(race_group)[0]
    race_losses = tf.map_fn(compute_race_loss, unique_races, fn_output_signature=tf.float32)
    return tf.reduce_mean(race_losses)

def cox_partial_log_likelihood_tf(y_true, y_pred):
    """
    y_true: Tensor de forma (N, 2)
        - y_true[:, 0] = tiempos (t_i)
        - y_true[:, 1] = eventos (delta_i), 1 si evento, 0 si censurado
    y_pred: Tensor de forma (N,) o (N,1)
        - h_i = log-riesgo o score predicho por la red

    Devuelve: escalar (pérdida). Cuanto menor, mejor es el modelo
              (porque es -log-verosimilitud).
    """
    # Asegurar que sean float
    times = tf.cast(y_true[:, 0], dtype=tf.float32)
    events = tf.cast(y_true[:, 1], dtype=tf.float32)
    # Aplanar la predicción si viene en forma (N,1)
    hazards = tf.reshape(y_pred, [-1])

    # Ordenar individuos por tiempo descendente (para aplicar cumsum)
    order = tf.argsort(times, direction='DESCENDING')
    sorted_times = tf.gather(times, order)
    sorted_events = tf.gather(events, order)
    sorted_hazards = tf.gather(hazards, order)

    # Calcular exp(hazards) y luego su cumsum (para el denominador)
    exp_hazard = tf.exp(sorted_hazards)
    log_cumsum_hazard = tf.math.log(tf.cumsum(exp_hazard, axis=0))

    # Filtrar solo los individuos con evento=1
    event_indices = tf.where(tf.equal(sorted_events, 1.0))
    # h_i solo para eventos
    event_hazards = tf.gather(sorted_hazards, event_indices)
    event_log_cumsum = tf.gather(log_cumsum_hazard, event_indices)

    # partial log-likelihood = sum(h_event - log(sum_j>=i exp(h_j)))
    partial_ll = tf.reduce_sum(event_hazards - event_log_cumsum)
    # Queremos minimizar la pérdida => retornamos la negativa
    return -partial_ll

def combined_loss(y_true, y_pred):
    loss_cox = cox_partial_log_likelihood_tf(y_true, y_pred)
    loss_cindex = stratified_c_index_loss(y_true, y_pred)

    return 0.20 * loss_cox + 0.80 * loss_cindex

def create_basic_model(num_features):
    model = Sequential()
    model.add(Dense(64, input_dim=num_features, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=combined_loss)

    return model

def train(model, X_trn, y_trn, X_val, y_val, epochs=50):
    # Convertir los DataFrames a arrays numpy con tipos de datos adecuados
    # Para las características
    X_trn_np = X_trn.values.astype('float32')
    X_val_np = X_val.values.astype('float32')

    # Para las variables objetivo
    y_trn_np = y_trn.values.astype('float32')
    y_val_np = y_val.values.astype('float32')

    # Definir callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

    # Entrenar el modelo con los datos convertidos
    history = model.fit(
        X_trn_np,
        y_trn_np,
        epochs=epochs,
        batch_size=16,
        validation_data=(X_val_np, y_val_np),
        verbose=2,
        callbacks=[reduce_lr, early_stop]
    )

    return history


def main():
    X, submit_data = cnn_lib.get_defs(exclude_columns=['ID', 'efs', 'efs_time', 'race_group'])

    label_encoder = LabelEncoder()
    label_encoder.fit(X['race_group'])  # Primero ajustamos con los datos de entrenamiento

    # Luego aplicamos la transformación a ambos conjuntos
    X['race_group'] = label_encoder.transform(X['race_group'])
    submit_data['race_group'] = label_encoder.transform(submit_data['race_group'])

    #preparamos la y
    y = X[['efs','efs_time','race_group']]

    #dividimos datos
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    ids = X_val[['ID']]

    #eliminamos columnas indeseadas
    X_train = X_train.drop(['ID','efs','efs_time','race_group'],axis=1)
    X_val = X_val.drop(['ID','efs','efs_time','race_group'],axis=1)

    #creamos modelos y entrenamos
    model = create_basic_model(X_train.shape[1])
    train(model, X_train, y_train, X_val, y_val, 10)

    #predecimos y score
    prediction = model.predict(X_val.values.astype('float32'))
    y_val['ID'] = ids['ID']

    score = cnn_lib.get_score(y_val, prediction.flatten())
    print('scoring....' + str(score))
    pass

if __name__ == "__main__":
    main()

