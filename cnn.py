import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import otra_forma.io_utiil as io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. FUNCIÓN SCORE (asegúrate de tenerla en tu script)
#    Ejemplo: la que tú proporcionaste.
# ------------------------------------------------------
from lifelines.utils import concordance_index  # si lo necesitas



def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'

    # Verifica que 'submission' solo tenga columnas numéricas
    # (excepto 'ID' si la incluyes, deberás adaptarlo).
    for col in submission.columns:
        if col != 'ID' and not pd.api.types.is_numeric_dtype(submission[col]):
            print('error!!!!!!!!!!!!')
            exit(1)

    # "Merge" de solution y submission.
    # Tu función hace un concat por columnas (axis=1), no un merge real.
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)

    # Agrupar por 'race_group'
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)

    metric_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]

        # Calcular el c-index para este grupo
        c_index_race = concordance_index(
            merged_df_race[interval_label],
            -merged_df_race[prediction_label],  # ojo el negativo, depende de tu lógica
            merged_df_race[event_label]
        )
        metric_list.append(c_index_race)

    # Promedio de los c-index por raza - sqrt de la varianza.
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))


# ------------------------------------------------------
# 2. CARGA DE DATAFRAMES
# ------------------------------------------------------
train, submit = io.load_dataframes()

# Extrae las columnas objetivo
y_data = train[['efs', 'efs_time', 'race_group']].copy()
y_submit = submit[['efs', 'efs_time', 'race_group']].copy()



# Codifica 'race_group'
le = LabelEncoder()
y_data['race_group'] = le.fit_transform(y_data['race_group'])
y_submit = le.transform(y_submit['race_group'])

# Prepara X
X = train.drop(['ID', 'efs', 'efs_time', 'race_group'], axis=1)
X_submit = submit.drop(['ID','efs','efs_time','race_group'], axis=1, errors='ignore')

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype('float32')
X_submit_scaled = scaler.transform(X_submit).astype('float32')

# ------------------------------------------------------
# 3. SPLIT: ENTRENAMIENTO vs VALIDACIÓN
# ------------------------------------------------------
X_trn, X_val, y_trn, y_val = train_test_split(
    X_scaled,
    y_data.values,
    test_size=0.2,
    random_state=42
)




# ------------------------------------------------------
# 4. DEFINIR LA FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------
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
    # Combinar 50% y 50%
    return 0.15 * loss_cox + 0.85 * loss_cindex


# ------------------------------------------------------
# 5. CONSTRUIR Y COMPILAR EL MODELO
# ------------------------------------------------------
model = Sequential()
model.add(Dense(64, input_dim=X_trn.shape[1], activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1, activation='linear'))


optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=combined_loss)



early_stop = EarlyStopping(
    monitor='val_loss',        # la métrica de validación que vigilas
    patience=5,                # número de épocas sin mejora que toleras
    restore_best_weights=True  # si quieres restaurar el mejor modelo
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # reduce la LR a la mitad
    patience=3,          # si no mejora en 3 épocas seguidas
    min_lr=1e-7
)

# ------------------------------------------------------
# 6. ENTRENAR CON VALIDACIÓN
# ------------------------------------------------------
history = model.fit(
    X_trn,
    y_trn,
    epochs=50,  # Pónlo en 100 o más si lo deseas
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[reduce_lr,early_stop]
)

# ------------------------------------------------------
# 7. GRAFICAR LA EVOLUCIÓN DE LAS PÉRDIDAS
# ------------------------------------------------------
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('C-Index Loss')
plt.legend()
plt.title('Evolución de la Pérdida')
plt.show()

# ------------------------------------------------------
# 8. PREDECIR EN EL CONJUNTO DE VALIDACIÓN Y CALCULAR EL SCORE
# ------------------------------------------------------
val_preds = model.predict(X_val)

# y_val tiene la forma (N, 3) = (efs, efs_time, race_group).
# Construimos un DataFrame 'solution_val' con esas columnas.
solution_val = pd.DataFrame({
    'ID': range(len(X_val)),  # Opcional, si deseas 'ID'
    'efs': y_val[:, 0],
    'efs_time': y_val[:, 1],
    'race_group': y_val[:, 2]
})

# 'submission_val' con la predicción
submission_val = pd.DataFrame({
    'ID': range(len(X_val)),  # debe coincidir con 'solution_val'
    'prediction': val_preds.flatten()
})

# Llamamos a la función 'score' que definiste arriba.
val_score = score(solution_val, submission_val)
print("Score en el conjunto de validación:", val_score)

# ------------------------------------------------------
# 9. USO FINAL SOBRE 'submit' Y GUARDA
# ------------------------------------------------------
# Para predecir en el DataFrame 'submit':

submit_predictions = model.predict(X_submit_scaled)

results = pd.DataFrame({
    'ID': submit['ID'],
    'risk_score': submit_predictions.flatten()
})

results.to_csv('submission.csv', index=False)
print("\nEntrenamiento completado y predicciones guardadas en 'submission.csv'.")
