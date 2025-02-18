import xgboost as xgb
import pandas as pd
import numpy as np
import contest_lib as cl
from sklearn.model_selection import train_test_split

# 1️⃣ Cargar el dataset
df, df_submit = cl.get_dfs()



features =['ID','race_group','efs','efs_time',
'conditioning_intensity_RIC',
'conditioning_intensity_MAC',
'prim_disease_hct',
'year_hct',
'sex_match_F-M',
'hla_high_res_6',
'donor_related_Unrelated',
'gvhd_proph',
'sex_match_M-F',
'comorbidity_score',
'dri_score',
'hla_match_a_high',
'mrd_hct',
'tbi_status',
'age_at_hct',
'conditioning_intensity_NMA',
'cmv_status',
'in_vivo_tcd',
'graft_type',
'hla_match_b_low',
'cyto_score',
'hla_low_res_8',
'tce_div_match',
'hla_low_res_10',
'cyto_score_detail',
'hla_match_b_high',
'hla_match_c_high',
'hla_high_res_10',
'hla_match_c_low',
'karnofsky_score']

df = df[features].copy()

# 2️⃣ Verificar que no haya valores NaN o Inf antes de continuar
assert not df["efs_time"].isna().any(), "efs_time tiene NaNs"
assert not df["efs"].isna().any(), "efs tiene NaNs"

# 3️⃣ Dividir en entrenamiento (80%) y validación (20%)
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)

# 4️⃣ Preparar X e y
X_train = df_train.drop(columns=["efs_time", "efs", "ID", "race_group"]).astype(np.float32)
X_valid = df_valid.drop(columns=["efs_time", "efs", "ID", "race_group"]).astype(np.float32)

# 5️⃣ Crear las etiquetas AFT para XGBoost
def create_aft_labels(df, event_col, time_col):
    y_lower = df[time_col].astype(np.float32).values
    y_upper = np.where(df[event_col] == 1, y_lower, np.inf).astype(np.float32)
    return y_lower, y_upper

y_train_lower, y_train_upper = create_aft_labels(df_train, "efs", "efs_time")
y_valid_lower, y_valid_upper = create_aft_labels(df_valid, "efs", "efs_time")

# 6️⃣ Convertir a DMatrix
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info("label_lower_bound", y_train_lower)
dtrain.set_float_info("label_upper_bound", y_train_upper)

dvalid = xgb.DMatrix(X_valid)
dvalid.set_float_info("label_lower_bound", y_valid_lower)
dvalid.set_float_info("label_upper_bound", y_valid_upper)

# 7️⃣ Definir hiperparámetros de XGBoost
params = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "max_depth": 5,  # Más profundidad para captar relaciones más complejas
    "eta": 0.05,  # Aprendizaje más lento pero estable
    "subsample": 0.8,  # Evitar overfitting
    "colsample_bytree": 0.8,
    "seed": 42
}

# 8️⃣ Entrenar el modelo XGBoost
model = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dvalid, "validation")])

# 9️⃣ Predecir en conjunto de validación
predictions = model.predict(dvalid)

# 🔟 Calcular C-index usando contest_lib
y_valid_df = pd.DataFrame({
    "ID": df_valid["ID"],
    "prediction": predictions
})



c_index = cl.score(df_valid, y_valid_df, "ID")
print("C-index:", c_index)
