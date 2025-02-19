import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import pandas.api.types
from lifelines.utils import concordance_index

def drop_top_n_corr(df, n, col):
    c = df.corr()[col].abs().drop(col).sort_values(ascending=False)
    return df.drop(columns=c.head(n).index.tolist())


def encoding_binary(df, col):
    mapping = {'Yes': 1, 'No': 0}
    # Aplicar el mapeo
    df[col] = df[col].map(mapping)

    # Calcular la media global. Si es NaN (porque no se mapearon valores), se asigna 0.
    global_mean = df[col].mean()
    if pd.isna(global_mean):
        global_mean = 0

    # Rellenar los NaN con la media global (o el valor por defecto)
    df[col] = df[col].fillna(global_mean).astype(float)
    return df

def target_encoding(df, col, target):
    # Calcular la media global del target
    global_mean = df[target].mean()
    # Calcular la media del target por cada categoría en 'col'
    mapping = df.groupby(col)[target].mean()
    # Reemplazar los valores de la columna por la media correspondiente
    df[col] = df[col].map(mapping)
    # Rellenar posibles NaN con la media global
    df[col].fillna(global_mean, inplace=True)
    return df


def frequency_encoding(df, col):
    placeholder = '__nan__'
    df[col] = df[col].fillna(placeholder)
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq).astype(np.float32)
    return df

def one_hot_encoding(df,col):
    df[col] = df[col].replace(['Not done', 'Not one', np.nan],'missing')
    df = pd.get_dummies(df,columns=[col],prefix=col,dtype='int',drop_first=True)
    #df = df.drop(col+'_missing',axis=1,errors='ignore')
    return df




def impute_weighted_nan(df, col):
    # Calcula la distribución de frecuencias de los valores no nulos
    freq = df[col].dropna().value_counts(normalize=True)

    # Encuentra los índices donde la columna es NaN
    missing_indices = df[df[col].isna()].index

    # Si hay valores NaN, muestrea para cada uno
    if len(missing_indices) > 0:
        imputed_values = np.random.choice(freq.index, size=len(missing_indices), p=freq.values)
        df.loc[missing_indices, col] = imputed_values

    return df

def scale(df,col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())




def get_dfs():

    train=0
    test=0

    if 'KAGGLE_URL_BASE' in os.environ or os.path.exists('/kaggle'):
        print("Estás en un entorno Kaggle")
        train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
        test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')



    df = pd.concat([train, test], axis=0, join="outer", keys=["train", "test"])

    #ID

    #race_group
    df_dummies = pd.get_dummies(df['race_group'], prefix='race_group', dtype=int, drop_first=True)
    df = pd.concat([df, df_dummies], axis=1)

    #dri_score
    # Primero, reemplazamos los valores NaN por la cadena 'missing'
    df['dri_score'] = df['dri_score'].fillna('missing')

    # Definimos el diccionario de mapeo
    mapping = {
        'Intermediate': 1,
        'N/A - pediatric': -1,
        'High': 2,
        'N/A - non-malignant indication': 0,
        'TBD cytogenetics': -1,
        'Low': 0,
        'High - TED AML case <missing cytogenetics': 2,
        'Intermediate - TED AML case <missing cytogenetics': 1,
        'N/A - disease not classifiable': -1,
        'Very high': 3,
        'missing': -1,
        'Missing disease status': -1
    }

    # Convertimos a cadena (por si acaso) y aplicamos el mapeo, reasignando la columna
    df['dri_score'] = df['dri_score'].astype(str).map(mapping).astype(int)

    #psych_disturb
    col = 'psych_disturb'
    #df = encoding_binary(df, col)
    df = df.drop(col,axis=1)

    #cyto_score
    col = 'cyto_score'

    mapping = {
        'Favorable': 0,
        'Intermediate': 2,
        'Poor': 3,
        'Normal': 1,
        'Other': -1,
        'TBD': -1,
        'Not tested': -1,
        'nan': -1
    }

    df[col] = df[col].astype(str).map(mapping).astype(int)



    #diabetes
    col = 'diabetes'
    df = encoding_binary(df,'diabetes')

    #hla_match_c_high
    col = 'hla_match_c_high'
    df[col] = df[col].fillna(df[col].median())
    df = one_hot_encoding(df,col)

    #hla_high_res_8
    col = 'hla_high_res_8'
    #df[col] = df[col].fillna(df[col].median())
    df = df.drop(col,axis=1)

    #tbi_status
    col = 'tbi_status'
    df = one_hot_encoding(df, col)

    #arrhythmia
    col = 'arrhythmia'
    #df = encoding_binary(df, col)
    df = df.drop(col,axis=1)

    #hla_low_res_6
    col = 'hla_low_res_6'
    df[col] = df[col].fillna(df[col].median())


    #graft_type
    col = 'graft_type'
    mapping={'Peripheral blood':0,'Bone marrow':1}
    df[col] = df[col].map(mapping).astype(int)

    #vent_hist
    col ='vent_hist'
    df = encoding_binary(df, col)

    #renal_issue
    col = 'renal_issue'
    df = encoding_binary(df, col)

    #pulm_severe
    col = 'pulm_severe'
    df = encoding_binary(df, col)


    #prim_disease_hct
    #df = frequency_encoding(df,'prim_disease_hct')
    col = 'prim_disease_hct'
    df =  one_hot_encoding(df, col)

    #hla_high_res_6
    col = 'hla_high_res_6'
    df[col] = df[col].fillna(df[col].median())

    #cvm_status
    col = 'cmv_status'
    mapping = {'-/-': 0,'-/+': 1,'+/+': 2,'+/-': 3,'nan':-1}
    df[col] = df[col].astype(str).map(mapping).astype(int)

    #hla_high_res_10
    col = 'hla_high_res_10'
    #df[col]=df[col].fillna(df[col].median())
    df = df.drop(col,axis=1)

    #hla_match_dqb1_high
    col = 'hla_match_dqb1_high'
    df[col] = df[col].fillna(df[col].median())

    #tce_imm_match
    col = 'tce_imm_match'
    #df = frequency_encoding(df,col)
    df = df.drop(col,axis=1)

    #hla_nmdp_6
    col='hla_nmdp_6'
    df[col] = df[col].fillna(df[col].median())

    #hla_match_c_low
    col = 'hla_match_c_low'
    df[col] = df[col].fillna(df[col].median())

    #hla_match_drb1_low
    col = 'hla_match_drb1_low'
    df[col] = df[col].fillna(df[col].median())

    # rituximab
    col ='rituximab'
    df = encoding_binary(df, col)

    #hla_match_dqb1_low
    col = 'hla_match_dqb1_low'
    df[col] = df[col].fillna(df[col].median())

    #prod_type
    col = 'prod_type'
    mapping={'PB':0,'BM':1}
    df[col] = df[col].map(mapping).astype(int)

    # cyto_score_detail
    col = 'cyto_score_detail'
    mapping={'Poor':0,'Intermediate':1,'Favorable':2}
    df[col] = df[col].map(mapping).fillna(-1)

    #conditioning_intensity
    col = 'conditioning_intensity'
    df = frequency_encoding(df,col)

    #ethnicity
    col = 'ethnicity'
    df=frequency_encoding(df,col)

    #year_hct

    #obesity
    col ='obesity'
    df = encoding_binary(df, col)

    #mrd_hct
    col = 'mrd_hct'
    df = encoding_binary(df,col)

    #in_vivo_tcd
    col = 'in_vivo_tcd'
    df = encoding_binary(df, col)

    #tce_match
    col = 'tce_match'
    mapping={'Fully matched':0,'Permissive':1,'HvG non-permissive':2,'GvH non-permissive':3}
    df[col] = df[col].map(mapping).fillna(-1)

    #hla_match_a_high
    col = 'hla_match_a_high'
    df[col]  = df[col].fillna(df[col].median())

    #hla_match_a_high
    col = 'hla_match_a_high'
    df[col]= df[col].fillna(df[col].median())

    #hepatic_severe
    col = 'hepatic_severe'
    df = encoding_binary(df, col)

    #donor_age
    col = 'donor_age'
    df[col] = df[col].fillna(df[col].mean())
    df[col] = scale(df,col)

    #prior_tumor
    col = 'prior_tumor'
    df = encoding_binary(df, col)

    #hla_match_b_low
    col = 'hla_match_b_low'
    df[col] = df[col].fillna(df[col].median())

    #peptic_ulcer
    col = 'peptic_ulcer'
    df = encoding_binary(df, col)

    #age_at_hct
    col = 'age_at_hct'
    df[col].fillna(df[col].mean())
    df[col] =  scale(df,col)

    #hla_match_a_low
    col = 'hla_match_a_low'
    df[col] = df[col].fillna(df[col].median())

    #gvhd_proph
    col = 'gvhd_proph'
    df = frequency_encoding(df,col)

    #rheum_issue
    col = 'rheum_issue'
    df = encoding_binary(df, col)

    #sex_match
    col = 'sex_match'
    df = one_hot_encoding(df, col)

    #hla_match_b_high
    col = 'hla_match_b_high'
    df[col] = df[col].fillna(df[col].median())

    #race_group
    col = 'race_group'

    race_dummies = pd.get_dummies(df['race_group'], prefix='race', dtype=int)
    df = pd.concat([df, race_dummies], axis=1)

    #comorbidity_score
    col = 'comorbidity_score'
    df[col] = df[col].fillna(df[col].median())

    #karnofsky_score
    col = 'karnofsky_score'
    df[col] = df[col].fillna(df[col].median())
    df[col] = scale(df,col)

    #hepatic_mild
    col = 'hepatic_mild'
    df = encoding_binary(df, col)

    #tce_div_match
    mapping = {'Permissive mismatched': 0,'HvG non-permissive': 1,'GvH non-permissive': 2,'Bi-directional non-permissive': 3,'nan':-1}
    df['tce_div_match'] = df['tce_div_match'].astype(str).map(mapping).astype(int)

    #donor_related
    col = 'donor_related'
    df = one_hot_encoding(df,col)


    #melphalan_dose
    col = 'melphalan_dose'
    df = one_hot_encoding(df,col)

    #hla_low_res_8
    col = 'hla_low_res_8'
    df[col] = df[col].fillna(df[col].median())

    #cardiac
    col = 'cardiac'
    df = encoding_binary(df, col)

    #hla_match_drb1_high
    col = 'hla_match_drb1_high'
    df = encoding_binary(df, col)

    #pulm_moderate
    col = 'pulm_moderate'
    df = encoding_binary(df, col)

    #hla_low_res_10
    col = 'hla_low_res_10'
    df[col] = df[col].fillna(df[col].median())

    # score 0.6473492483275347
    # score 0.6453358050565493 df=df.drop(['tce_match', 'mrd_hct', 'cyto_score_detail', 'tce_div_match', 'tce_imm_match'],axis=1)
    # score 0.6474994978538315 df=df.drop(['cyto_score_detail', 'tce_div_match', 'tce_imm_match'],axis=1)

    df_train = df.loc["train"].copy()
    df_test = df.loc["test"].copy()

    return df_train,df_test




class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
            merged_df_race[interval_label],
            -merged_df_race[prediction_label],
            merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))

def predict(model, xdata):
    chf =  model.predict_cumulative_hazard_function(xdata.drop(['ID','efs','efs_time','race_group'],axis=1))
    risk_scores = np.array([fn(fn.x[-1]) for fn in chf])
    return risk_scores

def predict2(model, xdata):
    return model.predict(xdata.drop(['ID','efs','efs_time','race_group'],axis=1))

def train(model, xtrain ,weights=None):
    print('training '+ str(xtrain.shape[0]) + ' lines')
    ytrain = np.array([(bool(e), t) for e, t in zip(xtrain["efs"], xtrain["efs_time"])],
                       dtype=[("event", "bool"), ("time", "float")])

    model.fit(xtrain.drop(['ID','efs','efs_time','race_group'],axis=1),ytrain,sample_weight=weights)
    return model

def create_model(n_estimators=100, max_depth=10,min_samples_split=10,min_samples_leaf=5,n_jobs=1):
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth = max_depth,  # Limitar profundidad para evitar overfitting
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=42,
        low_memory=True
    )

    return model

def get_score(x, prediction):
    prediction = pd.DataFrame({
        'ID': x['ID'],
        'prediction': prediction
    })

    return score(x.copy(),prediction,'ID')



import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance

def get_relevant_features(df_train, threshold=0.01):
    print('returning relevant features...')
    # Crear copia para no modificar el original
    df_copy = df_train.copy()

    # Columnas que siempre deben estar incluidas
    mandatory_columns = ['ID', 'efs', 'efs_time'] + [col for col in df_copy.columns if col.startswith('race_')]

    # Eliminar `race_group` del análisis ya que es una variable string
    df_copy = df_copy.drop(columns=['race_group'], errors='ignore')

    # Seleccionar solo columnas numéricas para el análisis
    numeric_features = df_copy.select_dtypes(include=[np.number]).drop(columns=mandatory_columns, errors='ignore')

    # Crear el vector objetivo (evento y tiempo de supervivencia)
    y = np.array([(e, t) for e, t in zip(df_copy['efs'], df_copy['efs_time'])],
                 dtype=[('event', 'bool'), ('time', 'float')])

    # Entrenar un modelo de Random Survival Forest solo con variables numéricas
    rsf = RandomSurvivalForest(n_estimators=100, n_jobs=-1, random_state=42, low_memory=True)
    rsf.fit(numeric_features, y)

    # Usar permutation importance para obtener la importancia de las características
    perm_importance = permutation_importance(rsf, numeric_features, y, n_repeats=5, random_state=42, n_jobs=-1)

    # Obtener importancia de características
    feature_importances = pd.Series(perm_importance.importances_mean, index=numeric_features.columns)

    # Ordenar por importancia
    feature_importances = feature_importances.sort_values(ascending=False)

    # Seleccionar columnas más relevantes según el umbral
    relevant_columns = feature_importances[feature_importances >= threshold].index.tolist()

    # Incluir siempre las columnas obligatorias
    relevant_columns = list(set(relevant_columns + mandatory_columns))

    print("\nColumnas más relevantes para el modelo:", relevant_columns)

    return relevant_columns
