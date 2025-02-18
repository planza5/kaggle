<<<<<<< HEAD
import torch
import shap
import torchtuples as tt
from pycox.models import CoxPH
from sklearn.preprocessing import StandardScaler
import contest_lib as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Variables globales para el análisis de supervivencia
duration = 'efs_time'
event = 'efs'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CoxPH_L1(CoxPH):
    def __init__(self, net, optimizer, l1_lambda=0.0001, **kwargs):
        super().__init__(net, optimizer, **kwargs)
        self.l1_lambda = l1_lambda

    def compute_loss(self, batch):
        x, target = batch
        loss = super().compute_loss(batch)
        l1_norm = sum(p.abs().sum() for p in self.net.parameters())
        return loss + self.l1_lambda * l1_norm


def create_model(features):
    num_nodes = [512, 256, 128, 64]
    out_features = 1
    dropout = 0.4
    activation = torch.nn.ReLU

    net_model = tt.practical.MLPVanilla(
        in_features=features,
        num_nodes=num_nodes,
        out_features=out_features,
        dropout=dropout,
        batch_norm=True,
        activation=activation
    )

    model = CoxPH_L1(net_model, tt.optim.Adam)
    for param_group in model.optimizer.param_groups:
        param_group['lr'] = 0.001
        #param_group['weight_decay'] = 0.001

    model.net.to(device)
    return model


def train_and_score(df, batch_size=128, epochs=100, verbose=True):
    df_train = df.sample(frac=0.7, random_state=42)
    df_test = df.drop(df_train.index)

    y_train = (
        df_train['efs_time'].values.astype(np.float32),
        df_train['efs'].values.astype(np.float32)
    )
    y_test = (
        df_test['efs_time'].values.astype(np.float32),
        df_test['efs'].values.astype(np.float32)
    )

    x_train_df = df_train.drop(['ID', 'race_group', 'efs_time', 'efs'], axis=1)
    x_test_df = df_test.drop(['ID', 'race_group', 'efs_time', 'efs'], axis=1)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_df)
    x_test_scaled = scaler.transform(x_test_df)

    x_train = x_train_scaled.astype(np.float32)
    x_test = x_test_scaled.astype(np.float32)

    model = create_model(x_train.shape[1])
    model.fit(x_train, y_train, batch_size, epochs, callbacks=None, verbose=verbose)

    # --- PREDICCIONES EN TEST PARA TU MÉTRICA ---
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    risk_scores = model.net(x_test_tensor).detach().cpu().numpy().flatten()

    prediction = pd.DataFrame({
        'ID': df_test['ID'].values,
        'prediction': risk_scores
    })
    prediction.index = df_test.index

    score_value = cl.score(
        df_test[['ID', 'race_group', 'efs', 'efs_time']],
        prediction,
        'ID'
    )
    print("Final score:", score_value)

    # --- CÁLCULO SHAP ---
    print("Calculando valores SHAP...")

    # 1) Define la función de predicción
    def predict_fn(data_as_np):
        # data_as_np es un array o DataFrame: conviértelo a tensor
        data_as_torch = torch.tensor(data_as_np, dtype=torch.float32).to(device)
        # Pasa por la red y regresa como numpy
        with torch.no_grad():
            out = model.net(data_as_torch).cpu().numpy()
        return out

    # 2) Crea el explainer con la función de predicción y los datos de entrenamiento (x_train)
    explainer = shap.Explainer(
        predict_fn,     # la función de predicción
        x_train,        # conjunto de referencia (numpy)
        # Puedes agregar argumentaciones extra si gustas, p.e. algorithm='permutation'
    )

    # 3) Calcula los valores SHAP en test
    shap_values = explainer(x_test)

    # shap_values.values tendrá forma (num_samples_test, num_features)
    # Calculamos importancia promedio en valor absoluto
    shap_importance = pd.DataFrame({
        "Feature": x_train_df.columns,
        "SHAP Value (mean abs)": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="SHAP Value (mean abs)", ascending=False)

    print("Importancia de las variables:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(shap_importance)

    # 4) Visualización con SHAP
    #   Para la gráfica, se recomienda pasar un DataFrame o array con el mismo shape de test
    #   y las columnas en el mismo orden
    shap.summary_plot(shap_values.values, x_test_df)
    plt.show()

    return score_value


def main():
    features=['ID','race_group','efs','efs_time','conditioning_intensity_RIC',
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

    df_train, _ = cl.
    actual_score = train_and_score(df=df_train[features].copy(), verbose=True)
    print(actual_score)
=======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest
import pandas.api.types


def drop_top_n_corr(df, n, col):
    c = df.corr()[col].abs().drop(col).sort_values(ascending=False)
    return df.drop(columns=c.head(n).index.tolist())

import random
#0.6482 con fillna -1
#0.6485 con fillna mean
#0.649 con median
#0.6459 con mode
#con 5000 arboles score 0.6560653650996325


def encoding_binary(df,col):
    mapping = {
        'Yes': 1,
        'No': 0
    }

    df[col] = df[col].map(mapping).astype(float)
    df[col] = df[col].fillna(df[col].mean())

    return df

#def frequency_encoding(df,col):
#    te = df.groupby(col)['efs_time'].mean()
#    df[col] = df[col].map(te)

#    return df


def frequency_encoding(df, col):
    placeholder = '__nan__'
    df[col] = df[col].fillna(placeholder)
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq).astype(np.float32)
    return df

def one_hot_encoding(df,col):
    df[col] = df[col].replace(['Not done', 'Not one', np.nan],'missing')
    df = pd.get_dummies(df,columns=[col],prefix=col,dtype='int8')
    #df = df.drop(col+'_missing',axis=1,errors='ignore')
    return df

def get_dfs():
    import os

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

    #dri_score
    df = frequency_encoding(df,'dri_score')

    #psych_disturb
    df = one_hot_encoding(df,'psych_disturb')


    #cyto_score
    mapping = {
        'Favorable': 0,
        'Intermediate': 1,
        'Poor': 2,
        'Normal': 1,
        'Other': -1,
        'TBD': -1,
        'Not tested': -1,
        'nan': -1
    }

    df['cyto_score'] = df['cyto_score'].astype(str).map(mapping).astype(int)

    #diabetes
    df['diabetes'] = df['diabetes'].replace([np.nan,'Not done'],'missing')
    df = encoding_binary(df,'diabetes')

    #hla_match_c_high
    col = 'hla_match_c_high'
    df[col] = df[col].fillna(df[col].median())

    #hla_high_res_8
    col = 'hla_high_res_8'
    df[col] = df[col].fillna(df[col].median())

    #tbi_status
    col = 'tbi_status'
    df = frequency_encoding(df,col)
    #df = one_hot_encoding(df, col)

    #arrhythmia
    df = encoding_binary(df, 'arrhythmia')

    #hla_low_res_6
    col = 'hla_low_res_6'
    df[col] = df[col].fillna(df[col].median())

    #graft_type
    mapping={'Peripheral blood':0,'Bone marrow':1}
    df['graft_type'] = df['graft_type'].map(mapping).astype(int)

    #vent_hist
    df = encoding_binary(df, 'vent_hist')

    #renal_issue
    df = encoding_binary(df, 'renal_issue')

    #pulm_severe
    df = encoding_binary(df, 'pulm_severe')

    #prim_disease_hct
    df = frequency_encoding(df,'prim_disease_hct')

    #hla_high_res_6
    df['hla_high_res_6'] = df['hla_high_res_6'].fillna(df['hla_high_res_6'].mean())

    #cvm_status
    mapping = {'-/-': 0,'-/+': 1,'+/+': 2,'+/-': 3,'nan':-1}

    df['cmv_status'] = df['cmv_status'].astype(str).map(mapping).astype(int)

    #hla_high_res_10
    df['hla_high_res_10']=df['hla_high_res_10'].fillna(df['hla_high_res_10'].median())

    #hla_match_dqb1_high
    df['hla_match_dqb1_high'] = df['hla_match_dqb1_high'].fillna(df['hla_match_dqb1_high'].median())

    #tce_imm_match
    frequency_encoding(df,'tce_imm_match')

    #hla_nmdp_6
    df['hla_nmdp_6'] = df['hla_nmdp_6'].fillna(df['hla_nmdp_6'].median())

    #hla_match_c_low
    df['hla_match_c_low'] = df['hla_match_c_low'].fillna(df['hla_match_c_low'].median())

    #hla_match_drb1_low
    col = 'hla_match_drb1_low'
    df[col] = df[col].fillna(df[col].median())

    # rituximab
    df = encoding_binary(df, 'rituximab')

    #hla_match_dqb1_low
    df['hla_match_dqb1_low'] = df['hla_match_dqb1_low'].fillna(df['hla_match_dqb1_low'].median())

    #prod_type
    mapping={'PB':0,'BM':1}
    df['prod_type'] = df['prod_type'].map(mapping).astype(int)

    # cyto_score_detail
    mapping={'Poor':0,'Intermediate':1,'Favorable':2}
    df['cyto_score_detail'] = df['cyto_score_detail'].map(mapping).fillna(-1)

    #conditioning_intensity
    df['conditioning_intensity'] = df['conditioning_intensity'].replace([np.nan, 'TBD', 'No drugs reported','N/A, F(pre-TED) not submitted'], 'missing')
    df = one_hot_encoding(df,'conditioning_intensity')

    #ethnicity
    df=frequency_encoding(df,'ethnicity')

    #year_hct

    #obesity
    df = encoding_binary(df, 'obesity')

    #mrd_hct
    mapping = {'Negative':0,'Positive':1}
    df['mrd_hct'] = df['mrd_hct'].map(mapping).fillna(-1).astype(int)

    #in_vivo_tcd
    df = encoding_binary(df, 'in_vivo_tcd')

    #tce_match
    mapping={'Fully matched':0,'Permissive':1,'HvG non-permissive':2,'GvH non-permissive':3}
    df['tce_match'] = df['tce_match'].map(mapping).fillna(-1)

    #hla_match_a_high
    df['hla_match_a_high'] = df['hla_match_a_high'].fillna(df['hla_match_a_high'].median())

    #hla_match_a_high
    col = 'hla_match_a_high'
    df[col].fillna(df[col].median())

    #hepatic_severe
    df = encoding_binary(df, 'hepatic_severe')


    #donor_age
    col = 'donor_age'
    df[col] = df[col].fillna(df[col].mean())

    #prior_tumor
    df = encoding_binary(df, 'prior_tumor')

    #hla_match_b_low
    col = 'hla_match_b_low'
    df[col] = df[col].fillna(df[col].median())

    #peptic_ulcer
    df = encoding_binary(df, 'peptic_ulcer')

    #age_at_hct
    col = 'age_at_hct'
    df[col].fillna(df[col].mean())

    #hla_match_a_low
    col = 'hla_match_a_low'
    df[col] = df[col].fillna(df[col].median())

    #gvhd_proph
    frequency_encoding(df,'gvhd_proph')

    #rheum_issue
    df = encoding_binary(df, 'rheum_issue')

    #sex_match
    col = 'sex_match'
    df = one_hot_encoding(df, col)

    #hla_match_b_high
    col = 'hla_match_b_high'
    df[col] = df[col].fillna(df[col].median())

    #race_group
    #col = 'race_group'
    #df = one_hot_encoding(df,col)

    #comorbidity_score
    col = 'comorbidity_score'
    df[col] = df[col].fillna(df[col].median())

    #karnofsky_score
    col = 'karnofsky_score'
    df[col] = df[col] = df[col].fillna(df[col].median())

    #hepatic_mild
    df = encoding_binary(df, 'hepatic_mild')

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
    df = encoding_binary(df, 'cardiac')

    #hla_match_drb1_high
    df = encoding_binary(df, 'hla_match_drb1_high')

    #pulm_moderate
    df = encoding_binary(df, 'pulm_moderate')

    #hla_low_res_10
    col = 'hla_low_res_10'
    df[col] = df[col].fillna(df[col].median())

    # score 0.6473492483275347
    # score 0.6453358050565493 df=df.drop(['tce_match', 'mrd_hct', 'cyto_score_detail', 'tce_div_match', 'tce_imm_match'],axis=1)
    # score 0.6474994978538315 df=df.drop(['cyto_score_detail', 'tce_div_match', 'tce_imm_match'],axis=1)




    df_train = df.loc["train"].copy()
    df_test = df.loc["test"].copy()


    return df_train,df_test


def top_n_correlations(df, col_target, exclude_cols, n=50):
    # Eliminar las columnas excluidas antes de calcular correlaciones
    df_filtrado = df.drop(columns=exclude_cols, errors='ignore')

    # Calcular correlaciones con col_target
    correlaciones = df_filtrado.corr()[col_target].dropna().drop(col_target, errors='ignore')

    # Obtener las N/2 correlaciones más altas (positivas) y más bajas (negativas)
    top_positivas = correlaciones.nlargest(n // 2)
    top_negativas = correlaciones.nsmallest(n // 2)

    # Columnas seleccionadas: las excluidas, la target y las top correlacionadas
    columnas_seleccionadas = exclude_cols + [col_target] + list(top_positivas.index) + list(top_negativas.index)

    # Retornar el dataframe con solo las columnas seleccionadas
    return columnas_seleccionadas


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

import pandas as pd
from lifelines.utils import concordance_index

def calcular_cindex_por_grupo(X_valid, predict, id_cols=['ID', 'race_group', 'efs', 'efs_time']):
    result = X_valid[id_cols].copy()
    result['risk_score'] = predict
    groups = result.groupby('race_group')
    cindex_per_group = {}
    for group_name, group_data in groups:
        c_index_group = concordance_index(
            group_data['efs_time'],
            -group_data['risk_score'],
            group_data['efs']
        )
        cindex_per_group[group_name] = c_index_group
        print(f"Grupo: {group_name}, C-index: {c_index_group:.3f}")
    return cindex_per_group

def get_score(x, prediction):
    prediction = pd.DataFrame({
        'ID': x['ID'],
        'prediction': prediction
    })

    return score(x.copy(),prediction,'ID')

def train(model, xtrain ,weights=None):
    print('training....')
    ytrain = np.array([(bool(e), t) for e, t in zip(xtrain["efs"], xtrain["efs_time"])],
                       dtype=[("event", "bool"), ("time", "float")])

    model.fit(xtrain.drop(['ID','efs','efs_time','race_group'],axis=1),ytrain,sample_weight=weights)
    return model

def predict(model, xdata):
    print('predicting....')
    chf =  model.predict_cumulative_hazard_function(xdata.drop(['ID','efs','efs_time','race_group'],axis=1))
    risk_scores = np.array([fn(fn.x[-1]) for fn in chf])
    return risk_scores

def normalization(df, exclude_columns=None):
    numeric_columns = df.select_dtypes(exclude=['object']).columns
    columns_to_normalize = [
        col for col in numeric_columns
        if col not in exclude_columns and not (set(df[col].unique()) <= {0, 1})
    ]
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def best_correlated_features(df, target_column, n=10, not_eligible=None):
    if not_eligible is None:
        not_eligible = []
    columns_a_excluir = ["race_group"] + not_eligible
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=columns_a_excluir, errors="ignore")
    if target_column not in numeric_df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no es numérica o no se encontró en el DataFrame.")
    corr_series = numeric_df.corr()[target_column].drop(labels=[target_column])
    positive_corr = corr_series[corr_series > 0].sort_values(ascending=False)
    negative_corr = corr_series[corr_series < 0].sort_values(ascending=True)
    half = n // 2
    top_positive = positive_corr.head(half).index.tolist()
    top_negative = negative_corr.head(half).index.tolist()
    return top_positive + top_negative + df[target_column].values() + df[not_eligible].values()

def main():
    X, X_submit = get_dfs()

    #X= normalization(X,['ID','efs','efs_time'])
    #features = best_correlated_features(X, "efs_time", 60, ['efs'])
    #X = X[features]
    X_train, X_valid = train_test_split(X, test_size=0.20, random_state=42)

    rsf = RandomSurvivalForest(n_estimators=30, n_jobs=2, min_samples_split=15, min_samples_leaf=10, random_state=42, low_memory=False,verbose=4)
    rsf = train(rsf, X_train)
    prediction = predict(rsf, X_valid)
    print('prediction....',prediction)
    print(get_score(X_valid, prediction))

    group_cindexes = calcular_cindex_por_grupo(X_valid,prediction)
    epsilon= 1e-6

    weights = {group: 1 / (cindex + epsilon) for group, cindex in group_cindexes.items()}
    total_weight = sum(weights.values())
    weights = {group: weight / total_weight for group, weight in weights.items()}

    mapping_weights = X_train['race_group'].map(weights).values

    if mapping_weights is None or np.any(np.isnan(mapping_weights)):
        raise ValueError("Algunos valores de race_group no tienen pesos asociados.")

    rsf = train(rsf,X_train,weights=mapping_weights)
    prediction = predict(rsf, X_valid)
    print(get_score(X_valid, prediction))

>>>>>>> 42ce8126f45679dd311a7d88d53f711a7af3d590


if __name__ == "__main__":
    main()
<<<<<<< HEAD
=======

>>>>>>> 42ce8126f45679dd311a7d88d53f711a7af3d590
