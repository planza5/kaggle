import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest


def drop_top_n_corr(df, n, col):
    c = df.corr()[col].abs().drop(col).sort_values(ascending=False)
    return df.drop(columns=c.head(n).index.tolist())

def eliminar_multicolinealidad(df, umbral=5.0):

    variables = df.columns.tolist()
    while True:
        # Calcular VIF para cada variable
        vif = pd.DataFrame()
        vif["Variable"] = variables
        vif["VIF"] = [variance_inflation_factor(df[variables].values, i) for i in range(len(variables))]

        # Encontrar la variable con el VIF más alto
        max_vif = vif["VIF"].max()
        if max_vif > umbral:
            # Identificar la variable con el VIF más alto
            variable_a_eliminar = vif.loc[vif["VIF"] == max_vif, "Variable"].values[0]
            print(f"Eliminando '{variable_a_eliminar}' con VIF: {max_vif:.2f}")
            variables.remove(variable_a_eliminar)
        else:
            break

    return df[variables]

def encoding_binary(df,col):
    mapping = {
        'Yes': 1,
        'No': 0
    }

    df[col] = df[col].map(mapping).replace(np.nan,-1).astype(int)

    return df

def frequency_encoding(df, col):
    placeholder = '__nan__'
    df[col] = df[col].fillna(placeholder)
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq).astype(np.float32)
    return df

def one_hot_encoding(df,col):
    df[col] = df[col].replace(['Not done', 'Not one', np.nan],'missing')
    df = pd.get_dummies(df,columns=[col],prefix=col,dtype='int8')
    df = df.drop(col+'_missing',axis=1,errors='ignore')
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
    df.drop(['ethnicity'], axis=1, inplace=True)

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
    df = df.drop(['hla_match_drb1_high'],axis=1)

    #pulm_moderate
    df = encoding_binary(df, 'pulm_moderate')

    #hla_low_res_10
    col = 'hla_low_res_10'
    df[col] = df[col].fillna(df[col].median())


    df_train = df.loc["train"].copy()
    df_test = df.loc["test"].copy()


    return df_train,df_test




pd.set_option("display.max_rows", None)  # Muestra todas las filas
pd.set_option("display.max_columns", None)  # Muestra todas las columnas
pd.set_option("display.width", 1000)  # Aumenta el ancho de impresión
pd.set_option("display.max_colwidth", None)  # No recorta los valores largos



X, X_submit = get_dfs()

X_train, X_valid = train_test_split(X, test_size=0.2, random_state=42)

y_train = np.array([(bool(e), t) for e, t in zip(X_train["efs"], X_train["efs_time"])],
             dtype=[("event", "bool"), ("time", "float")])

X_train = X_train.drop(['ID','race_group'],axis=1).copy()

print('Entrenando modelo con '+ str(len(X_train.columns)) + ' columnas')

cph = CoxPHFitter()
cph.fit(X_train, duration_col='efs_time', event_col='efs')
print('Fin entrenando modelo...')

cph.summary.to_csv('summary.csv')

print('Predicción.....')
ids={'ID':X_valid['ID'],'race_group':X_valid['race_group'],'efs':X_valid['efs'],'efs_time':X_valid['efs_time']}
X_valid = X_valid.drop(['ID','race_group','efs','efs_time'],axis=1).copy()
predict = cph.predict_expectation(X_valid)
print('Fin Predicción.....')

X_valid['ID']=ids['ID']
X_valid['race_group']=ids['race_group']



import score

y_valid = pd.DataFrame()
y_valid['ID']=ids['ID']
y_valid['prediction'] = predict
y_valid['efs'] = ids['efs']
y_valid['efs_time'] = ids['efs_time']

s=score.score(X_valid,y_valid,'ID')
print('score',s)

#submit
ids=X_submit['ID']


X_submit = X_submit.drop(['ID','race_group'],axis=1)
prediction = cph.predict_expectation(X_submit)

submit_prediction = pd.DataFrame({
    'ID': ids,
    'prediction': prediction
})

submit_prediction.to_csv('submission.csv',index=False)