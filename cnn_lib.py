import pandas as pd
import numpy as np
from lifelines.utils import concordance_index


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            print('error!!!!!!!!!!!!')
            exit(1)
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

def get_score(solution, prediction):
    prediction = pd.DataFrame({
        'ID': solution['ID'],
        'prediction': prediction
    })

    return score(solution.copy(),prediction)

def get_defs(csv1: str = 'train.csv', csv2: str = 'test.csv', exclude_columns: list = None):
    """
    Función que lee dos archivos CSV y devuelve dos dataframes procesados.

    Parámetros:
        csv1 (str, opcional): Ruta al primer archivo CSV. Por defecto 'train.csv'.
        csv2 (str, opcional): Ruta al segundo archivo CSV. Por defecto 'test.csv'.
        exclude_columns (list, opcional): Lista de nombres de columnas que se excluirán del procesamiento.
                                          Por defecto se considera una lista vacía.

    Operaciones realizadas sobre las columnas (excepto las excluidas):
        1. Se concatenan verticalmente las columnas a procesar de ambos archivos para aplicar un one-hot encoding conjunto
           sobre las columnas no numéricas.
        2. Se separa el dataframe combinado para obtener dos dataframes con las mismas columnas.
        3. Se reemplazan los NaN por la media de cada columna y se normalizan los valores (de 0 a 1) de forma independiente.
        4. Se reincorporan las columnas excluidas sin modificaciones.

    Retorna:
        Tuple con dos dataframes procesados.
    """
    # Si exclude_columns es None, se asigna una lista vacía.
    if exclude_columns is None:
        exclude_columns = []

    # Leer los archivos CSV
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Filtrar las columnas de exclusión que realmente existen en cada dataframe
    exclude_df1 = [col for col in exclude_columns if col in df1.columns]
    exclude_df2 = [col for col in exclude_columns if col in df2.columns]

    # Definir las columnas a procesar (excluyendo las que no deben procesarse)
    cols_proc_df1 = [col for col in df1.columns if col not in exclude_df1]
    cols_proc_df2 = [col for col in df2.columns if col not in exclude_df2]

    # Separar el dataframe en dos partes: las columnas a procesar y las columnas excluidas
    df1_proc = df1[cols_proc_df1].copy().reset_index(drop=True)
    df2_proc = df2[cols_proc_df2].copy().reset_index(drop=True)
    df1_exclude = df1[exclude_df1].copy().reset_index(drop=True) if exclude_df1 else pd.DataFrame()
    df2_exclude = df2[exclude_df2].copy().reset_index(drop=True) if exclude_df2 else pd.DataFrame()

    # Concatenar verticalmente para aplicar un one-hot encoding conjunto y que ambos dataframes tengan las mismas columnas
    combined = pd.concat([df1_proc, df2_proc], axis=0, ignore_index=True)

    # Detectar columnas no numéricas y aplicar one-hot encoding (incluyendo dummy para NaNs)
    non_numeric_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric_cols:
        combined = pd.get_dummies(combined, columns=non_numeric_cols, dummy_na=True)

    # Dividir nuevamente el dataframe combinado en los dos conjuntos originales
    len_df1 = df1_proc.shape[0]
    df1_encoded = combined.iloc[:len_df1].copy().reset_index(drop=True)
    df2_encoded = combined.iloc[len_df1:].copy().reset_index(drop=True)

    # Función interna para reemplazar NaN por la media y normalizar (escala 0 a 1) de forma independiente
    def fill_and_scale(df: pd.DataFrame) -> pd.DataFrame:
        # Reemplazar NaNs en columnas numéricas
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
        # Normalización de columnas numéricas
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0
        return df

    # Aplicar el procesamiento (imputación y normalización) de forma independiente en cada dataframe
    df1_final_proc = fill_and_scale(df1_encoded)
    df2_final_proc = fill_and_scale(df2_encoded)

    # Reincorporar las columnas excluidas sin procesar
    df1_final = pd.concat([df1_final_proc, df1_exclude.reset_index(drop=True)], axis=1)
    df2_final = pd.concat([df2_final_proc, df2_exclude.reset_index(drop=True)], axis=1)

    return df1_final, df2_final
