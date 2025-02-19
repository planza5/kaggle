from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import main_rsf  # Importar módulo externo


def get_fair_kfolds(df_train, n_splits=5):
    # Hacer una copia del DataFrame para no modificar el original
    df_copy = df_train.copy()

    # Convertir efs_time a años
    df_copy['efs_time_years'] = df_copy['efs_time'] / 12

    # Crear bins de tiempo basados en los cuartiles reales
    bins = [0, df_copy['efs_time_years'].quantile(0.25), df_copy['efs_time_years'].quantile(0.50),
            df_copy['efs_time_years'].quantile(0.75), df_copy['efs_time_years'].max()]
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    df_copy['time_bin_real'] = pd.cut(df_copy['efs_time_years'], bins=bins, labels=labels, include_lowest=True)

    # Crear la variable de estratificación combinando evento y tiempo bineado
    df_copy['strata'] = df_copy['efs'].astype(str) + "_" + df_copy['time_bin_real'].astype(str)

    # Aplicar validación cruzada estratificada con `strata`
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generar los folds con solo los índices para ahorrar memoria
    folds = []
    for train_idx, test_idx in skf.split(df_copy, df_copy['strata']):
        folds.append({'train_idx': train_idx, 'test_idx': test_idx})

    return folds

def main():
    # Obtener los DataFrames desde el módulo externo
    df_train, df_submit = main_rsf.get_dfs()

    df_train = main_rsf.get_relevant_features(df_train,0.005)

    # Generar folds con el dataset de entrenamiento
    folds = get_fair_kfolds(df_train)

    # Crear el modelo


    # Lista para almacenar métricas
    all_scores = []

    # Recorrer los folds y extraer los DataFrames de entrenamiento y test
    for i, fold in enumerate(folds):
        model = main_rsf.create_model(n_estimators=100,n_jobs=2)

        train_df = df_train.iloc[fold['train_idx']]
        test_df = df_train.iloc[fold['test_idx']]

        print(f"Fold {i + 1} - Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # Entrenar el modelo
        main_rsf.train(model, train_df)

        # Obtener predicciones
        predictions = main_rsf.predict2(model, test_df)

        # Calcular la métrica c-index estratificada
        score = main_rsf.get_score(test_df, predictions)
        all_scores.append(score)
        print(f"Fold {i + 1} - c-index: {score:.4f}")

    # Calcular la media de los scores
    mean_score = sum(all_scores) / len(all_scores)
    print(f"Promedio c-index en los folds: {mean_score:.4f}")


if __name__ == "__main__":
    main()
