from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import main_rsf  # Importar módulo externo
import numpy as np

def get_fair_kfolds(df_train, n_splits=5):
    df_copy = df_train.copy()
    # Estratificar por efs Y race_group
    df_copy['strata'] = df_copy['efs'].astype(str) + '_' + df_copy['race_group'].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in skf.split(df_copy, df_copy['strata']):
        folds.append({'train_idx': train_idx, 'test_idx': test_idx})
    return folds

def get_balanced_weights(train_df):
    race_counts = train_df['race_group'].value_counts()
    # Raíz cuadrada para suavizar el efecto
    weights = {race: np.sqrt(len(train_df) / (len(race_counts) * count))
               for race, count in race_counts.items()}
    return train_df['race_group'].map(weights).values

def main():
    # Obtener los DataFrames desde el módulo externo
    df_train, df_submit = main_rsf.get_dfs()

    #df_train = main_rsf.get_relevant_features(df_train,0.005)

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

        weights = get_balanced_weights(train_df)
        print(f"Fold {i + 1} - Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # Entrenar el modelo
        main_rsf.train(model, train_df, weights)

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
