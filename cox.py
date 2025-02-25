from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]

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


def get_score(x, prediction):
    prediction = pd.DataFrame({
        'ID': x['ID'],
        'prediction': prediction
    })

    return score(x.copy(), prediction, 'ID')


def clean_column_headers(df):
    replacements = {
        '(': '_', ')': '_', '+': 'plus', ',': '_',
        '-': '_', '.': '_', '/': '_',
        '<': 'less_than', '=': 'equals', '>': 'greater_than'
    }
    df.columns = df.columns.str.translate(str.maketrans(replacements))
    return df


def load_dataframes(train_path='train.csv', test_path='test.csv'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = clean_column_headers(train_df)
    test_df = clean_column_headers(test_df)

    special_columns = ['ID', 'race_group', 'efs', 'efs_time']
    procesable_columns = [col for col in train_df.columns if col not in special_columns]

    train_df['origin'] = 'train'
    test_df['origin'] = 'test'

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    for col in procesable_columns:
        if combined_df[col].dtype == 'object':
            one_hot = pd.get_dummies(combined_df[col], prefix=col, dummy_na=True, drop_first=False)
            combined_df = pd.concat([combined_df, one_hot], axis=1)
            combined_df.drop(columns=[col], inplace=True)
        elif pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())

    final_train = combined_df[combined_df['origin'] == 'train']
    final_test = combined_df[combined_df['origin'] == 'test']

    processed_train_df = final_train[
        special_columns + list(final_train.columns[~final_train.columns.isin(special_columns + ['origin'])])]
    processed_test_df = final_test[
        special_columns + list(final_test.columns[~final_test.columns.isin(special_columns + ['origin'])])]

    return processed_train_df, processed_test_df


def main():
    # Cargar y preprocesar los datos
    df_train, df_test = load_dataframes('train.csv', 'test.csv')

    # Preparar datos de entrenamiento
    X_train = df_train.drop(['ID', 'efs', 'efs_time', 'race_group'], axis=1)
    y_train = np.array([(bool(e), t) for e, t in zip(df_train["efs"], df_train["efs_time"])],
                       dtype=[("event", "bool"), ("time", "float")])

    # Preparar datos de prueba
    X_test = df_test.drop(['ID', 'race_group'], axis=1)
    if 'efs' in X_test.columns and 'efs_time' in X_test.columns:
        X_test = X_test.drop(['efs', 'efs_time'], axis=1)

    # Implementación de validación cruzada K-Fold
    from sklearn.model_selection import KFold
    from sksurv.metrics import concordance_index_censored
    import matplotlib.pyplot as plt

    # Probar diferentes valores de alpha para regularización
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    k_folds = 5

    # Diccionario para almacenar resultados
    cv_results = {alpha: [] for alpha in alphas}
    cv_race_results = {alpha: {race: [] for race in df_train['race_group'].unique()} for alpha in alphas}

    # KFold para validación cruzada
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for alpha in alphas:
        fold_scores = []
        print(f"\nEvaluando alpha={alpha}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # Dividir datos para este fold
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Crear y entrenar pipeline
            fold_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('cox', CoxPHSurvivalAnalysis(alpha=alpha, ties='breslow'))
            ])
            fold_pipeline.fit(X_fold_train, y_fold_train)

            # Predecir risk scores para validación
            val_risk_scores = fold_pipeline.predict(X_fold_val)

            # Preparar DataFrame para evaluación
            val_df = df_train.iloc[val_idx].copy()
            val_pred_df = pd.DataFrame({
                'ID': val_df['ID'],
                'prediction': val_risk_scores
            })

            # Calcular score usando nuestra función personalizada
            fold_score = get_score(val_df.copy(), val_risk_scores)
            fold_scores.append(fold_score)

            # Analizar rendimiento por grupo racial
            for race in df_train['race_group'].unique():
                race_indices = val_df[val_df['race_group'] == race].index
                if len(race_indices) > 0:
                    race_df = val_df.loc[race_indices]
                    race_predictions = val_risk_scores[val_df.index.get_indexer(race_indices)]

                    # Calcular c-index específico para este grupo racial
                    if sum(race_df['efs']) > 0:  # Asegurar que hay eventos positivos
                        c_index = concordance_index_censored(
                            race_df['efs'].astype(bool),
                            race_df['efs_time'],
                            -race_predictions  # Negativo porque risk_score más alto = peor pronóstico
                        )[0]
                        cv_race_results[alpha][race].append(c_index)

            print(f"  Fold {fold + 1}/{k_folds}: Score = {fold_score:.4f}")

        # Calcular promedio de scores para este alpha
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        cv_results[alpha] = fold_scores
        print(f"Alpha={alpha}: Score medio = {mean_score:.4f} ± {std_score:.4f}")

    # Encontrar el mejor alpha
    mean_scores = {alpha: np.mean(scores) for alpha, scores in cv_results.items()}
    best_alpha = max(mean_scores, key=mean_scores.get)
    print(f"\nMejor alpha encontrado: {best_alpha} con score medio: {mean_scores[best_alpha]:.4f}")

    # Análisis de rendimiento por grupo racial con el mejor alpha
    print("\nRendimiento por grupo racial:")
    race_means = {}
    for race in df_train['race_group'].unique():
        if race in cv_race_results[best_alpha] and cv_race_results[best_alpha][race]:
            race_mean = np.mean(cv_race_results[best_alpha][race])
            race_std = np.std(cv_race_results[best_alpha][race])
            race_means[race] = race_mean
            print(f"Grupo {race}: C-index = {race_mean:.4f} ± {race_std:.4f}")

    # Visualizar resultados de la validación cruzada
    plt.figure(figsize=(10, 6))
    plt.boxplot([cv_results[alpha] for alpha in alphas], labels=[str(alpha) for alpha in alphas])
    plt.title('Rendimiento del modelo por valor de alpha')
    plt.xlabel('Valor de alpha')
    plt.ylabel('Score (C-index ponderado)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('alpha_cv_results.png')

    # Visualizar resultados por grupo racial
    plt.figure(figsize=(10, 6))
    race_scores = []
    race_labels = []
    for race in sorted(race_means.keys()):
        race_scores.append(race_means[race])
        race_labels.append(f"Grupo {race}")

    plt.bar(race_labels, race_scores, color='skyblue')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Rendimiento aleatorio')
    plt.title(f'C-index por grupo racial (alpha={best_alpha})')
    plt.ylabel('C-index')
    plt.ylim(0.4, 0.7)  # Ajustar según resultados
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.savefig('race_performance.png')

    # Entrenar modelo final con el mejor alpha
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('cox', CoxPHSurvivalAnalysis(alpha=best_alpha, ties='breslow'))
    ])

    final_pipeline.fit(X_train, y_train)

    # Obtener rendimiento en datos de entrenamiento (solo para referencia)
    train_risk_scores = final_pipeline.predict(X_train)
    train_score = get_score(df_train.copy(), train_risk_scores)
    print(f"\nC-index en datos de entrenamiento (modelo final): {train_score:.4f}")

    # Predecir risk_score para datos de prueba
    test_risk_scores = final_pipeline.predict(X_test)

    # Preparar submission
    submission = pd.DataFrame({
        'ID': df_test['ID'],
        'prediction': test_risk_scores
    })

    # Guardar submission
    submission.to_csv('submission.csv', index=False)
    print("Archivo de submission creado con éxito.")

    # Análisis de coeficientes
    coefficients = pd.Series(
        final_pipeline.named_steps['cox'].coef_,
        index=X_train.columns
    )

    # Mostrar los factores de riesgo más importantes
    top_coeffs = coefficients.abs().sort_values(ascending=False).head(15)
    print("\nFactores de riesgo más importantes:")
    for feature, coef in zip(top_coeffs.index, coefficients[top_coeffs.index]):
        print(f"{feature}: {coef:.4f} {'↑' if coef > 0 else '↓'} riesgo")

    # Visualizar coeficientes más importantes
    plt.figure(figsize=(12, 8))
    features = top_coeffs.index
    coefs = coefficients[features]
    colors = ['r' if c > 0 else 'g' for c in coefs]
    plt.barh(features, coefs, color=colors)
    plt.title('Top 15 Coeficientes del Modelo Cox')
    plt.xlabel('Coeficiente (>0 aumenta riesgo, <0 reduce riesgo)')
    plt.tight_layout()
    plt.savefig('top_features.png')

    return submission, best_alpha


if __name__ == "__main__":
    main()