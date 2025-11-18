import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

def load_data(train_path: str, test_path: str, sample_frac):
    print("\n--- 1. Iniciando Carga de Datos ---")
    
    print(f"Cargando datos de entrenamiento desde: {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"Shape original de entrenamiento: {train_df.shape}")

    print(f"Cargando datos de prueba desde: {test_path}")
    test_df = pd.read_parquet(test_path)
    print(f"Shape original de prueba: {test_df.shape}")
    
    X_test = test_df.drop('Target', axis=1)
    y_test = test_df['Target']

    print(f"Tomando muestra de entrenamiento con frac={sample_frac}")

    train_sample = train_df.iloc[:int(len(train_df)*sample_frac)]

    X = train_sample.drop('Target', axis=1)
    y = train_sample['Target']
    print(f"Shape final de entrenamiento (X, y): {X.shape}, {y.shape}")

    categorical_features = ['use_chip', 'errors', 'card_brand', 'card_type','merchant_state', 'merchant_city']
    print(f"Características categóricas identificadas: {categorical_features}")

    full_train_pool = Pool(
        data=X,
        label=y,
        cat_features=categorical_features
    )

    y_count = y.value_counts()
    print("Distribución de la variable objetivo (Target) en el set de entrenamiento:")
    print(y_count)
    
    scale = y_count[0] / y_count[1]
    print(f"Scale_pos_weight calculado: {scale:.2f}")
    
    print("--- Carga de Datos Completa ---")
    return full_train_pool, X_test, y_test, scale, X, y
def model_training(full_train_pool, metric, scale, save_model=False):
    print("\n--- 2. Iniciando Entrenamiento del Modelo ---")
    print(f"Métrica de evaluación principal: {metric}")

    model_params = {
        'iterations': 5000,
        'learning_rate': 0.01,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': metric, 
        'scale_pos_weight': scale,     
        'random_seed': 42,
        'logging_level': 'Verbose', 
        'early_stopping_rounds': 100
    }
    
    print("Iniciando Cross-Validation (5-folds)...")
    
    cv_results = cv(
        pool=full_train_pool,
        params=model_params,
        verbose=50,
        fold_count=5,
        shuffle=False,
        stratified=True,
        plot=False       
    )

    best_iteration = cv_results[f'test-{metric}-mean'].idxmax()
    print(f"\nMejor iteración de CV (basada en {metric}): {best_iteration}")
    print(f"Mejor score de CV ({metric}): {cv_results[f'test-{metric}-mean'].max():.4f}")

    print("\nEntrenando modelo final en todos los datos de entrenamiento...")

    final_model_params = model_params.copy()
    
    final_model_params['iterations'] = best_iteration + 1 
    del final_model_params['early_stopping_rounds']
    del final_model_params['logging_level']

    final_model = CatBoostClassifier(**final_model_params)
    final_model.fit(full_train_pool, verbose=100)
    print("Entrenamiento final completado.")

    if save_model:
        
        print("Guardando el modelo en 'fraud_model.cbm'...")
        final_model.save_model("fraud_model.cbm")
        print("Modelo guardado exitosamente!")

    print("--- Entrenamiento del Modelo Completo ---")
    return final_model

def evaluate_model(final_model, X_test, y_test, X_train, y_train):
    
    print("\n--- 3. Iniciando Evaluación del Modelo ---")
    print("Evaluando el modelo final en el set de prueba...")

    print("\nFeature Importance (Top 10):")
    df_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': final_model.get_feature_importance()
    }).sort_values(by='Importance', ascending=False)

    print(df_importance.head(10))

    preds_proba = final_model.predict_proba(X_test)[:, 1]
    preds = final_model.predict(X_test)
    
    print("\nConfusion matrix (Train set):")
    preds_train = final_model.predict(X_train)
    print(pd.crosstab(index=y_train, columns=preds_train).rename(columns={0:'Pred: No Fraude', 1:'Pred: Fraude'}))
    
    
    print("\nMatriz de Confusión (Test Set):")
    print(pd.crosstab(index=y_test, columns=preds).rename(columns={0:'Pred: No Fraude', 1:'Pred: Fraude'}))

    print("\nMétricas de Performance (Test Set):")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Recall : {recall_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"F1 Score : {f1_score(y_test, preds):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, preds_proba):.4f}")
    print(f"PRAUC Score: {average_precision_score(y_test, preds_proba):.4f}")
    
    print("--- Evaluación del Modelo Completa ---")

if __name__ == "__main__":
    print("=== INICIO DEL SCRIPT DE DETECCIÓN DE FRAUDE ===")
    
    train_path = 'data/train_set.parquet'
    test_path = 'data/test_set.parquet'
    metric = 'PRAUC' 
    sample_frac = 1
    save_model = True
    
    print(f"Parámetros de ejecución:")
    print(f"  - Train path: {train_path}")
    print(f"  - Test path: {test_path}")
    print(f"  - Métrica: {metric}")
    print(f"  - Fracción de muestra: {sample_frac}")
    print(f"  - Guardar modelo: {save_model}")


    full_train_pool, X_test, y_test, scale, X_train, y_train = load_data(train_path, test_path, sample_frac=sample_frac)
    final_model = model_training(full_train_pool,metric, scale, save_model=save_model)
    evaluate_model(final_model, X_test, y_test, X_train, y_train)
    
    print("\n=== FIN DEL SCRIPT ===")