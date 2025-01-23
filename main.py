from feature_selection import data_preprocess, top_15_features, future_generation, future_selection, data_minimize
from blend_of_models import AGfit, LAMAfit
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

def main(data, target):
    data = data_preprocess(data)
    X = data.drop(columns=target)
    y = data[target]

    features = top_15_features(X, y)
    train_data, test_data = train_test_split(data, test_size=0.12, stratify=data[target], random_state=42)
    train_data, test_data = future_generation(train_data, test_data, features)

    selected_features = future_selection(train_data, target)
    train_data = data_minimize(train_data, target)

    predictor = AGfit(train_data, selected_features, target)
    automl = LAMAfit(train_data, target)

    autogluon_predictions = predictor.predict_proba(test_data)
    autogluon_probs = autogluon_predictions[1]  
    autogluon_probs = autogluon_probs.to_numpy().flatten()  

    lightautoml_predictions = automl.predict(test_data)
    lightautoml_probs = lightautoml_predictions.data  
    lightautoml_probs = lightautoml_probs.flatten()  

    # Инициализация переменных для хранения наилучших весов и ROC AUC
    best_roc_auc = 0
    best_weights = (0, 0)

    # Перебор весов с шагом 0.05
    for weight_ag in np.arange(0, 1.05, 0.05):  # Вес для AutoGluon
        weight_lama = 1 - weight_ag  # Вес для LightAutoML (сумма весов должна быть равна 1)
        
        # Блендинг предсказаний
        blended_predictions = weight_ag * autogluon_probs + weight_lama * lightautoml_probs
        
        # Оценка качества бленда
        roc_auc = roc_auc_score(test_data[target], blended_predictions)
        
        # Если текущий ROC AUC лучше, обновляем наилучшие веса и ROC AUC
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_weights = (weight_ag, weight_lama)

    # Вывод наилучших весов и ROC AUC
    print(f"Наилучшие веса: AutoGluon = {best_weights[0]:.2f}, LightAutoML = {best_weights[1]:.2f}")
    print(f"Наилучший ROC AUC: {best_roc_auc:.4f}")


data = pd.read_parquet('data.parquet')
target = 'target'
main(data, target)