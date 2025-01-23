from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.utils import resample
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def data_preprocess(df):
    df.select_dtypes(exclude=['object', 'category']).dropna()
    df = df.select_dtypes(include=['float', 'int']).dropna()
    df = df.loc[:, df.nunique() > 1]    
    return df

def top_15_features(X, y):
    # Разделение данных на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Создание датасета для LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Параметры модели
    params = {
        'objective': 'binary',  # Для бинарной классификации
        'metric': 'auc',        # Метрика ROC-AUC
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': 0
    }
    
    # Обучение модели
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Предсказание на тестовом наборе
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Оценка модели
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'ROC-AUC Selection Model Score: {roc_auc}')
    
    # Получение важности признаков
    feature_importance = model.feature_importance(importance_type='gain')
    feature_names = X.columns
    
    # Создание DataFrame с важностью признаков
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    
    # Сортировка по убыванию важности
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Вывод топ 15 признаков
    top_15_features = importance_df.head(15)
    
    # Возвращение массива с названиями признаков
    top_15_feature_names = top_15_features['Feature'].tolist()
    print("Лучшие признаки были одобраны") 

    return top_15_feature_names

def future_generation(train_data, test_data, features):
    new_features_train = {}
    new_features_test = {}
    
    for i, feature1 in enumerate(features):  # features отобраны методом top_15_features
        for j, feature2 in enumerate(features):
            if i < j:
                # Умножение
                new_feature_name_mul = f'{feature1}_{feature2}_mul'
                new_features_train[new_feature_name_mul] = train_data[feature1] * train_data[feature2]
                new_features_test[new_feature_name_mul] = test_data[feature1] * test_data[feature2]
                
                # Деление
                new_feature_name_div = f'{feature1}_{feature2}_div'
                new_features_train[new_feature_name_div] = train_data[feature1] / (train_data[feature2] + 1e-6)
                new_features_test[new_feature_name_div] = test_data[feature1] / (test_data[feature2] + 1e-6)
        
                # Синус
                new_feature_name_sin = f'{feature1}_sin'
                new_features_train[new_feature_name_sin] = np.sin(train_data[feature1])
                new_features_test[new_feature_name_sin] = np.sin(test_data[feature1])
                
                # Косинус
                new_feature_name_cos = f'{feature1}_cos'
                new_features_train[new_feature_name_cos] = np.cos(train_data[feature1])
                new_features_test[new_feature_name_cos] = np.cos(test_data[feature1])

    # Объединяем новые признаки с исходными DataFrame
    train_data = pd.concat([train_data, pd.DataFrame(new_features_train)], axis=1)
    test_data = pd.concat([test_data, pd.DataFrame(new_features_test)], axis=1)

    print('Новые признаки созданы')
    return train_data, test_data

def future_selection(train_data, target):
    precent_10 = int(train_data.shape[0] * 0.1)
    #Стратифицированно обрезаем - экономим время
    train_data = resample(train_data, 
                              replace=False,  
                              n_samples=precent_10,
                              stratify=train_data[target], 
                              random_state=42)
    # Разбиваем набор
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    
    # Настройка RandomForest для Boruta
    rf_model = RandomForestClassifier(
        n_estimators=75,    
        max_depth=5,         
        random_state=42,
        n_jobs=-1              
    )
    
    # Boruta для отбора признаков
    boruta_selector = BorutaPy(
        rf_model,
        n_estimators=50,
        random_state=42,
        alpha=0.1,
        max_iter=3 # Максимальное количество итераций
    )
    
    # Обучение Boruta
    boruta_selector.fit(X_train.values, y_train.values)
    
    # Разделение признаков по областям
    green_area = X_train.columns[boruta_selector.support_].tolist()
    blue_area = X_train.columns[boruta_selector.support_weak_].tolist()
    
    selected_features = green_area + blue_area
    selected_features.append(target)
    return selected_features

def data_minimize(train_data, target):
    
    # Разделяем данные на два класса
    df_majority = train_data[train_data[target] == 0]
    df_minority = train_data[train_data[target] == 1]
    
    # Проверяем сбалансированность классов
    majority_ratio = len(df_majority) / len(train_data)

    # Устанавливаем количество sample
    if train_data.shape[0] > 600000:
        precent_of_oversize = int((train_data.shape[0] - 600000) / train_data.shape[0])
        n_samples = int(train_data.shape[0] * (1 - precent_of_oversize))
    else:
        n_samples = df_majority.shape[0]
    
    if majority_ratio > 0.85:
        # Уменьшаем размер доминирующего класса
        df_majority_downsampled = resample(df_majority, 
                                           replace=False,   
                                           n_samples=n_samples,  
                                           random_state=42)
        
        # Объединяем обратно
        train_data = pd.concat([df_majority_downsampled, df_minority])
    else:
        # Делаем стратифицированный ресемпл 
        train_data = resample(train_data, 
                              replace=False,  
                              n_samples=n_samples, 
                              stratify=train_data[target], 
                              random_state=42)
    
    return train_data

