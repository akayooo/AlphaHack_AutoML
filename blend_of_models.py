from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task 
from sklearn.metrics import roc_auc_score


def AGfit(train_data, selected_features, target):
    #Разбиение тестовых данных для валидации
    train_data, test_data = train_test_split(train_data[selected_features], test_size=0.12, stratify=train_data[target], random_state=42)
    
    # Обучение модели с AutoGluon, оптимизация по ROC AUC
    predictor = TabularPredictor(
        label=target, # Целевая переменная
        problem_type='binary',  # Задача: бинарная классификация
        eval_metric='roc_auc'   # Метрика для оптимизации
    ).fit(
        train_data=train_data,
        time_limit=3000,        
        presets='best_quality', 
        num_stack_levels=2,
        num_gpus=1,
        hyperparameters={
            'GBM': {
                'num_boost_round': 500,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
            },
            'CAT': {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'grow_policy': 'SymmetricTree',
            },
            'XGB': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
            },
            'XT': {
                'n_estimators': 300,
                'max_depth': 30,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'criterion': 'gini',
                'random_state': 42
            }
        }
    )

    # Результаты обучения
    print(predictor.leaderboard())
    
    # Оценка модели на тестовом наборе
    predictions = predictor.predict(test_data)
    performance = predictor.evaluate(test_data)

    autogluon_predictions = predictor.predict_proba(test_data)
    autogluon_probs = autogluon_predictions[1]  # Вероятности для класса '1'
    autogluon_probs = autogluon_probs.to_numpy().flatten()  # Если это DataFrame
    
    print("Производительность AG:\n", performance)

    return predictor


def LAMAfit(train_data, target):
    #Разбиение тестовых данных для валидации
    train_data, test_data = train_test_split(train_data, test_size=0.12, stratify=train_data[target], random_state=42)
    roles = {'target': target}
    task = Task('binary', metric='auc')
    
    automl = TabularAutoML(
        task=task,
        timeout=3000,
        general_params={
            'use_algos': [['lgb_tuned', 'xgb_tuned', 'catboost']]
        },
        reader_params={'n_jobs': 8}
    )
    oof_pred = automl.fit_predict(train_data, roles=roles)

    lightautoml_predictions = automl.predict(test_data)
    lightautoml_probs = lightautoml_predictions.data  # Вероятности для класса '1'
    lightautoml_probs = lightautoml_probs.flatten()  # Если это Numpy массив
    roc_auc = roc_auc_score(test_data[target], lightautoml_probs)  
    
    # Выводим производительность
    print(f"Производительность LAMA: {roc_auc:.4f}")

    return automl
