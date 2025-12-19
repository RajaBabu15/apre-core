
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_all_supervised_models(X, y_encoded, accel, config):
    print("Training supervised models...")
    
    # Split data
    test_size = config.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=42, 
        stratify=y_encoded
    )
    
    results = {}
    
    # MLP
    print("Training MLP Classifier...")
    mlp_layers = config.get('mlp_layers', (256, 128, 64))
    mlp_max_iter = config.get('mlp_max_iter', 500)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=mlp_layers,
        max_iter=mlp_max_iter,
        random_state=42,
        early_stopping=True
    )
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    
    results['MLP'] = {
        'Accuracy': accuracy_score(y_test, y_pred_mlp),
        'Precision': precision_score(y_test, y_pred_mlp, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_mlp, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred_mlp, average='weighted', zero_division=0)
    }
    
    # XGBoost
    print("Training XGBoost Classifier...")
    n_estimators = config.get('xgb_n_estimators', 100)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    results['XGBoost'] = {
        'Accuracy': accuracy_score(y_test, y_pred_xgb),
        'Precision': precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_xgb, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0),
        'Feature_Importances': xgb_model.feature_importances_
    }
    
    return results
