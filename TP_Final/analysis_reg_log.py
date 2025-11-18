from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


train_set = pd.read_parquet('/Users/mcargnel/Documents/mea/Big-Data-UBA-Grupo-001/TP_Final/data/train_set.parquet')
    
X_train = train_set.drop('Target', axis=1)
y_train = train_set['Target']

X_train = X_train.drop(['merchant_city'], axis=1)

X_train = pd.get_dummies(X_train, columns=['use_chip','errors', 'merchant_state','card_brand','card_type'], drop_first=True, dtype=int)

pipeline_sgd = make_pipeline(
    StandardScaler(),
    SGDClassifier(
        loss='log_loss',
        penalty='l1',
        alpha=0.0001,
        class_weight='balanced',
        n_jobs=-1,
        early_stopping=True,
        random_state=444
    )
)

# Fit Option A
print("Starting training...")
pipeline_sgd.fit(X_train, y_train)

joblib.dump(pipeline_sgd, 'fraud_model_optimized.pkl')
print("Regression model saved as 'fraud_model_optimized.pkl'")