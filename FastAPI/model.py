import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
#import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("cattle_dataset.csv")

X = data[['body_temperature', 'heart_rate']]
y = data['health_status']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

voting_clf = VotingClassifier(estimators=[('lr', LogisticRegression()),
                                          ('rf', best_rf_model),
                                          ('svm', SVC())],
                               voting='hard')

lr_model = LogisticRegression(penalty='l2')




# Model Evaluation
models = [('Logistic Regression', lr_model),
          ('Random Forest', best_rf_model),
          ('SVM', SVC()),
          ('Voting Classifier', voting_clf)]
for name, model in models:
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} Cross-Validation Scores: {scores}")
    print(f"{name} Mean Accuracy: {scores.mean()}")

# Model Selection: Voting Classifier
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

# Error Analysis
print("Voting Classifier Classification Report:")
print(classification_report(y_test, y_pred_voting))

# Making predictions using Voting Classifier
example_data = [[39.0, 50]]  # Example data for prediction
voting_prediction = voting_clf.predict(example_data)

print("Voting Classifier Prediction:", le.inverse_transform(voting_prediction))


# Save the model to a file
joblib.dump(voting_clf, 'voting_clf_model.pkl')