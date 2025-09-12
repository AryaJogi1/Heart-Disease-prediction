import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


# 1) Data Load & Preprocess

df = pd.read_csv(r"C:/Users/piyus/python/app.py/heart (1).csv")

print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().values.any())

# Handle missing values
df['age'].fillna(df['age'].mean(), inplace=True)         # numeric
df['sex'].fillna(df['sex'].mode()[0], inplace=True)      # categorical
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)



# EDA: Pie Chart for Target Distribution

target_counts = df["target"].value_counts()

plt.figure(figsize=(5,5))
plt.pie(
    target_counts, 
    labels=target_counts.index.map({0:"No Disease", 1:"Disease"}), 
    autopct="%1.1f%%", 
    startangle=90, 
    colors=["#66b3ff","#ff9999"]
)
plt.title("Heart Disease Distribution")
plt.show()

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]


# 2) Split and Scale

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3) Models

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, 
                    use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
svc = SVC(C=1.0, probability=True, random_state=42)

# Fit
for m in [lr, rf, xgb, gb, svc]:
    m.fit(X_train_scaled, y_train)




# Dictionary of models
models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb,
    "Gradient Boosting": gb,
    "SVC": svc
}

results = []

for name, model in models.items():
    # Cross-validation (5 folds)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])
    
    # Append results
    results.append({
        "Model": name,
        "CV Mean Accuracy": round(mean_cv, 4),
        "CV Std": round(std_cv, 4),
        "Test Accuracy": round(acc, 4),
        "ROC AUC": round(roc, 4)
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="ROC AUC", ascending=False))

# 4) Evaluation Function

def eval_model(name, model):
    p = model.predict(X_test_scaled)
    prob = model.predict_proba(X_test_scaled)[:,1]
    print(f"\n{name} | Acc: {accuracy_score(y_test,p):.4f} | AUC: {roc_auc_score(y_test,prob):.4f}")
    print(classification_report(y_test,p))

# Evaluate all models
for name, model in [("LogReg",lr),("RandomForest",rf),("XGBoost",xgb),("GradBoost",gb),("SVC",svc)]:
    eval_model(name, model)


# 5) Voting Ensemble

voting = VotingClassifier(estimators=[('xgb',xgb),('rf',rf),('gb',gb)], voting='soft', n_jobs=-1)
voting.fit(X_train_scaled, y_train)

# Predictions
y_pred = voting.predict(X_test_scaled)
y_proba = voting.predict_proba(X_test_scaled)[:,1]

# Metrics
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"\nVotingEnsemble - Accuracy: {acc:.4f}, ROC AUC: {roc:.4f}")
print(classification_report(y_test, y_pred))


# 6) Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=voting.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Voting Ensemble")
plt.show()


# 7) ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'VotingEnsemble (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.title("ROC Curve - Voting Ensemble")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
