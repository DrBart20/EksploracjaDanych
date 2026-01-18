import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

#Czesc 1
print(f"{60*"="}Czesc 1{30*"="}")
df = sns.load_dataset('titanic')
print(f"Wymiary zbioru: {df.shape}")
print(df.info())
print(df.describe())
print("\nBrakujące wartości w kolumnach:")
print(df.isnull().sum())
print("\nPrzeżywalność ogółem:", df['survived'].mean())
print(df.groupby(['sex', 'pclass'])['survived'].mean())
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='pclass', y='survived', hue='sex')
plt.title('Przeżywalność w zależności od klasy i płci')
plt.show()

#Czesc 2
print(f"{60*"="}Czesc 1{30*"="}")
cols_to_drop = ['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class']
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

df_clean = df_clean.dropna(subset=['embarked'])

df_encoded = pd.get_dummies(df_clean, columns=['sex', 'embarked'], drop_first=True)

X = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']

print(f"{60*"="}Czesc 3{30*"="}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

coefs = pd.DataFrame({
    'Cecha': X.columns,
    'Współczynnik (Beta)': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0]) 
})

print(coefs.sort_values(by='Odds Ratio', ascending=False))


#Czesc 4
print(f"{60*"="}Czesc 4{30*"="}")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"Specificity: {specificity:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Krzywa ROC')
plt.legend()
plt.show()


#Czesc 5
print(f"{60*"="}Czesc 5{30*"="}")
print("\n--- Zmiana progu ---")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_new = (y_pred_proba >= threshold).astype(int)
    rec = recall_score(y_test, y_pred_new)
    prec = precision_score(y_test, y_pred_new)
    print(f"Próg: {threshold} -> Recall: {rec:.2f}, Precision: {prec:.2f}")

#Czesc 6
print(f"{60*"="}Czesc 6{30*"="}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

c_values = [0.001, 0.01, 0.1, 1, 10, 100]
auc_scores = []
coef_sums = []

for c in c_values:
    m = LogisticRegression(C=c, penalty='l2', max_iter=1000)
    m.fit(X_train_scaled, y_train)
    probs = m.predict_proba(X_test_scaled)[:, 1]
    auc_scores.append(roc_auc_score(y_test, probs))
    coef_sums.append(np.sum(np.abs(m.coef_)))

fig, ax1 = plt.subplots()

ax1.set_xlabel('Wartość C (skala log)')
ax1.set_ylabel('AUC', color='blue')
ax1.plot(c_values, auc_scores, color='blue', marker='o')
ax1.set_xscale('log')

ax2 = ax1.twinx()
ax2.set_ylabel('Suma |Beta|', color='red')
ax2.plot(c_values, coef_sums, color='red', marker='x')

plt.title('Wpływ parametru C na model')
plt.show()

print("\n--- Porównanie L1 vs L2 (C=0.1) ---")
model_l2 = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs')
model_l2.fit(X_train_scaled, y_train)
model_l1 = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
model_l1.fit(X_train_scaled, y_train)

print("Współczynniki L2 (Ridge):", model_l2.coef_)
print("Współczynniki L1 (Lasso):", model_l1.coef_)
