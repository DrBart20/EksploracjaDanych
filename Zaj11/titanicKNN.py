import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset('titanic')

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

df_model = df[features + [target]].copy()

df_model['age'] = df_model['age'].fillna(df_model['age'].median())
df_model.dropna(subset=['embarked'], inplace=True)

df_encoded = pd.get_dummies(df_model, columns=['sex', 'embarked'], drop_first=True)

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dane gotowe. X_train shape: {X_train.shape}")

depths = [2, 3, 5, 7, 10, None]
results_tree = []
best_f1 = 0
best_depth = 0
best_model_tree = None

print("\n--- Wyniki dla Drzewa Decyzyjnego ---")
print(f"{'Depth':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results_tree.append({'Depth': d, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})
    print(f"{str(d):<10} {acc:.4f}     {prec:.4f}     {rec:.4f}     {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_depth = d
        best_model_tree = clf

print(f"\nNajlepsze drzewo wg F1-score ma głębokość: {best_depth}")

plt.figure(figsize=(20, 10))
plot_tree(best_model_tree, 
          feature_names=X.columns, 
          class_names=['Died', 'Survived'], 
          filled=True, 
          fontsize=10,
          max_depth=3)
plt.title(f'Drzewo Decyzyjne (max_depth={best_depth})')
plt.show()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []

print("\n--- Wyniki dla KNN ---")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    acc = knn.score(X_test_scaled, y_test)
    accuracies.append(acc)
    print(f"k={k}: Accuracy = {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='purple')
plt.title('Dokładność modelu KNN w zależności od liczby sąsiadów (k)')
plt.xlabel('Liczba sąsiadów (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(k_values)
plt.show()

# Wskazanie optymalnego k
best_k = k_values[np.argmax(accuracies)]
print(f"Optymalne k: {best_k}")