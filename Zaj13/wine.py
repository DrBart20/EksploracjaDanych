import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

#Czesc 1
data = load_wine()
X = data.data
y = data.target

print("--- Podstawowe informacje ---")
print(f"Liczba próbek: {X.shape[0]}")
print(f"Liczba cech: {X.shape[1]}")
print("\n--- Nazwy cech ---")
print(data.feature_names)

print("\n--- Rozkład klas ---")
df_y = pd.DataFrame(y, columns=['Klasa'])
df_y['Nazwa_Klasy'] = df_y['Klasa'].map(dict(enumerate(data.target_names)))
print(df_y['Nazwa_Klasy'].value_counts())

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

print("\n--- Standaryzacja zakończona ---")
print(f"Średnia po standaryzacji (dla pierwszej cechy): {np.mean(X_std[:, 0]):.2f}")
print(f"Odchylenie po standaryzacji (dla pierwszej cechy): {np.std(X_std[:, 0]):.2f}")

#Czesc 2
pca = PCA()
pca.fit(X_std)

eigenvalues = pca.explained_variance_

explained_variance_ratio = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance_ratio)

results_df = pd.DataFrame({
    'Składowa': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Wartość Własna': eigenvalues,
    '% Wariancji': explained_variance_ratio * 100,
    '% Skumulowany': cumulative_variance * 100
})

print("--- Wyniki PCA (Wartości własne i Wariancja) ---")
print(results_df.round(4).to_string(index=False))

n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1

print(f"\n--- Wnioski ---")
print(f"Liczba składowych potrzebna do >= 80% wariancji: {n_components_80}")
print(f"Dokładna skumulowana wariancja dla {n_components_80} składowych: {cumulative_variance[n_components_80-1]*100:.2f}%")

X_pca = pca.transform(X_std)

plt.figure(figsize=(18, 5))

#Czesc 3
plt.subplot(1, 3, 1)
plt.plot(range(1, len(eigenvalues) + 1), explained_variance_ratio * 100, 'bo-', markersize=8)
plt.title('Scree Plot (Wykres osypiska)')
plt.xlabel('Numer Składowej')
plt.ylabel('% Wyjaśnionej Wariancji')
plt.grid(True)
plt.axvline(x=3, color='r', linestyle='--', label='Punkt łokcia (hipotetyczny)')
plt.legend()

plt.subplot(1, 3, 2)
colors = ['red', 'green', 'blue']
target_names = data.target_names

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                color=color, alpha=.8, lw=2, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA zbiór Wine (PC1 vs PC2)')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
plt.grid(True)
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, c=y, cmap='viridis')

coeff = np.transpose(pca.components_[0:2, :])
n = coeff.shape[0]
scale_factor = 3.5 

for i in range(n):
    plt.arrow(0, 0, coeff[i,0]*scale_factor, coeff[i,1]*scale_factor, 
              color='r', alpha=0.9, head_width=0.15)
    plt.text(coeff[i,0]*scale_factor*1.15, coeff[i,1]*scale_factor*1.15, 
             data.feature_names[i], color='black', ha='center', va='center')

plt.title('Biplot PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()

plt.tight_layout()
plt.show()

print("\n--- CZĘŚĆ 4---")

loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(X[0]))], index=data.feature_names)
print("Ładunki (korelacja cechy ze składową) dla PC1 i PC2:")
print(loadings_df[['PC1', 'PC2']].sort_values(by='PC1', ascending=False))

top_pc1 = loadings_df['PC1'].abs().sort_values(ascending=False).head(3).index.tolist()
top_pc2 = loadings_df['PC2'].abs().sort_values(ascending=False).head(3).index.tolist()

print(f"\n[ODPOWIEDŹ] Cechy dominujące w PC1 (wg modułu wartości): {top_pc1}")
print("PC1 w zbiorze Wine zazwyczaj reprezentuje 'złożoność chemiczną' (Flavanoids, Phenols).")

print(f"[ODPOWIEDŹ] Cechy dominujące w PC2 (wg modułu wartości): {top_pc2}")
print("PC2 często oddaje intensywność koloru (Color intensity) i zawartość popiołu (Ash).")

print("[ODPOWIEDŹ] Separacja klas: Patrząc na wykres 3.2, klasy są bardzo dobrze odseparowane,")
print("szczególnie klasa 0 od klasy 2. Klasa 1 znajduje się pośrodku.")

#  PCA bez standaryzacji
pca_raw = PCA(n_components=2)
X_pca_raw = pca_raw.fit_transform(X)

print(f"Wyjaśniona wariancja (BEZ standaryzacji): {pca_raw.explained_variance_ratio_}")
print(f"Suma wariancji (2 składowe): {sum(pca_raw.explained_variance_ratio_)*100:.2f}%")

# Wizualizacja PCA bez standaryzacji
plt.figure(figsize=(6, 5))
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca_raw[y == i, 0], X_pca_raw[y == i, 1], 
                color=color, alpha=.8, lw=2, label=target_name)
plt.title('PCA BEZ standaryzacji')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

raw_loadings = pd.Series(pca_raw.components_[0], index=data.feature_names)
dominating_feature = raw_loadings.abs().idxmax()
print(f"\n[ODPOWIEDŹ] Cecha dominująca analizę bez standaryzacji: {dominating_feature}")
print("Dzieje się tak, ponieważ 'proline' ma wartości rzędu 1000, podczas gdy inne cechy < 10.")
print("PCA 'widzi' tylko największą wariancję liczbową, ignorując korelacje.")

print("\n--- CZĘŚĆ 5: PCA k-NN ---")

def benchmark_knn(X_data, y_data, n_features_label):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    clf = KNeighborsClassifier(n_neighbors=3)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return acc, training_time

# Przypadek A: Oryginalne dane 
acc_orig, time_orig = benchmark_knn(X_std, y, "Wszystkie cechy (13)")

# Przypadek B: PCA 2 składowe
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_std)
acc_pca2, time_pca2 = benchmark_knn(X_pca_2, y, "PCA (2 składowe)")

# Przypadek C: PCA 5 składowych
pca_5 = PCA(n_components=5)
X_pca_5 = pca_5.fit_transform(X_std)
acc_pca5, time_pca5 = benchmark_knn(X_pca_5, y, "PCA (5 składowych)")

# === 5.3. Porównanie wyników ===
print(f"{'Zbiór danych':<25} | {'Dokładność':<10} | {'Czas (s)':<10}")
print("-" * 50)
print(f"{'Oryginał (13 cech)':<25} | {acc_orig:.4f}     | {time_orig:.6f}")
print(f"{'PCA (2 składowe)':<25} | {acc_pca2:.4f}     | {time_pca2:.6f}")
print(f"{'PCA (5 składowych)':<25} | {acc_pca5:.4f}     | {time_pca5:.6f}")

print("\n[WNIOSKI]")
print("PCA (2) może mieć niższą dokładność, bo tracimy ~45% informacji.")
print("PCA (5) powinno mieć dokładność zbliżoną do oryginału przy mniejszym wymiarze.")