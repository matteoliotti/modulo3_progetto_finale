import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

diabetes=load_diabetes()

# creazione di un DataFrame con le feature con l'aggiunta di "target"
X=pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y=pd.Series(diabetes.target, name="target")

df=X.copy()
df["target"]=y

# analisi 
print("---Dimensione dataset---")
print(f"Campioni: {df.shape[0]}\nFeatures: {df.shape[1]-1}")

print("\n---Analisi descrittiva---")
print(df.describe())

print("\n---Valori nulli---")
print(df.isnull().sum())

import matplotlib.pyplot as plt

fig,axes=plt.subplots(3,4, figsize=(16,12))
fig.suptitle("Istogrammi delle Features", fontsize=16)

for i, col in enumerate(X.columns):
    ax=axes[i//4][i%4]
    ax.hist(X[col], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_title(col)

axes[2][2].set_visible(False)
axes[2][3].set_visible(False) #nascondo i riquardi vuoti
plt.tight_layout()
plt.show()

fig,axes=plt.subplots(3,4, figsize=(16,12))
fig.suptitle("Boxplot delle Features", fontsize=16)

for i, col in enumerate(X.columns):
    ax=axes[i//4][i%4]
    ax.boxplot(X[col], patch_artist=True, boxprops=dict(facecolor="lightblue"))
    ax.set_title(col)

axes[2][2].set_visible(False)
axes[2][3].set_visible(False)
plt.tight_layout()
plt.show()

import seaborn as sns

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center =0)
plt.title("Mappa di correlazione")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Training e test
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")

# Standardizzazione
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)