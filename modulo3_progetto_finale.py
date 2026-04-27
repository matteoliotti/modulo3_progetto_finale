import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

diabetes=load_diabetes()

# creazione di un DataFrame con le feature con l'aggiunta di "target"

X=pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y=pd.Series(diabetes.target, name="target")

df=X.copy()
df["target"]=y

# analisi 
#print("---Dimensione dataset---")
#print(f"Campioni: {df.shape[0]}\nFeatures: {df.shape[1]-1}")

#print("\n---Analisi descrittiva---")
#print(df.describe())

#print("\n---Valori nulli---")
#print(df.isnull().sum())


fig,axes=plt.subplots(3,4, figsize=(16,12))
fig.suptitle("Istogrammi delle Features", fontsize=16)

for i, col in enumerate(X.columns):
    ax=axes[i//4][i%4]
    ax.hist(X[col], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_title(col)

axes[2][2].set_visible(False)
axes[2][3].set_visible(False) #nascondo i riquardi vuoti
plt.tight_layout()
#plt.show()

fig,axes=plt.subplots(3,4, figsize=(16,12))
fig.suptitle("Boxplot delle Features", fontsize=16)

for i, col in enumerate(X.columns):
    ax=axes[i//4][i%4]
    ax.boxplot(X[col], patch_artist=True, boxprops=dict(facecolor="lightblue"))
    ax.set_title(col)

axes[2][2].set_visible(False)
axes[2][3].set_visible(False)
plt.tight_layout()
#plt.show()


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center =0)
plt.title("Mappa di correlazione")
#plt.show()



# Training e test
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#print(f"Training set: {X_train.shape[0]} campioni")
#print(f"Test set: {X_test.shape[0]} campioni")

# Standardizzazione
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)



# dizionario di modelli
models={
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "KNN": KNeighborsRegressor(),
    "SVM": SVR()
}

# k-fold con k=5
kf=KFold(n_splits=5, shuffle=True, random_state=42)

#print(f"{"Modello":<22}{"NMSE medio":>12}{"Std Dev":>10}")
#print("-"*46)

cv_result={}
for name, model in models.items():
    scores=cross_val_score(
        model, X_train_scaled, y_train,
        cv=kf, scoring="neg_mean_squared_error"
    )
    cv_result[name]=scores
    #print(f"{name:<22}{scores.mean():>12.2f}{scores.std():>10.2f}")


means=[cv_result[m].mean() for m in models]
stds=[cv_result[m].std() for m in models]
colors=["#256EAA", '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']

plt.figure(figsize=(10,6))
plt.barh(list(models.keys()), means, xerr=stds, color=colors, alpha=0.8, capsize=5)
plt.xlabel("NMSE (Negative Mean Squared Errors)")
plt.title("Confronto dei Modelli - K-Fold CV (k=5)")
plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
#plt.show()

# modelllo con NMSE medio più alto
best_model_name=max(cv_result, key=lambda m: cv_result[m].mean())
#print(f"Modello migliore: {best_model_name}")

from sklearn.model_selection import GridSearchCV

# valori alpha da provare
param_grid={"alpha":[0.01,0.1,1,10,100,1000]}

# Cross-Validation Grid search
grid_search=GridSearchCV(Ridge(), param_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Miglior alpha: {grid_search.best_params_["alpha"]}")
print(f"Miglior NMSE CV: {grid_search.best_score_:.2f}")

from sklearn.metrics import mean_squared_error, r2_score

# addestramento usando miglior alpha
best_alpha=grid_search.best_params_["alpha"]
best_model=Ridge(alpha=best_alpha)
best_model.fit(X_train_scaled, y_train)

# predizione
y_pred=best_model.predict(X_test_scaled)

# metrichefinali
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R2: {r2:.4f}")