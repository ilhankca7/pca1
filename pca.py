import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/ilhan/Downloads/winequality-red.csv")
veri = data.copy()

y = veri["quality"]
X = veri.drop(columns="quality", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

pca = PCA()

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(np.cumsum(pca.explained_variance_ratio_) * 100)

lm = LinearRegression()
RMSE = []

for i in range(1, X_train_pca.shape[1] + 1):
    hata = np.sqrt(-1 * cross_val_score(lm, X_train_pca[:, :i], y_train.ravel(),
                                        cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                        scoring="neg_mean_squared_error").mean())
    RMSE.append(hata)

print(RMSE)
