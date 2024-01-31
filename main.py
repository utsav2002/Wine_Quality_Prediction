import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor

)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    confusion_matrix
)

from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    train_test_split,
    cross_validate
)

from sklearn.preprocessing import StandardScaler

from sklearn.svm import (
    LinearSVC,
    SVC,
    SVR
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

def evaluation_report(model, results, X_Train, Y_Train, X_Test, Y_Test, cv=10):
    """Prints a report and the cross validation score for a given classification model
    """

    print('Training set:\n')
    print(classification_report(Y_Train, results["pred_train"]))

    scores = cross_val_score(model, X_Train, Y_Train, cv=cv)
    print(f"{cv}-fold cross-validation scores for train: {scores.mean():.3f} (± {scores.std() * 2:.3f})")


    print('Testing set:\n')
    print(classification_report(Y_Test, results["pred_test"]))

    scores = cross_val_score(model, X_Test, Y_Test, cv=cv)
    print(f"{cv}-fold cross-validation scores for validation/test: {scores.mean():.3f} (± {scores.std() * 2:.3f})")

def regression_report(model, X, Y, cv=10):

    nmae = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_absolute_error")
    print("\nMean absolute error: ", -np.mean(nmae))
    print(nmae)

    nmse = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    print("\nMean squared error: ", -np.mean(nmse))
    print(nmse)

    r2 = cross_val_score(model, X, Y, cv=5, scoring="r2")
    print("\nR2 score: ", np.mean(r2))
    print(r2)

    return {
        "nmae": nmae,
        "nmse": nmse,
        "r2": r2
    }

def model_predict(model, X, X_2):
    return {
        "pred_train": model.predict(X),
        "pred_test": model.predict(X_2)
        }
df = pd.read_csv('winequality-red.csv', header=0, sep=";")
print("Red data shape: ", df.shape)
print("First 10 red data points:")
df.head(10)

g = sns.countplot(x="quality", data=df)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.show()

alcohol_std_dev = np.std(df["alcohol"])
alcohol_avg = np.mean(df["alcohol"])

low = alcohol_avg - alcohol_std_dev
high = alcohol_avg + alcohol_std_dev

bins = [0, low, high, np.max(df["alcohol"])+1]
labels = ["Low", "Mid", "High"]

df["alcohol_cat"] = pd.cut(df["alcohol"], bins=bins, labels=labels, right=False, include_lowest=True)

print(df["alcohol_cat"])

d = df.melt(id_vars="alcohol_cat", value_vars="quality", value_name="Quality")
# print(d)
sns.countplot(x="Quality", hue="alcohol_cat", data=d, edgecolor="k", linewidth=1)
sns.despine()

df_white = pd.read_csv('winequality-white.csv', header=0, sep=";")

fig, ax = plt.subplots(1,2, figsize=(20,5))

g = sns.histplot(df, x="residual sugar", ax=ax[0])
ax[0].set_title("Red wine residual sugars")

sns.histplot(df_white, x="residual sugar", ax=ax[1])
ax[1].set_title("White wine residual sugars")

plt.tight_layout()

df["residual sugar"].describe()

sugar_threshold = np.median(df["residual sugar"])  # This will split the classes evenly, as sugar is skewed in these datasets
print(sugar_threshold)

bins = [0, sugar_threshold, np.max(df["residual sugar"])+1]
labels = ["Not sweet", "Sweet"]

df["isSweet"] = pd.cut(df["residual sugar"], bins=bins, labels=labels, right=False, include_lowest=True)

g = sns.FacetGrid(df, col="isSweet", hue="isSweet")
g.map(sns.histplot, "quality")

for v in df.columns:
    g = sns.catplot(x="quality", y=v, data=df, kind="box")

fig, ax = plt.subplots(figsize=(10,6))
ax = sns.heatmap(df.corr(method="pearson"), annot=True)
