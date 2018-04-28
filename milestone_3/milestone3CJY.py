# Imports

# pandas
import pandas as pd
from pandas import Series, DataFrame

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
# milestones1
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree

# milestones2
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.gaussian_process.kernels import Product, ConstantKernel as C

# milestone3
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

# cross validation
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# new log error
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer


########################################### load data
train_df = pd.read_csv("/Users/jiayaocheng/Documents/517Python/Titanic/train.csv")

########################################### preprocess
# dropping features that not related(name and id) or not very useful(ticket) or too many null values(cabin)
train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)

# Pclass: this feature only have 1,2,and 3, do not need preprocess

# Sex: convert sex feature from str to int, female=1 and male=0
for data in [train_df]:
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Age: fill null values of age based on age distribution in different Pclass
guess_age = np.zeros((2, 3))
for data in [train_df]:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = data[(data['Sex'] == i) & (data['Pclass'] == j + 1)]['Age'].dropna()
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_age[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            data.loc[(data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j + 1), 'Age']=guess_age[i, j]
    data['Age'] = data['Age'].astype(int)
# replace age with ordinals
for data in [train_df]:
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age'] = 4

# SibSp & Parch: create Alone feature
train_df['Familynumbers'] = train_df['SibSp'] + train_df['Parch'] + 1
for data in [train_df]:
    data['Alone'] = 0
    data.loc[data['Familynumbers'] == 1, 'Alone'] = 1
train_df[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean()
# drop Parch, SibSp, and Familynumbers features
train_df = train_df.drop(['Parch', 'SibSp', 'Familynumbers'], axis=1)

# Fare: processing fare
train_df['Fare2'] = pd.qcut(train_df['Fare'], 4)
train_df[['Fare2', 'Survived']].groupby(['Fare2'], as_index=False).mean().sort_values(by='Fare2', ascending=True)
for data in [train_df]:
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3
train_df = train_df.drop(['Fare2'], axis=1)

# Embarked: convert embarked feature from str to int, S=0, C=1, Q=2
frequency_embarked = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(frequency_embarked)
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


########################################### predition
#X_train, X_test, Y_train, Y_test = train_test_split(train_df.drop("Survived", axis=1), train_df["Survived"],test_size=0.2, random_state=0)
X = train_df.drop("Survived", axis=1)
Y = train_df["Survived"]

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
###################### Add two high correlated data
X_gender = train_df.drop(["Survived","Embarked","Fare","Alone","Age"], axis=1)
#X_pclass = train_df["Pclass"]

#kf=KFold(len(Y), n_folds=10)   in cross_val_score, is cv=int n, then, is use n_fold validation

logreg = LogisticRegression()
logregscores = cross_val_score(logreg, X, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of logreg after 10fold-val: %0.2f (+/- %0.2f)" % (logregscores.mean(), logregscores.std() * 2))

percep = Perceptron(max_iter=1000, tol=None)
percepscores = cross_val_score(percep, X, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of percep after 10fold-val: %0.2f (+/- %0.2f)" % (percepscores.mean(), percepscores.std() * 2))

DCtree = tree.DecisionTreeClassifier()
DCtreescores = cross_val_score(DCtree, X, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of DCtree after 10fold-val: %0.2f (+/- %0.2f)" % (DCtreescores.mean(), DCtreescores.std() * 2))
#
# #Gaua = GaussianProcessClassifier(kernel=default RBF + dotproduct + dotproduct**2 + Constant)
Gaua1 = GaussianProcessClassifier(1.0 * RBF(1.0))
Gaua1scores = cross_val_score(Gaua1, X, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of GP+RBF after 10fold-val: %0.2f (+/- %0.2f)" % (Gaua1scores.mean(), Gaua1scores.std() * 2))
#
# # Gaua2 = GaussianProcessClassifier(1.0 * DotProduct(sigma_0=1.0))
# # Gaua2scores = cross_val_score(Gaua2, X, Y , scoring=make_scorer(log_loss), cv=10)
# # print("Negative-log-loss of GP+Dot after 10fold-val: %0.2f (+/- %0.2f)" % (Gaua2scores.mean(), Gaua2scores.std() * 2))
#
# Gaua3 = GaussianProcessClassifier(1.0 * DotProduct(sigma_0=1.0)**2)
# Gaua3scores = cross_val_score(Gaua3, X, Y , scoring=make_scorer(log_loss), cv=10)
# print("Negative-log-loss of GP+D^2 after 10fold-val: %0.2f (+/- %0.2f)" % (Gaua3scores.mean(), Gaua3scores.std() * 2))
#
# # Gaua3 = GaussianProcessClassifier(Product(1.0 * RBF(1.0), 1.0 * DotProduct(sigma_0=1.0)**2))
# # Gaua3scores = cross_val_score(Gaua3, X, Y , cv=10)
# # print("Accuracy of GP+Sum after 10fold-val: %0.2f (+/- %0.2f)" % (Gaua3scores.mean(), Gaua3scores.std() * 2))
#
# Gaua4 = GaussianProcessClassifier(C(0.1, (0.00001, 10.0)))
# Gaua4scores = cross_val_score(Gaua4, X, Y , scoring=make_scorer(log_loss), cv=10)
# print("Negative-log-loss of GP+Cst after 10fold-val: %0.2f (+/- %0.2f)" % (Gaua4scores.mean(), Gaua4scores.std() * 2))


############################################### dimension reduction
X_centered = X - X.mean(axis=0)
kpca = KernelPCA(kernel="rbf",n_components=2, gamma=0.125)
X_kpca = kpca.fit_transform(X_centered)
pca = PCA(n_components=2, svd_solver="auto", whiten=False)
X_pca = pca.fit_transform(X_centered)

plt.figure()

plt.subplot(2, 1, 1, aspect='equal')
plt.scatter(X_pca[Y == 0, 0], X_pca[Y == 0, 1], c="red", s=20, edgecolor='k')
plt.scatter(X_pca[Y == 1, 0], X_pca[Y == 1, 1], c="blue",s=20, edgecolor='k')
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.subplot(2, 1, 2, aspect='equal')
plt.scatter(X_kpca[Y == 0, 0], X_kpca[Y == 0, 1], c="red", s=20, edgecolor='k')
plt.scatter(X_kpca[Y == 1, 0], X_kpca[Y == 1, 1], c="blue",s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

# plt.subplot(2, 2, 3, aspect='equal')
# plt.scatter(X_gender[Y == 0, 0], X_gender[Y == 0, 1], c="red", s=20, edgecolor='k')
# plt.scatter(X_gender[Y == 1, 0], X_gender[Y == 1, 1], c="blue",s=20, edgecolor='k')
# plt.title("Projection by choose")
# plt.xlabel("1st principal component")
# plt.ylabel("2nd component")


plt.show()

logregscoresSex = cross_val_score(logreg, X_gender, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of logreg of two main features after 10fold-val: %0.2f (+/- %0.2f)" % (logregscoresSex.mean(), logregscoresSex.std() * 2))

logregscores2 = cross_val_score(logreg, X_pca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of logreg after PCA & 10fold-val: %0.2f (+/- %0.2f)" % (logregscores2.mean(), logregscores2.std() * 2))

percepscores2 = cross_val_score(percep, X_pca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of percep after PCA & 10fold-val: %0.2f (+/- %0.2f)" % (percepscores2.mean(), percepscores2.std() * 2))

DCtreescores2 = cross_val_score(DCtree, X_pca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of DCtree after PCA & 10fold-val: %0.2f (+/- %0.2f)" % (DCtreescores2.mean(), DCtreescores2.std() * 2))

Gaua1scores2 = cross_val_score(Gaua1, X_pca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of GP+RBF after PCA & 10fold-val: %0.2f (+/- %0.2f)" % (Gaua1scores2.mean(), Gaua1scores2.std() * 2))


logregscores3 = cross_val_score(logreg, X_kpca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of logreg after KPCA & 10fold-val: %0.2f (+/- %0.2f)" % (logregscores3.mean(), logregscores3.std() * 2))

percepscores3 = cross_val_score(percep, X_kpca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of percep after KPCA & 10fold-val: %0.2f (+/- %0.2f)" % (percepscores3.mean(), percepscores3.std() * 2))

DCtreescores3 = cross_val_score(DCtree, X_kpca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of DCtree after KPCA & 10fold-val: %0.2f (+/- %0.2f)" % (DCtreescores3.mean(), DCtreescores3.std() * 2))

Gaua1scores3 = cross_val_score(Gaua1, X_kpca, Y , scoring=make_scorer(log_loss), cv=10)
print("Negative-log-loss of GP+RBF after KPCA & 10fold-val: %0.2f (+/- %0.2f)" % (Gaua1scores3.mean(), Gaua1scores3.std() * 2))