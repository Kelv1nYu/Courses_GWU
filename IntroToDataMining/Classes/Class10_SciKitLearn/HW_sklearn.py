#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#%%

def KNNModel(n, x1, y1, scaled = False):
    if not scaled:
        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1)
        knn_split = KNeighborsClassifier(n_neighbors=n) # instantiate with n value given
        knn_split.fit(x1_train,y1_train)
        y1test_pred = knn_split.predict(x1_test)
        print('unscaled_result', y1test_pred)
        print("unscaled_result_score", knn_split.score(x1_test,y1_test))
        cv_results = cross_val_score(knn_split, x1, y1, cv=10)
        print("unscaled_cv_result", cv_results)
        print("unscaled_cv_mean", np.mean(cv_results))
    else:
        if type(x1) is np.ndarray:
            xs1 = pd.DataFrame(scale(x1))
        else:
            xs1 = pd.DataFrame(scale(x1), columns = x1.columns)
        ys1 = y1.copy()
        xs1_train, xs1_test, ys1_train, ys1_test = train_test_split(xs1, ys1)
        knn_s = KNeighborsClassifier(n_neighbors=n) # instantiate with n value given
        knn_s.fit(xs1_train,ys1_train)
        ys1test_pred = knn_s.predict(xs1_test)
        print("scaled_result", ys1test_pred)
        print("scaled_result_score", knn_s.score(xs1_test,ys1_test))
        scv_results = cross_val_score(knn_s, xs1, ys1, cv=10)
        print("scaled_cv_result", scv_results) 
        print("scaled_cv_mean", np.mean(scv_results))

# %%
# get data
dfadmit = pd.read_csv("gradAdmit.csv")
wine = sklearn.datasets.load_wine()

# %%
# Q1

x1 = dfadmit[['admit','gre', 'gpa']]
y1 = dfadmit['rank']

# k = 5, unscaled
print("k = 5, unscaled")
KNNModel(5, x1, y1)

# k = 5, scaled
print("k = 5, scaled")
KNNModel(5, x1, y1, True)

# k = 9, unscaled
print("k = 9, unscaled")
KNNModel(9, x1, y1)

# k = 9, scaled
print("k = 9, scaled")
KNNModel(9, x1, y1, True)

# k = 15, unscaled
print("k = 15, unscaled")
KNNModel(15, x1, y1)

# k = 15, scaled
print("k = 15, scaled")
KNNModel(15, x1, y1, True)

# k = 21, unscaled
print("k = 21, unscaled")
KNNModel(21, x1, y1)

# k = 21, scaled
print("k = 21, scaled")
KNNModel(21, x1, y1, True)

# %%
# Q2

x1 = wine.data
y1 = wine.target

# k = 5, unscaled
print("k = 5, unscaled")
KNNModel(5, x1, y1)

# k = 5, scaled
print("k = 5, scaled")
KNNModel(5, x1, y1, True)

# k = 9, unscaled
print("k = 9, unscaled")
KNNModel(9, x1, y1)

# k = 9, scaled
print("k = 9, scaled")
KNNModel(9, x1, y1, True)

# k = 21, unscaled
print("k = 21, unscaled")
KNNModel(21, x1, y1)

# k = 21, scaled
print("k = 21, scaled")
KNNModel(21, x1, y1, True)

# k = 25, unscaled
print("k = 25, unscaled")
KNNModel(25, x1, y1)

# k = 25, scaled
print("k = 25, scaled")
KNNModel(25, x1, y1, True)

# %%
# Q3

lr = LogisticRegression()
lr.fit(wine.data, wine.target)
lr.score(wine.data, wine.target)

# The score of Logistic Regression is 0.9719
# The score of the best KNN model is 0.9777 data is scaled, cv_mean is 0.9725 when k = 21.
# I think we cannot say that KNN model is better than the Logistic Regression Model.
# Because we may spend a lot of time in getting a suitable k-value.



# %%
# Q4

# logit
xadmit = dfadmit[['gre', 'gpa', 'rank']]
yadmit = dfadmit['admit']
x_trainAdmit, x_testAdmit, y_trainAdmit, y_testAdmit = train_test_split(xadmit, yadmit)
admitlogit = LogisticRegression()  # instantiate
admitlogit.fit(x_trainAdmit, y_trainAdmit)
admitlogit.predict(x_testAdmit)
print('Logit model accuracy (with the test set):', admitlogit.score(x_testAdmit, y_testAdmit))
y_true, y_pred = y_testAdmit, admitlogit.predict(x_testAdmit)
print(classification_report(y_true, y_pred))

# knn
knn_split = KNeighborsClassifier(n_neighbors=3) # instantiate with n value given
knn_split.fit(x_trainAdmit,y_trainAdmit)
ytest_pred = knn_split.predict(x_testAdmit)
print(ytest_pred)
print(knn_split.score(x_testAdmit,y_testAdmit))
knn_cv = KNeighborsClassifier(n_neighbors=3) # instantiate with n value given
cv_results = cross_val_score(knn_cv, xadmit, yadmit, cv=5)
print(cv_results) 
np.mean(cv_results) 


# The results of two model are very close to each other.
# But logit has a accuracy with 0.67, and knn has a score with 0.7

# %%
# Q5

dfpizza = pd.read_csv('Pizza.csv')
xpizza = dfpizza[['mois', 'prot', 'fat', 'ash', 'sodium', 'carb']]


# plot the 3 clusters
def plotClusters(index1, index2, n):
    km_xpizza = KMeans( n_clusters=n, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
    y_km = km_xpizza.fit_predict(xpizza)
    color = ['orange', 'yellow', 'green', 'blue', 'purple', 'pink']
    for i in range(0, n):
        plt.scatter( xpizza[y_km==i].iloc[:,index1], xpizza[y_km==i].iloc[:,index2], s=50, c=color[i], marker='s', edgecolor='black', label = i )
# plot the centroids
    plt.scatter( km_xpizza.cluster_centers_[:, index1], km_xpizza.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
    plt.legend(scatterpoints=1)
    plt.xlabel(xpizza.columns[index1])
    plt.ylabel(xpizza.columns[index2])
    plt.grid()
    plt.show()

plotClusters(1, 3, 3)
plotClusters(1, 3, 4)
plotClusters(1, 3, 5)

plotClusters(2, 3, 3)
plotClusters(2, 3, 4)
plotClusters(2, 3, 5)

plotClusters(0, 3, 3)
plotClusters(0, 3, 4)
plotClusters(0, 3, 5)


# The most interesting things I found is that with the increase of K value, the clustering effect is getting better and better, and the range of each cluster is getting smaller and smaller.
# But when my K value is getting smaller and smaller, the scope of each class is not getting larger, but only one class is getting larger. Most of the time, this class is in the upper right corner.

# %%
