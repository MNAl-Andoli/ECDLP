import numpy as np
# Using simple Decision Tree classifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time
from sklearn.decomposition import PCA

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn import tree
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# load dataset
x_train=np.load("x_train2.npy")

'''x_train=np.load("X_train.npy")
pca_x_train = PCA(n_components=250)
x_train = pca_x_train.fit_transform(x_train)'''

#x_train=np.load("x_train2.npy")
#===normalize x_train. z-score
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
#===
y_train=np.load("y_train11.npy")

# -------------------------------------

x_test=np.load("x_test2.npy")
'''x_test=np.load("X_test.npy")
pca_x_test = PCA(n_components=250)
x_test = pca_x_test.fit_transform(x_test)
'''
#====z_score normalization
scaler2=StandardScaler()
scaler2.fit(x_test)
x_test=scaler2.transform(x_test)
#====
y_test=np.load("y_test11.npy")

print(x_train.shape, x_test.shape)

# Using simple Decision Tree classifier
t1=time.time()
dt_clf = tree.DecisionTreeClassifier(max_depth=7)
dt_clf.fit(x_train, y_train)
acc_dt=dt_clf.score(x_test, y_test)

t2=time.time()

dt_pred=dt_clf.predict(x_test)
#conf_matrix = confusion_matrix(y_true=y_test, y_pred=dt_pred)
print('Precision: %.3f' % precision_score(y_test, dt_pred))
print('Recall: %.3f' % recall_score(y_test, dt_pred))
print('F1 Score: %.3f' % f1_score(y_test, dt_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, dt_pred))
print("Accuracy of Decision Tree Classifier :",acc_dt, " time: ", (t2-t1))


#Using Random Forest Classifier
t1=time.time()
rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(x_train, y_train)
acc_rf=rf_clf.score(x_test, y_test)

rf_pred=rf_clf.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, rf_pred))
print('Recall: %.3f' % recall_score(y_test, rf_pred))
print('F1 Score: %.3f' % f1_score(y_test, rf_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, rf_pred))

t2=time.time()

print("Accuracy of Random Forest Classifier :",acc_rf, " time: ", (t2-t1))

#Using Gradient Boosting Classifier
t1=time.time()
gb_clf = ensemble.GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)
acc_gb=gb_clf.score(x_test, y_test)

t2=time.time()
print("Accuracy of Gradient Boosting Classifier :",acc_gb, " time: ", (t2-t1))

gb_pred=gb_clf.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, gb_pred))
print('Recall: %.3f' % recall_score(y_test, gb_pred))
print('F1 Score: %.3f' % f1_score(y_test, gb_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, gb_pred))


#Using Naive Bayes Classifier
t1= time.time()
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)
# evaluaye the accuracy by this instruction
acc_nb=nb_clf.score(x_test, y_test)
# Or by this, 
#predict the response for test dataset
#y_pred = nb1_clf.predict(x_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
nb_pred=nb_clf.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, nb_pred))
print('Recall: %.3f' % recall_score(y_test, nb_pred))
print('F1 Score: %.3f' % f1_score(y_test, nb_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, nb_pred))

t2=time.time()
print("Accuracy of Naive Bayes Classifier :",acc_nb, " time: ", (t2-t1))


#Using K-nearest neighbors Classifier


t1= time.time()
knn_clf = KNeighborsClassifier(n_neighbors=2)
knn_clf.fit(x_train, y_train)
acc_knn_clf=knn_clf.score(x_test, y_test)
t2=time.time()
print("Accuracy of K-nearest neighbors Classifier :",acc_knn_clf, " time: ", (t2-t1))


nb_pred=knn_clf.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, nb_pred))
print('Recall: %.3f' % recall_score(y_test, nb_pred))
print('F1 Score: %.3f' % f1_score(y_test, nb_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, nb_pred))


#Using Logistic Regression Classifier
t1= time.time()
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(x_train, y_train)
acc_lr=lr_clf.score(x_test, y_test)

lr_pred=lr_clf.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, lr_pred))
print('Recall: %.3f' % recall_score(y_test, lr_pred))
print('F1 Score: %.3f' % f1_score(y_test, lr_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, lr_pred))

t2=time.time()

print("Accuracy of Naive Bayes Classifier :",acc_lr, " time: ", (t2-t1))


#Using SVM Classifier
t1= time.time()
svm_clf = SVC(probability=True)
svm_clf.fit(x_train, y_train)
acc_svm=svm_clf.score(x_test, y_test)
t2=time.time()
print("Accuracy of SVM Classifier :",acc_svm, " time: ", (t2-t1))


svm_pred=svm_clf.predict(x_test)

print('Precision: %.3f' % precision_score(y_test, svm_pred))
print('Recall: %.3f' % recall_score(y_test, svm_pred))
print('F1 Score: %.3f' % f1_score(y_test, svm_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, svm_pred))

