# -*- coding: utf-8 -*-
"""B21CS044_LabAssignment_8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n1E9Q2sFnXxLxxVf4X1s4iatR5I6EBOn
"""

#importing all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

"""QUESTION 1"""

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#loading the dataset
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

#find the no of unique values in each column
X_train.nunique()

#find the null values in each column
X_train.isnull().sum()

X_test.isnull().sum()

#dropping id and unnamed columns and seperating the target variable
X_train.drop(['id','Unnamed: 0'],axis=1,inplace=True)
X_test.drop(['id','Unnamed: 0'],axis=1,inplace=True)
y_train = X_train['satisfaction']
X_train.drop(['satisfaction'],axis=1,inplace=True)
y_test = X_test['satisfaction']
X_test.drop(['satisfaction'],axis=1,inplace=True)

X_train.head()

#filling the null values with mean
X_train.fillna(X_train.mean(),inplace=True)
X_test.fillna(X_test.mean(),inplace=True)

#visualizing the features with scatter plot and hue as target variable
sns.scatterplot(x='Age',y='Flight Distance',hue=y_train,data=X_train)

#visualizing the features with histogram
X_train.hist(figsize=(20,20))

#encoding the categorical variables with ordinal encoding
X_train['Type of Travel'] = X_train['Type of Travel'].map({'Personal Travel':0,'Business travel':1})
X_test['Type of Travel'] = X_test['Type of Travel'].map({'Personal Travel':0,'Business travel':1})
X_train['Class'] = X_train['Class'].map({'Eco':0,'Eco Plus':1,'Business':2})
X_test['Class'] = X_test['Class'].map({'Eco':0,'Eco Plus':1,'Business':2})
X_train['Customer Type'] = X_train['Customer Type'].map({'Loyal Customer':0,'disloyal Customer':1})
X_test['Customer Type'] = X_test['Customer Type'].map({'Loyal Customer':0,'disloyal Customer':1})
X_train['Gender'] = X_train['Gender'].map({'Female':0,'Male':1})
X_test['Gender'] = X_test['Gender'].map({'Female':0,'Male':1})

X_train.head()

#plotting the correlation matrix
plt.figure(figsize=(20,20))
sns.heatmap(X_train.corr(),annot=True)

"""2) Create an object of SFS by embedding the Decision Tree classifier object, providing 10 features, 
forward as True, floating as False and scoring = accuracy. Train SFS and report accuracy for all 10 
features. Also, list the names of the 10 best features selected by SFS. [10 marks]

"""

#using the sequential feature selector to select the best 10 features
sfs = SFS(DecisionTreeClassifier(),k_features=10,forward=True,floating=False,verbose=2,scoring='accuracy',cv=5)
sfs.fit(X_train,y_train)
sfs.k_feature_names_

#calculating the accuracy score for all 10 features
clf = DecisionTreeClassifier()
clf.fit(X_train[list(sfs.k_feature_names_)],y_train)
y_pred = clf.predict(X_test[list(sfs.k_feature_names_)])
print('accuracy_score for 10 best features',accuracy_score(y_test,y_pred))

""" Using the forward and Floating parameter toggle between SFS(forward True, floating False), SBS 
(forward False, floating False), SFFS (forward True, floating True), SBFS (forward False, floating True), 
and choose cross validation = 4 for each configuration. Also, report cv scores for each configuration. [5 
marks]
"""

sfs_T_F = SFS(DecisionTreeClassifier(),k_features=10,forward=True,floating=False,verbose=2,scoring='accuracy',cv=4)
sfs_T_F.fit(X_train,y_train)

sfs_T_T = SFS(DecisionTreeClassifier(),k_features=10,forward=True,floating=True,verbose=2,scoring='accuracy',cv=4)
sfs_T_T.fit(X_train,y_train)

sfs_F_F = SFS(DecisionTreeClassifier(),k_features=10,forward=False,floating=False,verbose=2,scoring='accuracy',cv=4)
sfs_F_F.fit(X_train,y_train)

sfs_F_T = SFS(DecisionTreeClassifier(),k_features=10,forward=False,floating=True,verbose=2,scoring='accuracy',cv=4)
sfs_F_T.fit(X_train,y_train)

#printing the scores of the models
print('Sequential Feature Selection forward = True and floating = False : ',sfs_T_F.k_score_)
print('Sequential Feature Selection forward = True and floating = True : ',sfs_T_T.k_score_)
print('Sequential Feature Selection forward = False and floating = False : ',sfs_F_F.k_score_)
print('Sequential Feature Selection forward = False and floating = True : ',sfs_F_T.k_score_)

""" Visualize the output from the feature selection in a pandas DataFrame format using the get_metric_dict 
for all four configurations. Finally, plot the results for each configuration (from mlxtend. plotting import 
plot_sequential_feature_selection as plot_sfs). [10 marks]

"""

#converting get_metric_dict to dataframe
df_T_F = pd.DataFrame.from_dict(sfs_T_F.get_metric_dict()).T
df_T_T = pd.DataFrame.from_dict(sfs_T_T.get_metric_dict()).T
df_F_F = pd.DataFrame.from_dict(sfs_F_F.get_metric_dict()).T
df_F_T = pd.DataFrame.from_dict(sfs_F_T.get_metric_dict()).T

print('Sequential Feature Selection forward = True and floating = True : ')
df_T_T.head()

print('Sequential Feature Selection forward = False and floating = False : ')
df_F_F.head()

print('Sequential Feature Selection forward = False and floating = True : ')
df_F_T.head()

print('Sequential Feature Selection forward = True and floating = False : ')
df_T_F.head()

from mlxtend. plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs_T_F.get_metric_dict(),kind='std_dev')
plt.title('Sequential Forward Selection (w. StdDev) T_F')
plt.grid()
plt.show()

#plotting the sequential feature selection for forward = True and floating = True
fig2 = plot_sfs(sfs_T_T.get_metric_dict(),kind='std_dev')
plt.title('Sequential Forward Selection (w. StdDev) T_T')
plt.grid()
plt.show()

#plotting the sequential feature selection for forward = False and floating = False
fig3 = plot_sfs(sfs_F_F.get_metric_dict(),kind='std_dev')
plt.title('Sequential Forward Selection (w. StdDev) F_F')
plt.grid()
plt.show()

#plotting the sequential feature selection for forward = False and floating = True
fig4 = plot_sfs(sfs_F_T.get_metric_dict(),kind='std_dev')
plt.title('Sequential Forward Selection (w. StdDev) F_T')
plt.grid()
plt.show()

"""5) Implement Bi-directional Feature Set Generation Algorithm from scratch. It must take a Full Set of 
features as well as similarity measures as input. [10 marks]
6) Use the function implemented in part 5 and use selection criteria from the following: [10 marks]
● Accuracy Measures: using Decision Tree and SVM Classifiers
● Information Measures: Information gain
● Distance Measure: Angular Separation, Euclidian Distance and City-Block Distance 
● Distance Measures. - Measures of separability, discrimination or divergence measures. The 
most typical is derived from the distance between the class conditional density functions.)
"""

#implementing the sequential feature selector with forward = True and floating = True from scratch
from sklearn.model_selection import cross_val_score
#the fuction takes the training data, target variable, model, no of features to be selected, scoring metric and no of folds as input and returns the selected features and the score
def sfs_scratch(X,y,model,k,scoring,cv):
    features = list(X.columns)
    #first we will sort the features on thier correlation with the target variable
    selected_features = []
    score = 0
    while(len(selected_features) < k):
        max_score = 0
        for feature in features:
            selected_features.append(feature)
            score = cross_val_score(model,X[selected_features],y,scoring=scoring,cv=cv).mean()
            if(score > max_score):
                max_score = score
                best_feature = feature
            selected_features.remove(feature)
        selected_features.append(best_feature)
        features.remove(best_feature)
    score = cross_val_score(model,X[selected_features],y,scoring=scoring,cv=cv).mean()
    return selected_features,score

"""Train any classifier of your choice on the Selected features generated from each measure and report its 
classification results. [10 marks]
"""

#using decision tree classifier as the model, and information gain as the scoring metric and distance mesure as euclidean distance
selected_features_dtc,score_dtc = sfs_scratch(X_train,y_train,DecisionTreeClassifier(),10,'accuracy',5)
print('The selected features are : ',selected_features_dtc)
print('The score is : ',score_dtc)

#making a new dataframe with the selected features_dtc from X_train
X_train_dtc = X_train[selected_features_dtc]
X_test_dtc = X_test[selected_features_dtc]

from sklearn.neighbors import KNeighborsClassifier
#fitting the knn on the selected features
knn_dtc = KNeighborsClassifier()
knn_dtc.fit(X_train_dtc,y_train)
y_pred_dtc = knn_dtc.predict(X_test_dtc)
print('The accuracy score is : ',accuracy_score(y_test,y_pred_dtc))

#applying gnb on dtc selected features
from sklearn.naive_bayes import GaussianNB
gnb_dtc = GaussianNB()
gnb_dtc.fit(X_train_dtc,y_train)
y_pred_gnb_dtc = gnb_dtc.predict(X_test_dtc)
print('The accuracy score is : ',accuracy_score(y_test,y_pred_gnb_dtc))

#applying svm on dtc selected features
from sklearn.svm import SVC
svm_dtc = SVC()
svm_dtc.fit(X_train_dtc,y_train)
y_pred_svm_dtc = svm_dtc.predict(X_test_dtc)
print('The accuracy score is : ',accuracy_score(y_test,y_pred_svm_dtc))

#plotting the k-fold scores of knn,svm and gnb
from sklearn.model_selection import cross_val_score
knn_scores = cross_val_score(knn_dtc,X_train_dtc,y_train,cv=5)
svm_scores = cross_val_score(svm_dtc,X_train_dtc,y_train,cv=5)
gnb_scores = cross_val_score(gnb_dtc,X_train_dtc,y_train,cv=5)

plt.plot(knn_scores,label='knn')
plt.plot(svm_scores,label='svm')
plt.plot(gnb_scores,label='gnb')
plt.legend()
plt.show()

"""QUESTION 2

Make a Dataset of 1000 points sampled from a zero-centred gaussian distribution with a covariance 
matrix
"""

cov_mat=np.array([[0.6006771,0.14889879,0.244939],[0.14889879,0.58982531,0.24154981],[0.244939,0.24154981,0.48778655]])

#generating 1000 random samples from cov_mat
data=np.random.multivariate_normal([0,0,0],cov_mat,1000)
#converting data to dataframe
df=pd.DataFrame(data,columns=['x1','x2','x3'])
df.head()

#making vector v
from math import sqrt
vector_v = np.array([1/sqrt(6),1/sqrt(6),-2/sqrt(6)])
vector_v

#making dustribution of data into classses
df['class'] = df.apply(lambda row: 0 if np.dot(row,vector_v) > 0 else 1, axis=1)
df.head()

#splitting into X_data and y_data
X_data=df.drop('class',axis=1)
y_data=df['class']

#Visualizing the data as a 3D scatter-plot using plotly’s scatter_3d function
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(x=X_data['x1'], y=X_data['x2'], z=X_data['x3'], mode='markers',marker=dict(color=y_data))])
fig.show()

"""Apply Principal Component analysis (using sklearn) with n_components=3 on the input data X 
and transform the data accordingly. 
"""

#applying PCA on the data
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X_data)
X_pca = pca.transform(X_data)

#printing the shape of X_pca
X_pca.shape

#visualizing the data after applying PCA using plotly’s scatter_3d function
fig = go.Figure(data=[go.Scatter3d(x=X_pca[:,0], y=X_pca[:,1], z=X_pca[:,2], mode='markers',marker=dict(color=y_data))])
fig.show()

#convert X_pca to dataframe
X_pca_df=pd.DataFrame(X_pca,columns=['x1','x2','x3'])
X_pca_df.head()

"""Perform Complete FS on the Transformed Data with a number of features in subset =2. Fit a 
Decision Tree for every subset-set of features of size 2 and plot their decision boundaries 
superimposed with the data
"""

#performing feature selection with a number of features in a subset = 2.
#using Exhaustive Feature Selection
#fitting decision tree classifier at a subset of features of size 2
clf = DecisionTreeClassifier()
efs = EFS(clf, min_features=2, max_features=2, scoring='accuracy', print_progress=True, cv=5)
efs = efs.fit(X_pca_df, y_data)

#printing the subset of features
efs.subsets_

#plotting decision boundaries for x1 and x2
X_x1_x2 = X_pca_df[['x1','x2']]
dtc_x1_x2 = DecisionTreeClassifier()

#train_test_split
X_train_x1_x2, X_test_x1_x2, y_train_x1_x2, y_test_x1_x2 = train_test_split(X_x1_x2, y_data, test_size=0.2, random_state=42)
dtc_x1_x2.fit(X_train_x1_x2, y_train_x1_x2)
fig = plot_decision_regions(X_train_x1_x2.values, y=y_train_x1_x2.values, clf=dtc_x1_x2, legend=2)
plt.title('Decision Boundary for x1 and x2')
plt.show()

#plotting decision boundaries for x1 and x3
X_x1_x3 = X_pca_df[['x1','x3']]
dtc_x1_x3 = DecisionTreeClassifier()

#train_test_split
X_train_x1_x3, X_test_x1_x3, y_train_x1_x3, y_test_x1_x3 = train_test_split(X_x1_x3, y_data, test_size=0.2, random_state=42)
dtc_x1_x3.fit(X_train_x1_x3, y_train_x1_x3)
fig = plot_decision_regions(X_train_x1_x3.values, y=y_train_x1_x3.values, clf=dtc_x1_x3, legend=2)
plt.title('Decision Boundary for x1 and x3')
plt.show()

#plotting decision boundaries for x2 and x3
X_x2_x3 = X_pca_df[['x2','x3']]
dtc_x2_x3 = DecisionTreeClassifier()

#train_test_split
X_train_x2_x3, X_test_x2_x3, y_train_x2_x3, y_test_x2_x3 = train_test_split(X_x2_x3, y_data, test_size=0.2, random_state=42)
dtc_x2_x3.fit(X_train_x2_x3, y_train_x2_x3)
fig = plot_decision_regions(X_train_x2_x3.values, y=y_train_x2_x3.values, clf=dtc_x2_x3, legend=2)
plt.title('Decision Boundary for x2 and x3')
plt.show()

"""Which of the above feature subsets represents the one that can be obtained by applying 
PCA(n_components =2)? Explain the difference in the accuracies between this subset and other 
subsets by running suitable experiments
"""

#plotting corelation matrix
plt.figure(figsize=(10,10))
sns.heatmap(X_pca_df.corr(),annot=True,cmap='coolwarm')
plt.show()

#plotting variance explained by each component
print(pca.explained_variance_ratio_)

"""We know that pca sorts the order of PC's according to the ration explained by them. So, the first two PC's will be the ones that explain the most variance in the data. So, the subset obtained by PCA(n_components =2) will be the one that has the first two PC's as the features."""

#printing the acuracies of all the subsets using all models of dtc above
print('Accuracy of x1 and x2:',accuracy_score(y_test_x1_x2,dtc_x1_x2.predict(X_test_x1_x2)))
print('Accuracy of x1 and x3:',accuracy_score(y_test_x1_x3,dtc_x1_x3.predict(X_test_x1_x3)))
print('Accuracy of x2 and x3:',accuracy_score(y_test_x2_x3,dtc_x2_x3.predict(X_test_x2_x3)))

#printing the f1_score of all the subsets using all models of dtc above
print('F1_score of x1 and x2:',f1_score(y_test_x1_x2,dtc_x1_x2.predict(X_test_x1_x2)))
print('F1_score of x1 and x3:',f1_score(y_test_x1_x3,dtc_x1_x3.predict(X_test_x1_x3)))
print('F1_score of x2 and x3:',f1_score(y_test_x2_x3,dtc_x2_x3.predict(X_test_x2_x3)))

#roc_auc_score of all the subsets using all models of dtc above
print('roc_auc_score of x1 and x2:',roc_auc_score(y_test_x1_x2,dtc_x1_x2.predict(X_test_x1_x2)))
print('roc_auc_score of x1 and x3:',roc_auc_score(y_test_x1_x3,dtc_x1_x3.predict(X_test_x1_x3)))
print('roc_auc_score of x2 and x3:',roc_auc_score(y_test_x2_x3,dtc_x2_x3.predict(X_test_x2_x3)))

"""As from above results we can see that it is not necessary that the subset obtained by PCA(n_components =2) will have the highest accuracy. The subset obtained by PCA(n_components =2) has the lowest accuracy. This is because the subset obtained by PCA(n_components =2) explains the most variance in the data but it does not mean that it will have the highest accuracy. The subset obtained by PCA(n_components =2) has the lowest accuracy because it is not able to capture the non-linear relationship between the features and the target variable. In short, it is not necesary that variance always explains best information about the data.

#appling lda on the data
"""

#applying lda to the dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_data, y_data)

#printing the accuracy of the lda model
print('Accuracy of lda model:',accuracy_score(y_data,lda.predict(X_data)))

"""It shows that variance explaination not always gives the best information about the data. The subset obtained by PCA(n_components =2) has the lowest accuracy because it is not able to capture the non-linear relationship between the features and the target variable.But in lda it give best result even on n_components=1.This is because the classes goes severly mixed up on performing PCA."""