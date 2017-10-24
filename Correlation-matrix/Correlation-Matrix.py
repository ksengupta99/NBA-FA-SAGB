import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
%matplotlib inline

import math
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# import advanced statistics and drop empty columns
adv_stats = pd.read_csv('player-data\player2017\Adv-2017.csv')
adv_stats = adv_stats.drop(['Unnamed: 19','Unnamed: 24'], axis = 1)

per_36 = pd.read_csv('player-data\player2017\Per36-2017.csv')
print(per_36.columns)

# merge adv and per 36 and drop non-metric columns (rk, player, etc.)/ redundant columns

# dropped column: explanation

# All %'s dropped in leiu of their corresponding numeric total (e.g. BLK% dropped and kept BLK)
# G: minutes better measure of time contributed
# USG% (.86), PER (.78), WS (.56) highly correlated with points 'PER','WS','VORP'

# dropped, but could be useful
# Pos: could be useful - not a continuous variable
adv_per36 = adv_stats.merge(per_36, how = 'inner').drop(['Rk',\
                                                         'Pos','Tm','ORB%','TRB%','DRB%','3P','2P','2PA',\
                                                         '3PA','3PAr','FGA','FG','FTA','FT','ORB','DRB','BPM',\
                                                         'OBPM','DBPM','TOV%','BLK%','STL%','AST%','G','GS','OWS',\
                                                         'USG%','WS/48','FG%'], axis = 1)

# clean up player names in First Last format
index = 0
for i in adv_per36['Player']:
    i = i.split("\\")[0]
    adv_per36.iloc[index,0] = i
    index += 1
    
# for players with multiple rows due to switching teams mid-season, groupby player and use average values over the season
adv_per36 = adv_per36.groupby('Player').mean().round(2).reset_index()
adv_per36.set_index('Player', drop = True, inplace=True)

# Check correlation between features 

colormap = plt.cm.magma
plt.figure(figsize=(30,30))

plt.title('Pearson Correlation of NBA Salary Features', y=1.05, size=15)

corr = adv_per36.astype(float).corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)

# mask upper diagonal of heatmap
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr,\
            mask = mask, linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

# Use this cell for Bball Ref salary data (already removed rookie and non year-long contracts)

salaries_2017 = pd.read_csv('salary-data/average-salaries/2017-bballref.csv')
salaries_2017['Player'] = salaries_2017['Player'].apply(str)

index = 0

for i in salaries_2017['Player']:
    i = i.split("\\")[0]
    salaries_2017.iloc[index,1] = i
    index += 1

salaries_2017.set_index('Player',inplace=True)
salaries_2017 = salaries_2017[['Average Salary']]

# bin salaries by the million. label 0 = min, label 18 = >18M (i.e. max)
bins = [0,999999,1999999,2999999,3999999,4999999,5999999,6999999,7999999,8999999,9999999,10999999,11999999,12999999,13999999,14999999,15999999,16999999,17999999,salaries_2017.max()]
labels = []
for i in range(19):
    labels.append(i)

# bin salaries by the million. max cutoff at $18M, as this is generally the min of max contracts, where variance between maxes does is not indicative of skill level
salaries_2017 = pd.cut(salaries_2017['Average Salary'], bins = bins, labels = labels)
salaries_2017 = pd.DataFrame(salaries_2017)


# merge statistics and salary databases for players who have valid salaries from 2017

salary_statistics_2017 = adv_per36.merge(salaries_2017, how = 'inner', left_index = True, right_index = True)
salary_statistics_2017.head()

# 24 players NaN 3P%, 2 players NaN FT%. assume these players provide no value in these fields - fill with min from sample
print(salary_statistics_2017.isnull().sum())

salary_statistics_2017['3P%'].fillna(salary_statistics_2017['3P%'].min(), inplace = True)
salary_statistics_2017['FT%'].fillna(salary_statistics_2017['FT%'].min(), inplace = True)


standardize = StandardScaler()

statistics_no_salary = salary_statistics_2017.drop('Average Salary', axis = 1)
# split salary into training and test sets of features (stats) and predictions (salaries)
X_train, X_test, Y_train, Y_test = train_test_split(statistics_no_salary,\
                                                   salary_statistics_2017['Average Salary'],train_size = .8, test_size =.2,random_state=2)

# standardize data (min-max) - more accurate than normalizing
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# normalize data

# normalize = preprocessing.normalize
# X_train = normalize(X_train)
# X_test = normalize(X_test)

#KNN 

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)

knn_train_accuracy = knn.score(X_train, Y_train)
knn_test_accuracy = knn.score(X_test, Y_test)
print(knn_train_accuracy, knn_test_accuracy)
Y_pred = knn.predict(X_test)

Y_comparison = pd.DataFrame([Y_test])
Y_comparison = Y_comparison.T
Y_comparison['Predicted Salary'] = Y_pred
Y_comparison['Error'] = (Y_comparison['Average Salary'] - Y_comparison['Predicted Salary'])
Y_comparison

# SVM

svc = SVC(coef0 = 3)
svc.fit(X_train, Y_train)
svc_train_accuracy = svc.score(X_train, Y_train)
svc_test_accuracy = svc.score(X_test, Y_test)

print('SVC training accuracy:', svc_train_accuracy, '\nSVC Forest testing accuracy:', svc_test_accuracy)

# Random Forest

random_forest = RandomForestClassifier(n_estimators=1200)
random_forest.fit(X_train, Y_train)
rf_train_accuracy = random_forest.score(X_train, Y_train)
rf_test_accuracy = random_forest.score(X_test, Y_test)

print('Random Forest training accuracy:', rf_train_accuracy, '\nRandom Forest testing accuracy:', rf_test_accuracy)

