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

# import advanced statistics and drop empty columns

years = ['2012','2013','2014','2015','2016','2017']

adv_stats = pd.read_csv('..\player-data\player2011\Adv-2011.csv')
per_36 = pd.read_csv('..\player-data\player2011\Per36-2011.csv')

for year in years:
    adv_stats = adv_stats.append(pd.read_csv('..\player-data\player' + year + '\Adv-' + year + '.csv'))
    per_36 = per_36.append(pd.read_csv('..\player-data\player' + year + '\Per36-' + year + '.csv'))
    
# drop empty columns
adv_stats = adv_stats.drop(['Unnamed: 19','Unnamed: 24'], axis = 1)

# merge adv and per 36 and drop non-metric columns (rk, player, etc.)/ redundant columns

# dropped column: explanation

#   All %'s dropped in leiu of their corresponding numeric total (e.g. BLK% dropped and kept BLK)
#   G: minutes better measure of time contributed
#   USG% (.86), PER (.78), WS (.56) highly correlated with points 'PER','WS','VORP'
#   dropped, but could be useful
#   Pos: could be useful - not a continuous variable

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
