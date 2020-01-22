# -*- coding: utf-8 -*-
"""
Use Linear Models, that were trained on using All_Seasons_Data.csv data, on 
the 2019 data for qualified batters.

The models use three variables (Off, Def, BSR) to predict what a batter's WAR
would be in 2019.

Created on Wed Dec 25 10:31:08 2019

@author: afs95
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bfuncs import clean_pcts
import seaborn as sns
import pickle
from sklearn.metrics import (r2_score, mean_squared_error,
                             explained_variance_score)

# load 2019 data
data = pd.read_csv('./data/2019/EndofSeason.csv')
# clean BB% and K% data
data['BB%'] = data['BB%'].apply(clean_pcts)
data['K%'] = data['K%'].apply(clean_pcts)

# create FacetGrid to plot Off and Def runs with color based on WAR
plt.figure(figsize=(10,10))
fg = sns.FacetGrid(data=data, hue='WAR')
fg.map(plt.scatter, 'Off', 'Def')
plt.show()

# get independent variables and dependent variable
X_test = data[['Off', 'Def', 'BsR']]
y_test = data['WAR']

# import the trained models
trained_models = pickle.load(open('./models/models.pickle', 'rb'))
predictions = {}
for name, model in trained_models.items():
    predictions[name] = model.predict(X_test)

# gather metrics    
dypreds = {}
MSEs = {}
R2s = {}
exp_vars = {}
metric_names = ['dypreds', 'MSEs', 'R2s', 'exp_vars']
metric_dicts = [dypreds, MSEs, R2s, exp_vars]
for name, preds in predictions.items():
    MSEs[name] = mean_squared_error(y_test, preds)
    dypreds[name] = 1.96 * np.sqrt(MSEs[name]) # 95% confidence intervals
    R2s[name] = r2_score(y_test, preds)
    exp_vars[name] = explained_variance_score(y_test, preds)

# put metrics all in one dict
metrics = {name: mets for name, mets in zip(metric_names, metric_dicts)}

# plot true WAR vs Predicted WAR
fig, ax = plt.subplots(3,3, figsize=(10, 10))
for i, (name, preds) in enumerate(predictions.items()):
    axi = ax.flat[i]
    axi.scatter(y_test, predictions[name])
    axi.set_title(name)
plt.show()

# write metrics to a .txt file
f_name = input("What would you like to name the results file?\n>>> ")
with open('./results/%s.txt' % f_name, 'w') as f:
    for name, preds in trained_models.items():
        f.write(name)
        f.write('\n\nMean Square Error: {}'.format(np.round(MSEs[name], 4)))
        f.write('\nR^2: {}'.format(np.round(R2s[name], 4)))
        f.write('\nexplained_var_score: {}'.format(np.round(exp_vars[name], 4)))
        f.write('\n\n{}\n\n'.format('*'*25))