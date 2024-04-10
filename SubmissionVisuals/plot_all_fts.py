'''
Plot all features
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df_all = pd.read_csv('./Features_ManualRS.csv')

df_all = df_all.drop(['Unnamed: 0'], axis=1)
fts = df_all['Feature'].unique()
patIDs = df_all['PatID'].unique()


# Plot all features


for ft in fts:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for pat in patIDs:
        df_temp = df_all[df_all['PatID'] == pat]
        df_temp = df_temp[df_temp['Feature'] == ft]
        plt.plot(df_temp['Fraction'], df_temp['FeatureValue'], alpha=0.5, color='gray')
    
    sns.lineplot(x='Fraction', y='FeatureValue', data=df_all[df_all['Feature'] == ft], color='blue', estimator='mean', ci=95)
    plt.title(ft, fontsize=20)
    plt.xlabel('Fraction', fontsize=16)
    plt.ylabel('Feature Value', fontsize=16)
    plt.xticks(df_all['Fraction'].unique(), fontsize=12)
    plt.yticks(fontsize=12)

    
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('grey')

    plt.savefig(f'./SubmissionVisuals/SignalPlot/{ft}.png')
    

