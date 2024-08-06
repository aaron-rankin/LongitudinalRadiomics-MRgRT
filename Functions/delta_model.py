import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from scipy import stats

####################################################

def correlation_matrix(df, outdir, plot=True):
    '''
    caclulates the correlation matrix between delta values of features 
    using the pearson correlation coefficient
    '''

    print('Calculating Correlation Matrix...')

    features = df["Feature"].unique()

    df_corr = pd.DataFrame()

    # make directory for correlation matrix
    if os.path.isdir(outdir + "/CM/") == False:
        os.mkdir(outdir + "/CM/")
    
    # create an empty matrix to store the results
    matrix = np.zeros((len(features), len(features)))

    # loop through features
    for i, ft1 in tqdm(enumerate(features)):
        vals_ft1 = df[df["Feature"] == ft1]["FeatureValue"].values
        for j, ft2 in enumerate(features):
            vals_ft2 = df[df["Feature"] == ft2]["FeatureValue"].values
            rho = stats.spearmanr(vals_ft1, vals_ft2)[0]

            matrix[i,j] = abs(rho)
    
    # save results to csv
    df_corr = pd.DataFrame(matrix, index=features, columns=features)
    df_corr.to_csv(outdir + "/CM/CorrMatrix.csv")

    if plot == True:
        plt.figure(figsize=(10,10))
        sns.heatmap(df_corr, cmap="RdBu_r", vmin=0, vmax=1, square=False)
        plt.savefig(outdir + "/CM/CorrMatrixPlot.png")
        plt.show()

    return df_corr

####################################################
def feature_selection(df_corr, outdir):
    '''
    Selects features based on corr matrix
    If corr > 0.95 between 2 features, keep ft with lowest mean corr
    across all other features
    '''
    fts_keep = []
    fts_remove = []

    for i in range(len(df_corr.columns)):
        for j in range(i+1, len(df_corr.columns)):
            if i != j:
                if df_corr.iloc[i, j] >= 0.95:
                    # get meean of each feature
                    
                    mean1 = np.mean(df_corr[df_corr.columns[i]])
                    mean2 = np.mean(df_corr[df_corr.columns[j]])
                    # keep lowest mean
                    if mean1 < mean2:
                        fts_keep.append(df_corr.columns[i])
                        fts_remove.append(df_corr.columns[j])
                    else:
                        fts_keep.append(df_corr.columns[j])
                        fts_remove.append(df_corr.columns[i])

    # remove duplicates
    fts_keep = list(set(fts_keep))
    fts_remove = list(set(fts_remove))

    # remove features from fts_keep
    for ft in fts_remove:
        if ft in fts_keep:
            fts_keep.remove(ft)

    print('-'*30)
    print(f'Selected Features: ({len(fts_keep)})')
    for ft in fts_keep:
        print(ft)

    df_fts = pd.DataFrame(fts_keep, columns=["Feature"])
    df_fts.to_csv(outdir + "/features/Features_Selected.csv", index=False)

    return None

####################################################
