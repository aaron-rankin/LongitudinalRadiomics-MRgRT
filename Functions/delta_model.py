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
            rho = stats.pearsonr(vals_ft1, vals_ft2)[0]

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
    Selects features based on the correlation matrix
    '''

    features = df_corr.index.values
    array_corr = df_corr.values[:,1:]
    
    print("Selecting Features...")
    
    results = np.zeros(len(features), len(features))
    selected_fts = []

    # loop through correlation matrix
    for i in tqdm(range(len(features))):
        for j in range(len(features)):
            # select features with correlation less than 0.6
            if array_corr[i,j] <= 0.8:
                results[i,j] = array_corr[i,j]
                selected_fts.append([features[i], features[j]])
            else:
                results[i,j] = 1
    
    # create a dataframe to store the results
    df_res = pd.DataFrame(results, columns=features, index=features)

    # loop through results with ft pairs and select ft with lowest mean value
    # create an empty array to store the results
    fts_keep = []
    fts_remove = []

    for i in range(len(selected_fts)):
        # select ft pair
        df_ft1 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][0]]
        df_ft2 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][1]]
        # get mean values across row
        mean_ft1 = float(df_ft1.mean(axis=1))
        mean_ft2 = float(df_ft2.mean(axis=1))

        # compare mean values
        if mean_ft1 < mean_ft2:
            fts_keep.append(selected_fts[i][0])
            fts_remove.append(selected_fts[i][1])
        else:
            fts_keep.append(selected_fts[i][1])
            fts_remove.append(selected_fts[i][0])

    # remove duplicates
    fts_keep = list(dict.fromkeys(fts_keep))
    fts_remove = list(dict.fromkeys(fts_remove))

    # compare lists
    fts_keep2 = [x for x in fts_keep if x not in fts_remove]

    # save results
    df_fts = pd.DataFrame(fts_keep2, columns=["Feature"])
    df_fts.to_csv(outdir + "/features/Features_Selected.csv")

    print("-" * 30)    
    print("Selected Features: ")
    for ft in fts_keep2:
        print(ft)
    print("-" * 30)
    print("Number of Selected Features: {}".format(len(fts_keep2)))
    print("-"*30)

####################################################

def CorrMatrix(df, Rescale, outdir, plot=False):
    features = df["Feature"].unique()
    
    df_res = pd.DataFrame()

    print("Calculating Delta Correlation pairs...")

    if os.path.isdir(outdir + "/CM/") == False:
        os.mkdir(outdir + "/CM/")


    if Rescale == True:
        print("Rescaling Features...")
        df = RescaleFts(df)

    matrix = np.zeros((len(features), len(features)))


    for i, ft1 in tqdm(enumerate(features)):
        vals_ft1 = df[df["Feature"] == ft1]["FeatureValue"].values
        for j, ft2 in enumerate(features):
            vals_ft2 = df[df["Feature"] == ft2]["FeatureValue"].values
            rho = stats.pearsonr(vals_ft1, vals_ft2)[0]

            matrix[i,j] = rho


    df_res = pd.DataFrame(matrix, index=features, columns=features)
    df_res.to_csv(outdir + "/CM/CorrMatrix.csv")


####################################################

def FeatureSelection(df, outdir):
    # read in fts from csv
    fts = df.index.values
    df_corr = pd.read_csv(outdir + "CM\\CorrMatrix.csv")
    array_corr = df_corr.values[:,1:]
    array_corr = abs(array_corr)

    results = np.zeros((len(fts), len(fts)))
    selected_fts = []

    for i in tqdm(range(len(fts))):
        for j in range(len(fts)):
            
            if array_corr[i,j] <= 0.6:
                results[i,j] = array_corr[i,j]
                selected_fts.append([fts[i], fts[j]])
            else:
                results[i,j] = 1

    df_res = pd.DataFrame(results, columns=fts, index=fts)
    # plt.figure(figsize=(20,20))
    # sns.heatmap(df_res, cmap="RdBu_r", vmin=0, vmax=0.5, square=False)
    # plt.show()

    # loop through results with ft pairs and select ft with lowest mean value
    # create an empty array to store the results
    fts_keep = []
    fts_remove = []
    for i in range(len(selected_fts)):
        # select ft pair
        df_ft1 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][0]]
        df_ft2 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][1]]
        # get mean values across row
        mean_ft1 = float(df_ft1.mean(axis=1))
        mean_ft2 = float(df_ft2.mean(axis=1))

        # compare mean values
        if mean_ft1 < mean_ft2:
            fts_keep.append(selected_fts[i][0])
            fts_remove.append(selected_fts[i][1])
        else:
            fts_keep.append(selected_fts[i][1])
            fts_remove.append(selected_fts[i][0])
        
    # remove duplicates
    fts_keep = list(dict.fromkeys(fts_keep))
    fts_remove = list(dict.fromkeys(fts_remove))

    # compare lists
    fts_keep2 = [x for x in fts_keep if x not in fts_remove]

    # save results
    df_fts = pd.DataFrame(fts_keep2, columns=["Feature"])
    if output == True:
        print("Selected Features: ({})".format(str(len(fts_keep2))))
        fts = df_fts["Feature"].values
        for ft in fts:
            print(ft)
    df_fts.to_csv(outdir + "Features\\Delta_SelectedFeatures.csv", index=False)

# ####################################################

# def DeltaModel(DataRoot, Norm, tag, output=False):
#     # Make Directories if they don't exist
#     print("------------------------------------")
#     print("------------------------------------")
#     print("Root: {} Norm: {} Tag: {}".format(DataRoot, Norm, tag))
#     uf.CD(DataRoot, "No", Norm, tag)

#     print("------------------------------------\n")
#     print("             Delta Model            \n")
#     print("------------------------------------\n")

#     # Get Delta Features
#     if output == True:
#         print("------------------------------------")
#         print("------------------------------------")
#         print("Calculating Delta Features...")
#         print("------------------------------------")
#     FE.DeltaValues(DataRoot, Norm, tag)

#     # Feature Reduction
#     if output == True:
#         print("------------------------------------")
#         print("------------------------------------")
#         print("Reducing Features...")
#         print("------------------------------------")
#         print("ICC Feature Reduction: ")
#         print("------------------------------------")
#     FR.ICC(DataRoot, Norm, "Delta", tag, output)
#     FR.Volume(DataRoot, Norm, "Delta", tag, output)
#     if output == True:
#         print("------------------------------------")
#         print("------------------------------------\n ")
#         print("------------------------------------")
#         print("------------------------------------")
#         print("Feature Selection...")
#         print("------------------------------------")
#         print("Creating Correlation Matrix:")
#         print("------------------------------------")
#     CorrMatrix(DataRoot, Norm, tag,)
#     if output == True:
#         print("------------------------------------")
#         print("Feature Selection:")
#         print("------------------------------------")
#     FeatureSelection(DataRoot, Norm, tag, output)
#     print("------------------------------------")
#     print("------------------------------------\n ")

# ####################################################
