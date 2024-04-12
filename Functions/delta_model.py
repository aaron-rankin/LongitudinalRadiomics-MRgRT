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
    Selects features based on the correlation matrix
    '''

    features = df_corr.index.values
    array_corr = df_corr.values[:,1:]
    
    print("Selecting Features...")
    results = np.zeros((len(features), len(features)))
    selected_features = []

    array_corr = df_corr.values[:,1:]
    array_corr = abs(array_corr)

    results = np.zeros((len(features), len(features)))
    selected_features = []

    for i in tqdm(range(len(features)-1)):
        for j in range(len(features)-1):
            if array_corr[i,j] > 0.8:
                results[i,j] = array_corr[i,j]
                selected_features.append([features[i], features[j]])
            else:
                results[i,j] = array_corr[i,j]

    df_res = pd.DataFrame(results, columns=features, index=features)
    
    # # create a dataframe to store the results
    df_res = pd.DataFrame(results, columns=features, index=features)
    # heatmap of results
    plt.figure(figsize=(20,20))
    sns.heatmap(df_res, cmap="RdBu_r", vmin=0, vmax=0.8, square=True)
    plt.savefig(outdir + "/CM/CorrMatrix-PreSelection.png")
    plt.close()

    # loop through results with ft pairs and select ft with lowest mean value
    # create an empty array to store the results
    features_keep = []
    features_remove = []

    for i in range(len(selected_features)):
        # select ft pair
        df_ft1 = df_corr[df_corr.index == selected_features[i][0]]
        df_ft2 = df_corr[df_corr.index == selected_features[i][1]]
        #df_ft1 = df_corr[df_corr. == selected_features[i][0]]
        # df_ft2 = df_corr[df_corr["Unnamed: 0"] == selected_features[i][1]]
        # get mean values across row
        mean_ft1 = float(df_ft1.mean(axis=1))
        mean_ft2 = float(df_ft2.mean(axis=1))

        # compare mean values
        if mean_ft1 < mean_ft2:
            features_keep.append(selected_features[i][0])
            features_remove.append(selected_features[i][1])
        else:
            features_keep.append(selected_features[i][1])
            features_remove.append(selected_features[i][0])

    # remove duplicates
    features_keep = list(dict.fromkeys(features_keep))
    features_remove = list(dict.fromkeys(features_remove))

    # compare lists
    features_keep2 = [x for x in features_keep if x not in features_remove]

    # save results
    df_features = pd.DataFrame(features_keep2, columns=["Feature"])
    df_features.to_csv(outdir + "/features/Features_Selected.csv")

    print("-" * 30)    
    print("Selected Features: ")
    for ft in features_keep2:
        print(ft)
    print("-" * 30)
    print("Number of Selected Features: {}".format(len(features_keep2)))
    print("-"*30)

    return None

####################################################

def CorrMatrix(df, Rescale, outdir, plot=False):
    features = df["Feature"].unique()
    
    df_res = pd.DataFrame()

    print("Calculating Delta Correlation pairs...")

    if os.path.isdir(outdir + "/CM/") == False:
        os.mkdir(outdir + "/CM/")


    if Rescale == True:
        print("Rescaling Features...")
        df = Rescalefeatures(df)

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

def FeatureSelection2(df, outdir):
    # read in fts from csv

    df_corr = pd.read_csv(outdir + "CM\\CorrMatrix.csv")
    fts = df['Feature'].unique()
    array_corr = df_corr.values[:,1:]
    array_corr = abs(array_corr)

    results = np.zeros((len(fts), len(fts)))
    selected_fts = []

    for i in tqdm(range(len(fts))):
        for j in range(len(fts)):
            
            if array_corr[i,j] <= 0.8:
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

    # heatmap of results
    plt.figure(figsize=(20,20))
    sns.heatmap(df_res, cmap="RdBu_r", vmin=0, vmax=0.8, square=False)
    #plt.show()

    fts_keep = []
    fts_remove = []
     
    for i in range(len(selected_fts)):
        # select ft pair
        df_ft1 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][0]]
        df_ft2 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][1]]
        # get mean values across row
        mean_ft1 = float(df_ft1.mean(axis=1))
        mean_ft2 = float(df_ft2.mean(axis=1))

        # # compare mean values
        # if mean_ft1 > mean_ft2:
        #     fts_keep.append(selected_fts[i][0])
        #     fts_remove.append(selected_fts[i][1])
        # else:
        #     fts_keep.append(selected_fts[i][1])
        #     fts_remove.append(selected_fts[i][0])
        



    # remove duplicates
    fts_keep = list(dict.fromkeys(fts_keep))
    fts_remove = list(dict.fromkeys(fts_remove))

    # compare lists
    fts_keep2 = [x for x in fts_keep if x not in fts_remove]

    # save results
    df_fts = pd.DataFrame(fts_keep2, columns=["Feature"])
    print('-'*30)
    print("Selected Features: ({})".format(str(len(fts_keep2))))
    fts = df_fts["Feature"].values
    for ft in fts:
        print(ft)
    df_fts.to_csv(outdir + "features\\Features_Selected.csv", index=False)

####################################################

def remove_highly_correlated_features(correlation_matrix, threshold=0.6):
    """
    Remove one of each highly correlated feature pair based on the average absolute correlation coefficient.
    
    Parameters:
    - features: List or array containing feature names.
    - correlation_matrix: 2D array or DataFrame containing correlation coefficients between features.
    - threshold: Threshold value for considering features as highly correlated (default: 0.8).
    
    Returns:
    - List of selected features after removing highly correlated features.
    """
    # Create a mask to exclude upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    features = correlation_matrix.columns

    mask = correlation_matrix.mask(np.triu(pd.DataFrame(True, index=correlation_matrix.index, columns=correlation_matrix.columns), k=1))
    
    # Initialize list to store indices of features to remove
    features_to_remove = []
    features_to_keep = [] 
    
    # Iterate over each pair of features
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if mask.iloc[i, j]:  # Check if the pair has not been processed yet
                correlation_coefficient = correlation_matrix.iloc[i, j]
                
                # If correlation coefficient is above the threshold
                if abs(correlation_coefficient) >= threshold:
                    # Determine which feature to remove based on average absolute correlation coefficient
                    avg_corr_i = correlation_matrix.iloc[i, :].abs().mean()
                    avg_corr_j = correlation_matrix.iloc[j, :].abs().mean()
                    
                    # Remove the feature with larger average absolute correlation coefficient
                    if avg_corr_i > avg_corr_j:
                        features_to_remove.append(features[j])
                        
                        if features[i] not in features_to_keep:
                            features_to_keep.append(features[i])

                    else:
                        features_to_remove.append(features[i])
                        
                        if features[j] not in features_to_keep:
                            features_to_keep.append(features[j])

    # Selected features are what remain after removing highly correlated features
    #selected_features = [feat for idx, feat in enumerate(features) if idx not in features_to_remove]
    
    selected_features = features_to_keep
    
    # if in fts to remove, remove from selected fts
    selected_features = [x for x in selected_features if x not in features_to_remove]

    print("Selected Features: ({})".format(str(len(selected_features))))
    df_features = pd.DataFrame(selected_features, columns=["Feature"])
    
    for ft in selected_features:
            print(ft)
    return selected_features

####################################################
def FeatureSelection(df, outdir):
    # read in features from csv
    features = df.index.values
    df_corr = pd.read_csv(outdir + "CM\\CorrMatrix.csv")
    array_corr = df_corr.values[:,1:]
    array_corr = abs(array_corr)

    results = np.zeros((len(features), len(features)))
    selected_features = []

    for i in tqdm(range(len(features))):
        for j in range(len(features)):
            
            if array_corr[i,j] <= 0.6:
                results[i,j] = array_corr[i,j]
                selected_features.append([features[i], features[j]])
            else:
                results[i,j] = 1

    df_res = pd.DataFrame(results, columns=features, index=features)
    # plt.figure(figsize=(20,20))
    # sns.heatmap(df_res, cmap="RdBu_r", vmin=0, vmax=0.5, square=False)
    # plt.show()

    # loop through results with ft pairs and select ft with lowest mean value
    # create an empty array to store the results
    features_keep = []
    features_remove = []
    for i in range(len(selected_features)):
        # select ft pair
        df_ft1 = df_corr[df_corr["Unnamed: 0"] == selected_features[i][0]]
        df_ft2 = df_corr[df_corr["Unnamed: 0"] == selected_features[i][1]]
        # get mean values across row
        mean_ft1 = float(df_ft1.mean(axis=1))
        mean_ft2 = float(df_ft2.mean(axis=1))

        # compare mean values
        if mean_ft1 < mean_ft2:
            features_keep.append(selected_features[i][0])
            features_remove.append(selected_features[i][1])
        else:
            features_keep.append(selected_features[i][1])
            features_remove.append(selected_features[i][0])
        
    # remove duplicates
    features_keep = list(dict.fromkeys(features_keep))
    features_remove = list(dict.fromkeys(features_remove))

    # compare lists
    features_keep2 = [x for x in features_keep if x not in features_remove]

    # save results
    df_features = pd.DataFrame(features_keep2, columns=["Feature"])
    if output == True:
        print("Selected Features: ({})".format(str(len(features_keep2))))
        features = df_features["Feature"].values
        for ft in features:
            print(ft)
    df_features.to_csv(outdir + "Features\\Delta_SelectedFeatures.csv", index=False)

# ####################################################

def FeatureSelection3(df, outdir):
    # read in fts from csv

    df_corr = pd.read_csv(outdir + "CM\\CorrMatrix.csv")
    fts = df['Feature'].unique()
    array_corr = df_corr.values[:,1:]
    print(array_corr)

    array_corr = abs(array_corr)

    results = np.zeros((len(fts), len(fts)))
    mask = np.triu(np.ones_like(results, dtype=bool), k=1)
    selected_fts = []

    threshold = 0.8

    for i in range(len(fts)):
        for j in range(i+1, len(fts)):
            if mask[i, j]:
                corr_coeff = array_corr[i, j]

                if corr_coeff > threshold:
                    mean_corr_i = np.mean(array_corr[i, :])
                    mean_corr_j = np.mean(array_corr[j, :])

                    if mean_corr_i < mean_corr_j:
                        selected_fts.append(fts[i])
                        np.delete(array_corr, i, axis=0)
                        np.delete(array_corr, i, axis=1)
                    else:
                        np.delete(array_corr, j, axis=0)
                        np.delete(array_corr, j, axis=1)
                        selected_fts.append(fts[j])
    # selected features are what remain
    print(selected_fts)

    
    
    
#     for i in tqdm(range(len(fts))):
#         for j in range(len(fts)):
            
#             if array_corr[i,j] <= 0.8:
#                 results[i,j] = array_corr[i,j]
#                 selected_fts.append([fts[i], fts[j]])
#             else:
#                 results[i,j] = 1

#     df_res = pd.DataFrame(results, columns=fts, index=fts)
#     # plt.figure(figsize=(20,20))
#     # sns.heatmap(df_res, cmap="RdBu_r", vmin=0, vmax=0.5, square=False)
#     # plt.show()

#     # loop through results with ft pairs and select ft with lowest mean value
#     # create an empty array to store the results
    
#     # heatmap of results
#     plt.figure(figsize=(20,20))
#     sns.heatmap(df_res, cmap="RdBu_r", vmin=0, vmax=0.8, square=False)
#     #plt.show()

#     fts_keep = []
#     fts_remove = []
     
#     for i in range(len(selected_fts)):
#         # select ft pair
#         df_ft1 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][0]]
#         df_ft2 = df_corr[df_corr["Unnamed: 0"] == selected_fts[i][1]]
#         # get mean values across row
#         mean_ft1 = float(df_ft1.mean(axis=1))
#         mean_ft2 = float(df_ft2.mean(axis=1))

#         # # compare mean values
#         # if mean_ft1 > mean_ft2:
#         #     fts_keep.append(selected_fts[i][0])
#         #     fts_remove.append(selected_fts[i][1])
#         # else:
#         #     fts_keep.append(selected_fts[i][1])
#         #     fts_remove.append(selected_fts[i][0])
        



#     # remove duplicates
#     fts_keep = list(dict.fromkeys(fts_keep))
#     fts_remove = list(dict.fromkeys(fts_remove))

#     # compare lists
#     fts_keep2 = [x for x in fts_keep if x not in fts_remove]

#     # save results
#     df_fts = pd.DataFrame(fts_keep2, columns=["Feature"])
#     print('-'*30)
#     print("Selected Features: ({})".format(str(len(fts_keep2))))
#     fts = df_fts["Feature"].values
#     for ft in fts:
#         print(ft)
#     df_fts.to_csv(outdir + "features\\Features_Selected.csv", index=False)
# # def DeltaModel(DataRoot, Norm, tag, output=False):
# #     # Make Directories if they don't exist
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

def find_correlation(x, cutoff=0.8, verbose=True, names=True, exact=None):
    """
    Find correlated variables based on a correlation matrix.

    Args:
        x (numpy.ndarray): A correlation matrix.
        cutoff (float): A numeric value for the pair-wise absolute correlation cutoff.
        verbose (bool): A boolean for printing the details.
        names (bool): A boolean indicating whether to return column names (True) or indices (False).
        exact (bool): A boolean indicating whether to recompute average correlations at each step.

    Returns:
        numpy.ndarray: A vector of indices denoting the columns to remove (when names = False)
        otherwise a vector of column names. If no correlations meet the criteria, an empty array is returned.
    """

    cor_matrix = x.values

    # if isinstance(x, pd.DataFrame):
    #     cor_matrix = np.abs(x.values)
    # elif isinstance(x, np.ndarray):
    #     cor_matrix = np.abs(x)
    # else:
    #     raise ValueError("Input must be a pandas DataFrame or a numpy array.")

    if exact is None:
        exact = x.shape[1] < 100


    cor_matrix = np.abs(cor_matrix)
    np.fill_diagonal(cor_matrix, 0)

    avg_cor = cor_matrix.mean(axis=0)

    while True:
        max_cor_idx = np.argmax(avg_cor)
        max_cor_val = avg_cor[max_cor_idx]

        if max_cor_val < cutoff:
            break

        if verbose:
            print(f"Removing column {max_cor_idx}, mean correlation: {max_cor_val}")

        cor_matrix = np.delete(np.delete(cor_matrix, max_cor_idx, axis=0), max_cor_idx, axis=1)
        avg_cor = cor_matrix.mean(axis=0)

    correlated_indices = np.where(avg_cor >= cutoff)[0]

    if names:
        return correlated_indices
    else:
        return correlated_indices.tolist()