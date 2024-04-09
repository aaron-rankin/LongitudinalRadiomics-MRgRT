import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import scipy.cluster.hierarchy as spch
import statsmodels.tsa.stattools as sts
from scipy import stats
from Functions import useful_functions as uf
from scipy.spatial import distance

####################################################

def distance_matrices(df, outdir, plot=True):
    '''
    Calculates the Euclidean distance between feature pair trajectories
    for each patient.
    Saves the distance matrix for each patient to a csv file.
    
    df: dataframe with all feature values across treatment for one region
    out_dir: output
    tag: tag for output to denote any changes
    output: print output
    plot: saves a heatmap of the distance matrix
    '''
    
    features = df["Feature"].unique()
    PatIDs = df["PatID"].unique()

    df_res = pd.DataFrame()

    print("Calculating Euclidean distance between feature pair trajectories...")

    if os.path.isdir(outdir + "/DM/") == False:
        os.mkdir(outdir + "/DM/")
        os.mkdir(outdir + "/DM/data/")
        os.mkdir(outdir + "/DM/figs/")
    
    for pat in tqdm(PatIDs):
        df_pat = df[df["PatID"] == pat]

        matrix = np.zeros((len(features), len(features)))

        for i, ft1 in enumerate(features):
            df_ft = df_pat[df_pat["Feature"] == ft1]
            vals1 = df_ft["FeatureValue"].values
            if vals1[0] == 0:
                vals1[0] = 1
            vals1_ch = (vals1 - vals1[0]) / vals1[0]
            for j, ft2 in enumerate(features):
                df_ft2 = df_pat[df_pat["Feature"] == ft2]
                vals2 = df_ft2["FeatureValue"].values
                if vals2[0] == 0:
                    vals2[0] = 1
                
                vals2_ch = (vals2 - vals2[0]) / vals2[0]
            
                # get euclidean distance
                # fill nan with 0
                if np.isnan(vals1_ch).any() == True:
                    print(pat)
                    print(ft1, vals1)
                if np.isnan(vals2_ch).any() == True:
                    print(pat)
                    print(ft2, vals2)
                
                matrix[i,j] = distance.euclidean(vals1, vals2)
    
        df_dist = pd.DataFrame(matrix, columns=features, index=features)
        df_dist.to_csv(outdir + "/DM/data/" + str(pat) + ".csv")

        if plot == True:
            plt.figure(figsize=(10,10))
            sns.heatmap(df_dist, cmap="viridis", vmin=0, vmax=2.5)
            plt.title(str(pat), fontsize=20)
            # make sure all ticks show

            plt.xticks(np.arange(len(features)) + 0.5, features, fontsize=6)
            plt.yticks(np.arange(len(features)) + 0.5, features, fontsize=6)
            
            plt.savefig(outdir + "/DM/figs/" + str(pat) + ".png")
            plt.close()

# ####################################################

# def RescaleFts(df):
#     '''
#     Rescale features to be between -1 and 1 across all patients
#     '''
    
#     df = df.copy()

#     # Get the features
#     fts = df["Feature"].unique()
#     for ft in fts:
#         # Get the feature
#         df_ft = df.loc[df["Feature"] == ft]
#         # Get the values
#         vals = df_ft["FeatureValue"].values
#         vals = MinMaxScaler(feature_range=(0,1)).fit_transform(vals.reshape(-1,1))
#         # Replace
#         df.loc[df["Feature"] == ft, "FeatureValue"] = vals

#     return df

# ####################################################

# def DistanceMatrix_Delta(df, Rescale, outdir, plot=False):
#     '''
#     Calculates the Euclidean distance between feature pair trajectories
#     for each patient.
#     Saves the distance matrix for each patient to a csv file.
    
#     df: dataframe with all feature values across treatment for one region
#     out_dir: output
#     tag: tag for output to denote any changes
#     output: print output
#     plot: saves a heatmap of the distance matrix
#     '''
    
#     features = df["Feature"].unique()
#     PatIDs = df["PatID"].unique()

#     df_res = pd.DataFrame()

#     print("Calculating Euclidean distance between feature pair trajectories...")

#     if os.path.isdir(outdir + "/DM/") == False:
#         os.mkdir(outdir + "/DM/")
#         os.mkdir(outdir + "/DM/data/")
#         os.mkdir(outdir + "/DM/figs/")

#     if Rescale == True:
#         print("Rescaling Features...")
#         df = RescaleFts(df)
    
#     for pat in tqdm(PatIDs):
#         df_pat = df[df["PatID"] == pat]

#         matrix = np.zeros((len(features), len(features)))

#         for i, ft1 in enumerate(features):
#             df_ft = df_pat[df_pat["Feature"] == ft1]
#             vals1 = df_ft["FeatureValue"].values

#             for j, ft2 in enumerate(features):
#                 df_ft2 = df_pat[df_pat["Feature"] == ft2]
#                 vals2 = df_ft2["FeatureValue"].values
            
#                 # get euclidean distance
                
#                 matrix[i,j] = distance.euclidean(vals1, vals2)
    
#         df_dist = pd.DataFrame(matrix, columns=features, index=features)
#         df_dist.to_csv(outdir + "/DM/data/" + str(pat) + ".csv")

#         if plot == True:
#             plt.figure(figsize=(10,10))
#             sns.heatmap(df_dist, cmap="viridis")
#             plt.title(str(pat), fontsize=20)
#             # make sure all ticks show
#             plt.xticks(np.arange(len(features)) + 0.5, features, fontsize=6)
#             plt.yticks(np.arange(len(features)) + 0.5, features, fontsize=6)
            

#             plt.savefig(outdir + "/DM/figs/" + str(pat) + ".png")
#             plt.close()

# ####################################################
def check_recluster(df, fts, t_val, tries, df_DM):
        '''
        If cluster has more than 10 features, re-cluster with smaller t_val
        '''
        df_c = df
        df_new = pd.DataFrame()
        # feature names
        df_new["Feature"] = fts
        # cluster labels
        c = df_c["ClusterLabel"].values[0]
        
        # need to filter distance matrix to only include features in cluster
        df_DM_c = df_DM[fts]
        # only keep features in cluster
        df_DM_c = df_DM_c[df_DM_c.index.isin(fts)]
        
        # convert to numpy array
        arr_DM_c = df_DM_c.to_numpy()
        
        # cluster
        df_new["ClusterLabel"] = spch.fclusterdata(arr_DM_c, t=t_val, criterion="distance", method="weighted")
        df_new["ClusterLabel"] = str(c*100) + str(tries) + df_new["ClusterLabel"].astype(str)
        df_new["ClusterLabel"] = df_new["ClusterLabel"].astype(int)
        df_new["ClusterNumFts"] = df_new.groupby("ClusterLabel")["ClusterLabel"].transform("count")
        number_fts = df_new["ClusterNumFts"].unique()
        fts_check = df_new.loc[df_new["ClusterNumFts"] > 8]["Feature"].values
        #print(t_val, number_fts)#, df_new)
        return number_fts, df_new, fts_check

####################################################

def cluster_features(df, outdir, s_t_val, method="weighted"):
    '''
    Cluster features using distance matrix, 
    t_val is threshold for clustering, 
    method is clustering forumula
    performs clustering until all clusters have less than 10 features
    '''
    print("-"*30)
    print("Clustering Feature Trajectories...")
    
    DM_dir = outdir + "/DM/data/"
    
    outpath = outdir + "/Clustering/"
    if os.path.exists(outpath) == False:
        os.mkdir(outpath)
        os.mkdir(outpath + "/Labels/")
        os.mkdir(outpath + "/Plots")

    df["ClusterLabel"] = ""
    df["ClusterNumFts"] = ""
    patIDs = df["PatID"].unique()
    fts = df["Feature"].unique()

    for pat in tqdm(patIDs):
        df_DM = pd.read_csv(DM_dir + str(pat) + ".csv")
        df_DM.set_index("Unnamed: 0", inplace=True)
        arr_DM = df_DM.to_numpy()
        fts = df_DM.columns

        # create temp df to hold ft name and label
        df_labels = pd.DataFrame()
        df_labels["Feature"] = fts

        # cluster function using DM, need to experiment with t_val and method
        df_labels["ClusterLabel"] = spch.fclusterdata(arr_DM, t=s_t_val, criterion="distance", method=method)
        df_labels.set_index("Feature", inplace=True)
        
        # check number of features in each cluster
        df_labels["ClusterNumFts"] = df_labels.groupby("ClusterLabel")["ClusterLabel"].transform("count")
        df_labels["ClusterLabel"] = df_labels["ClusterLabel"].astype(int)
        
        # loop through clusters 
        for c in df_labels["ClusterLabel"].unique():
                df_c = df_labels[df_labels["ClusterLabel"] == c]
                number_fts = len(df_c)
                # check numnber of features in cluster
                if number_fts > 8:
                        # if more than 10 features in cluster, reduce t_val and recluster
                        t_val = s_t_val - 0.2
                        check_fts = df_c.index.values
                        tries = 1
                        number_fts, df_labels2, check_fts = check_recluster(df_c, check_fts, t_val, tries, df_DM)
                        new_fts = df_labels2["Feature"].unique()
                        df_labels.loc[new_fts, "ClusterLabel"] = df_labels2["ClusterLabel"].values
                        df_labels["ClusterNumFts"] = df_labels.groupby("ClusterLabel")["ClusterLabel"].transform("count")

                        while number_fts.max() > 8:
                                t_val = t_val - 0.2
                                tries += 1
                                #print("Cluster: {} Tries: {} T_val: {}".format(c, tries, t_val))
                                number_fts, df_labels2, check_fts = check_recluster(df_c, check_fts, t_val, tries, df_DM)
                                new_fts = df_labels2["Feature"].unique()
                                df_labels.loc[new_fts, "ClusterLabel"] = df_labels2["ClusterLabel"].values
                        
        df_labels["ClusterNumFts"] = df_labels.groupby("ClusterLabel")["ClusterLabel"].transform("count")
        
        df_labels.to_csv(outpath + "/Labels/" + str(pat) + ".csv")




        # read in df with ft vals and merge
        # ft_vals = pd.read_csv(root +"Aaron\\ProstateMRL\\Data\\Paper1\\"+ Norm + "\\Features\\Longitudinal_All_fts_" + tag + ".csv")
        # ft_vals["PatID"] = ft_vals["PatID"].astype(str)
        # pat_ft_vals = ft_vals[ft_vals["PatID"] == pat]
        # pat_ft_vals = pat_ft_vals.merge(df_labels, left_on="Feature", right_on="FeatureName")

        # output is feature values w/ cluster labels
        # pat_ft_vals.to_csv(out_dir + pat + "_" + tag + ".csv")
    print("-" *30)
####################################################

def count_clusters(outdir):
    '''
    Summarises clustering results
    '''
    outdir = outdir + "/Clustering/Labels/"
    patIDs = os.listdir(outdir)
    df_result = pd.DataFrame()

    for pat in patIDs:

        df = pd.read_csv(outdir + pat)
        df = df[["Feature", "ClusterLabel"]]
        df = df.drop_duplicates()
        # sort by cluster
        df = df.sort_values(by=["ClusterLabel"])
        # turn value counts into a dataframe
        df = df["ClusterLabel"].value_counts().rename_axis("ClusterLabel").reset_index(name="Counts")
        # set PatID as index
        df["PatID"] = pat
        df.set_index("PatID", inplace=True)
            
        # append to result
        df_result = df_result.append(df, ignore_index=False)
    #get number of clusters with more than 3 features
    df_stable = df_result[df_result["Counts"] > 3]
    df_stable = df_stable.groupby("PatID")["ClusterLabel"].count()
    # get mean number of stable clusters
    meanstable = df_stable.mean()
    #print(df_result)
    df_numclust= df_result.groupby("PatID")["ClusterLabel"].count()
    #print(df_numclust)
    df_numclust = df_numclust.rename_axis("PatID").reset_index(name="NumClusters")
    #print(df_numclust)
    # group by patient and get mean number of clusters
    df_numfts = df_result.groupby("PatID")["Counts"].mean()
    df_numfts = df_numfts.rename_axis("PatID").reset_index(name="MeanFeaturesperCluster")
    df_medianfts = df_result.groupby("PatID")["Counts"].median()
    df_medianfts = df_medianfts.rename_axis("PatID").reset_index(name="MedianFeaturesperCluster")

    meanftscluster = df_result["Counts"].mean()
    medianftscluster = df_result["Counts"].median()
    # get mean number of features per cluster
    #print(df_numfts)

    # merge dataframes
    df_numclust = pd.merge(df_numclust, df_numfts, on="PatID")
    df_numclust = pd.merge(df_numclust, df_medianfts, on="PatID")


    print("Mean number of stable clusters per patient: ", meanstable)
    print("Mean number of clusters per patient: ", df_numclust["NumClusters"].mean())
    print("Mean features per cluster per patient: ", df_numfts["MeanFeaturesperCluster"].mean())
    print("Std features per cluster per patient: ", df_numfts["MeanFeaturesperCluster"].std())

    print("Range of clusters: ", df_numclust["NumClusters"].min(), df_numclust["NumClusters"].max())
    print("Std of clusters: ", df_numclust["NumClusters"].std())
    print("Range of features per cluster: ", df_numfts["MeanFeaturesperCluster"].min(), df_numfts["MeanFeaturesperCluster"].max())

    # output to txt file:
    outfile = outdir.split("Labels")[0] + "ClusterSummary.txt"
    with open(outfile, "w") as f:
        f.write("Mean number of stable clusters per patient: {}\n".format(meanstable))
        f.write("Mean number of clusters per patient: {}\n".format(df_numclust["NumClusters"].mean()))
        f.write("Mean features per cluster per patient: {}\n".format(df_numfts["MeanFeaturesperCluster"].mean()))
        f.write("Std features per cluster per patient: {}\n".format(df_numfts["MeanFeaturesperCluster"].std()))
        f.write("Range of clusters: {} {}\n".format(df_numclust["NumClusters"].min(), df_numclust["NumClusters"].max()))
        f.write("Std of clusters: {}\n".format(df_numclust["NumClusters"].std()))
        f.write("Range of features per cluster: {} {}\n".format(df_numfts["MeanFeaturesperCluster"].min(), df_numfts["MeanFeaturesperCluster"].max()))
        f.close()

    # output to csv
####################################################

def cluster_correlation(Cluster_ft_df):
    '''
    Input - df filtered for norm, patient, cluster
    Output - performs cross-correlation within clustered fts and returns ft most strongly correlated with the rest, if more than 2 fts present
    '''
    fts = Cluster_ft_df.Feature.unique()
    num_fts = len(fts)
   
    if num_fts > 2:
        vals = {} # stores fts and values
        ccfs = {} # stores cc values for each feature
        mean_ccfs = {} # stores the mean cc value for every feature
        num_sel = np.rint(len(fts) * 0.2)
        
        for f in fts:
            ft_df = Cluster_ft_df[Cluster_ft_df["Feature"] == f]
            ft_vals = ft_df.FeatureValue.values
            vals[f] = ft_vals
        
        for v in vals:
            ft_1 = vals[v]
            ccfs[v] = v
            ccfs_vals = []

            for u in vals:
                ft_2 = vals[u]
                # calc pearson correlation
                corr = stats.pearsonr(ft_1, ft_2)[0] # cross correlation value, index [0] for for 0 lag in csc function
                # corr = sts.ccf(ft_1, ft_2)[0] # cross correlation value, index [0] for for 0 lag in csc function
                ccfs_vals.append(corr)
            
            mean_ccfs[v] = np.array(ccfs_vals).mean() # get mean across all cc values for each ft

        s_mean_ccfs = sorted(mean_ccfs.items(), key=lambda x:x[1], reverse=True)
        sorted_temp = s_mean_ccfs[0:int(num_sel)]
        ft_selected = [seq[0] for seq in sorted_temp]

    else: 
        ft_selected = 0

    return ft_selected

####################################################

def select_features(df, outdir):
    '''
    Loops through each patient  to select the 'best' feature for each cluster by performing cross-correlation
    Discards clusters with less than 3 features
    Selects features which are ranked in top 10 across all patients
    '''
    print("-" *30)
    print("Feature Selection")
    print("Calculating Cross-Correlation values...")
    patIDs = df["PatID"].unique()

    labels_dir = outdir + "/Clustering/Labels/"
    out_dir = outdir + "/Features/"
    
    df_result = pd.DataFrame()
        
    for pat in tqdm(patIDs):
        # read in feature vals and associated cluster labels
        df_pat_c = pd.read_csv(labels_dir + str(pat) + ".csv")
        df_pat_v = df.loc[df["PatID"] == pat]

        cluster_num = df_pat_c["ClusterLabel"].unique()
        fts_selected = []
        df_result_pat = pd.DataFrame()

        # for each patient loop through each cluster to get 'best' feature
        for c in cluster_num:
            df_cluster = df_pat_c[df_pat_c["ClusterLabel"] == c]
            features_c = df_cluster["Feature"].unique()
            df_cl_v = df_pat_v.loc[df_pat_v["Feature"].isin(features_c)]

            # function loops through each cluster and gets feature values
            # performs cross-correlation and returns feature with highest mean correlation to all other features
            # returns NULL if < 3 features in cluster 
            ft_selected = cluster_correlation(df_cl_v)

            if ft_selected != 0:
                for f in ft_selected:
                    fts_selected.append(f)
        
        # filter through all feature values and select only new features
            row = {}

        for f in fts_selected:
            row["PatID"] = pat
            row["Feature"] = f
            df_result_pat = df_result_pat.append(row, ignore_index=True)
        
        df_result = df_result.append(df_result_pat, ignore_index=True)

    df_result = df_result.Feature.value_counts().rename_axis("Feature").reset_index(name="Counts")
    # get number of counts at 10th row
    
    counts = df_result.iloc[9]["Counts"]
    #counts = 7
    # get length of df
    #length = len(df_result)
    # get top 20% of features
    # counts = df_result.iloc[int(len(df_result) * 0.1)]["Counts"]
    #print(df_result)
    # get features with counts >= counts
    fts = df_result[df_result["Counts"] >= counts]["Feature"].values
    print("-" * 30)    
    print("Selected Features: ")
    for ft in fts:
        print(ft)
    print("-" * 30)
    print("Number of Selected Features: {}".format(len(fts)))
    
    # df_result = df_result[df_result["Feature"].isin(fts)]
    df_result = df_result[df_result["Counts"] >= counts]
    # drop counts
    df_result.to_csv(out_dir + "Features_Selected.csv")
    print("-" * 30)
####################################################

def cluster_correlation_Delta(Cluster_ft_df):
    '''
    Input - df filtered for norm, patient, cluster
    Output - performs cross-correlation within clustered fts and returns ft most strongly correlated with the rest, if more than 2 fts present
    '''
    fts = Cluster_ft_df.Feature.unique()
    num_fts = len(fts)
    ft_selected = []

    if num_fts > 2:
        vals = {} # stores fts and values
        diffs = {}
        num_sel = np.rint(len(fts) * 0.2)
        
        for f in fts:
            ft_df = Cluster_ft_df[Cluster_ft_df["Feature"] == f]
            ft_vals = ft_df.FeatureValue.values
            vals[f] = ft_vals[0]
        
        values = list(vals.values())
        mean = sum(values) / len(values)

        for f in fts:
            diffs[f] = abs(mean - vals[f])
        
        sorted_vals = sorted(diffs.items(), key=lambda x: x[1])
        # print("sorted: ", sorted_vals[0])
        for i in range(0, int(num_sel), 1):
            ft_selected.append(sorted_vals[i][0])

    else: 
        ft_selected = 0

    return ft_selected

####################################################
def select_features_Delta(df, outdir):
    '''
    Loops through each patient  to select the 'best' feature for each cluster by performing cross-correlation
    Discards clusters with less than 3 features
    Selects features which are ranked in top 10 across all patients
    '''
    print("-" *30)
    print("Feature Selection")
    print("Calculating Cross-Correlation values...")
    patIDs = df["PatID"].unique()

    labels_dir = outdir + "/Clustering/Labels/"
    out_dir = outdir + "/Features/"
    
    df_result = pd.DataFrame()
    for pat in tqdm(patIDs):
        # read in feature vals and associated cluster labels
        df_pat_c = pd.read_csv(labels_dir + str(pat) + ".csv")
        df_pat_v = df.loc[df["PatID"] == pat]

        cluster_num = df_pat_c["ClusterLabel"].unique()
        fts_selected = []
        df_result_pat = pd.DataFrame()

        # for each patient loop through each cluster to get 'best' feature
        for c in cluster_num:
            df_cluster = df_pat_c[df_pat_c["ClusterLabel"] == c]
            features_c = df_cluster["Feature"].unique()
            df_cl_v = df_pat_v.loc[df_pat_v["Feature"].isin(features_c)]

            # function loops through each cluster and gets feature values
            # performs cross-correlation and returns feature with highest mean correlation to all other features
            # returns NULL if < 3 features in cluster 
            ft_selected = cluster_correlation_Delta(df_cl_v)

            if ft_selected != 0:
                for f in ft_selected:
                    fts_selected.append(f)
        
        # filter through all feature values and select only new features
            row = {}

        for f in fts_selected:
            row["PatID"] = pat
            row["Feature"] = f
            df_result_pat = df_result_pat.append(row, ignore_index=True)
        
        df_result = df_result.append(df_result_pat, ignore_index=True)

    df_result = df_result.Feature.value_counts().rename_axis("Feature").reset_index(name="Counts")
    # get number of counts at 10th row
    df_result.to_csv(out_dir + "Features_Selected_Full.csv")
    counts = df_result.iloc[10]["Counts"]
    #print(df_result)
    # get features with counts >= counts
    fts = df_result[df_result["Counts"] >= counts]["Feature"].values
    print("-" * 30)    
    print("Selected Features: ")
    for ft in fts:
        print(ft)
    print("-" * 30)
    print("Number of Selected Features: {}".format(len(fts)))
    
    df_result = df_result[df_result["Counts"] >= counts]

    # drop counts
    df_result.to_csv(out_dir + "Features_Selected.csv")
    print("-" * 30)

####################################################

def LongitudinalModel(DataRoot, Norm, Extract, t_val, tag, output=False):
    # Make Directories if they don't exist
    #print("------------------------------------")
    #print("------------------------------------")
    # print("Checking Directories...")
    print("Root: {} Norm: {}, Tag: {}".format(DataRoot, Norm, tag))
    uf.CD(DataRoot, Extract, Norm, tag)
    
    print("------------------------------------\n")
    print("         Longitudinal Model         \n")
    print("------------------------------------\n")

    # Extract Features
    if Extract == "Yes":
        print("------------------------------------")
        print("------------------------------------")
        print("Extracting Features...")
        FE.All(DataRoot, Norm, tag)
        print("Extracted - All")
        print("------------------------------------")
        FE.Limbus(DataRoot, Norm, tag)
        print("Extracted - Limbus")
        print("------------------------------------")
        print("------------------------------------\n ")

    # Feature Reduction
    if output == True:
        print("------------------------------------")
        print("------------------------------------")
        print("Reducing Features...")
        print("Volume Correlation Feature Reduction: ")
        print("------------------------------------")
        print("------------------------------------")

    FR.Volume(DataRoot, Norm, "Longitudinal", tag, output)
    if output == True:
        print("------------------------------------")
        print("ICC Feature Reduction: ")
        print("------------------------------------\n ")
    FR.ICC(DataRoot, Norm, "Longitudinal", tag, output)
    # Clustering
    if output == True:
        print("------------------------------------")
        print("------------------------------------")
        print("Clustering...")
        print("------------------------------------")
        print("Creating Distance Matrices: ")
        print("------------------------------------")
    DistanceMatrix(DataRoot, Norm, tag)
    
    if output == True:
        print("------------------------------------")
        print("Clustering Distance Matrices: ")
        print("------------------------------------")
    ClusterFeatures(DataRoot, Norm, t_val, tag)
    #count_clusters(DataRoot, Norm, output)
    if output == True:
        print("Feature Selection: ")
        print("------------------------------------")
    ClusterSelection(DataRoot, Norm, tag, output)
    print("------------------------------------")
    print("------------------------------------\n ")

####################################################

def ModelCompact(DataRoot, Norm, t_val, tag, output=False):
    print("------------------------------------")
    print("------------------------------------")
    print("Root: {} Norm: {} Tag: {}".format(DataRoot, Norm, tag))

    print("Creating Distance Matrices: ")
    print("------------------------------------")
    DistanceMatrix(DataRoot, Norm, tag, output)
    
    print("------------------------------------")
    print("Clustering Distance Matrices: ")
    print("------------------------------------")
    ClusterFeatures(DataRoot, Norm, t_val, tag, output)
    count_clusters(DataRoot, Norm, tag, output)
    print("Feature Selection: ")
    print("------------------------------------")
    ClusterSelection(DataRoot, Norm, tag, output)
    print("------------------------------------")
    print("------------------------------------\n ")

####################################################
def ClusterLinkedFts(ft, df):
    '''
    Given a feature, returns all features in the same cluster
    '''
    c = df[df["FeatureName"] == ft]["Cluster"].values[0]

    linked_fts = df[df["Cluster"] == c]["FeatureName"].values
    linked_fts = np.delete(linked_fts, np.where(linked_fts == ft))

    return linked_fts

####################################################
def ClusterSimilarity(fts_1, fts_2):
    '''
    Calculates the similarity between two sets of features
    '''
    fts_1, fts_2 = list(fts_1), list(fts_2)
    sim_fts = set(fts_1) & set(fts_2)
    num_sim_fts = len(sim_fts)
    
    if len(fts_1) != 0 and len(fts_2) != 0:
        
        ratio_a  = len(sim_fts) / len(fts_1)
        ratio_b = len(sim_fts) / len(fts_2)

        ratio = (ratio_a - ratio_b) 
    else: 
        ratio, ratio_a, ratio_b = 1,1,1
    
    return(num_sim_fts, ratio_a, ratio_b, ratio)

####################################################