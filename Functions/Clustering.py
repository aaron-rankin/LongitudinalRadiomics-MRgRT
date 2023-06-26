import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import scipy.cluster.hierarchy as spch
import statsmodels.tsa.stattools as sts
from scipy import stats
from Functions import UsefulFunctions as UF
from Functions import ImageFunctions as IF
from scipy.spatial import distance
from Features import Reduction as FR
from Features import Extraction as FE

####################################################

def DistanceMatrix(DataRoot, Norm, tag):
    '''
    Calculates Eucledian distance between all features for each patient
    '''
    root = DataRoot
    df_all = pd.read_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\Features\\Longitudinal_All_fts_" + tag + ".csv")

    fts_ICC = pd.read_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\Features\\Longitudinal_FeaturesRemoved_ICC_" + tag + ".csv")
    fts_Vol = pd.read_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\Features\\Longitudinal_FeaturesRemoved_Volume_" + tag + ".csv")

    df_all = df_all[~df_all["Feature"].isin(fts_ICC["Feature"])]
    df_all = df_all[~df_all["Feature"].isin(fts_Vol["Feature"])]

    patIDs = df_all["PatID"].unique()
    fts = df_all["Feature"].unique()

    print("Volume Redundant features: {}".format(len(fts_Vol)))
    print("ICC Redundant features: {}".format(len(fts_ICC)))
    print("Remainder of features: {}".format(len(fts)))
    # loop through patients
    for pat in tqdm(patIDs):
        df_pat = df_all[df_all["PatID"] == pat]

        # empty matrix
        mat = np.zeros((len(fts), len(fts)))

        for ft1 in range(len(fts)):
            vals_ft1 = df_pat[df_pat["Feature"] == fts[ft1]]["FeatureChange"].values

            for ft2 in range(len(fts)):
                vals_ft2 = df_pat[df_pat["Feature"] == fts[ft2]]["FeatureChange"].values

                # calculate correlation
                # mat[ft1, ft2] = stats.pearsonr(vals_ft1, vals_ft2)[0]
                mat[ft1, ft2] = distance.euclidean(vals_ft1, vals_ft2)

        # save matrix
        df_dist = pd.DataFrame(mat, columns = fts, index = fts)  
        df_dist.to_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm +"\\Longitudinal\\DM\\csvs\\" + str(pat) + "_" + tag + ".csv")

        # plot matrix
        plt.figure(figsize=(20,20))
        sns.set_theme(style="white")
        plt.title("DM - {}".format(pat), fontsize=40)
        sns.heatmap(df_dist, cmap='viridis', cbar_kws={'label': 'Euclidean Distance'})
        plt.savefig(root + "\\Aaron\\ProstateMRL\\Data\\Paper1\\"+ Norm +"\\Longitudinal\\DM\\Figs\\" + str(pat) + "_" + tag + ".png")

####################################################

def ClusterCheck(df, fts, t_val, tries, df_DM):
        '''
        If cluster has more than 10 features, re-cluster with smaller t_val
        '''
        df_c = df
        df_new = pd.DataFrame()
        # feature names
        df_new["FeatureName"] = fts
        # cluster labels
        c = df_c["Cluster"].values[0]
        
        # need to filter distance matrix to only include features in cluster
        df_DM_c = df_DM[fts]
        # only keep features in cluster
        df_DM_c = df_DM_c[df_DM_c.index.isin(fts)]
        
        # convert to numpy array
        arr_DM_c = df_DM_c.to_numpy()
        
        # cluster
        df_new["Cluster"] = spch.fclusterdata(arr_DM_c, t=t_val, criterion="distance", method="ward")
        df_new["Cluster"] = str(c*100) + str(tries) + df_new["Cluster"].astype(str)
        df_new["Cluster"] = df_new["Cluster"].astype(int)
        df_new["NumFts"] = df_new.groupby("Cluster")["Cluster"].transform("count")
        number_fts = df_new["NumFts"].unique()
        fts_check = df_new.loc[df_new["NumFts"] > 10]["FeatureName"].values
        #print(t_val, number_fts)#, df_new)
        return number_fts, df_new, fts_check

####################################################

def ClusterFeatures(DataRoot, Norm, s_t_val, tag):
    '''
    Cluster features using distance matrix, 
    t_val is threshold for clustering, 
    method is clustering forumula
    performs clustering until all clusters have less than 10 features
    '''
    root = DataRoot
    DM_dir = root + "\\Aaron\\ProstateMRL\\Data\\Paper1\\" + Norm + "\\Longitudinal\\DM\\csvs\\"
    out_dir = root + "\\Aaron\\ProstateMRL\\Data\\Paper1\\"+ Norm + "\\Longitudinal\\ClusterLabels\\"

    patIDs = UF.SABRPats()

    cluster_method = "weighted"

    for pat in tqdm(patIDs):
        df_DM = pd.read_csv(DM_dir + pat + "_" + tag + ".csv")
        df_DM.set_index("Unnamed: 0", inplace=True)
        arr_DM = df_DM.to_numpy()
        fts = df_DM.columns

        # create temp df to hold ft name and label
        df_labels = pd.DataFrame()
        df_labels["FeatureName"] = fts

        # cluster function using DM, need to experiment with t_val and method
        df_labels["Cluster"] = spch.fclusterdata(arr_DM, t=s_t_val, criterion="distance", method=cluster_method)
        df_labels.set_index("FeatureName", inplace=True)
        
        # check number of features in each cluster
        df_labels["NumFts"] = df_labels.groupby("Cluster")["Cluster"].transform("count")
        df_labels["Cluster"] = df_labels["Cluster"].astype(int)
        #print("---------------------------")
        #print("Patient: {}".format(pat))
        #print(df_labels.loc[df_labels["NumFts"] > 10])
        # loop through clusters 
        for c in df_labels["Cluster"].unique():
                df_c = df_labels[df_labels["Cluster"] == c]
                number_fts = len(df_c)
                # check numnber of features in cluster
                if number_fts > 10:
                        # if more than 10 features in cluster, reduce t_val and recluster
                        t_val = s_t_val - 0.2
                        check_fts = df_c.index.values
                        tries = 1
                        number_fts, df_labels2, check_fts = ClusterCheck(df_c, check_fts, t_val, tries, df_DM)
                        new_fts = df_labels2["FeatureName"].unique()
                        df_labels.loc[new_fts, "Cluster"] = df_labels2["Cluster"].values
                        df_labels["NumFts"] = df_labels.groupby("Cluster")["Cluster"].transform("count")

                        while number_fts.max() > 10:
                                t_val = t_val - 0.2
                                tries += 1
                                #print("Cluster: {} Tries: {} T_val: {}".format(c, tries, t_val))
                                number_fts, df_labels2, check_fts = ClusterCheck(df_c, check_fts, t_val, tries, df_DM)
                                new_fts = df_labels2["FeatureName"].unique()
                                df_labels.loc[new_fts, "Cluster"] = df_labels2["Cluster"].values
                        
        df_labels["NumFts"] = df_labels.groupby("Cluster")["Cluster"].transform("count")

        # read in df with ft vals and merge
        ft_vals = pd.read_csv(root +"Aaron\\ProstateMRL\\Data\\Paper1\\"+ Norm + "\\Features\\Longitudinal_All_fts_" + tag + ".csv")
        ft_vals["PatID"] = ft_vals["PatID"].astype(str)
        pat_ft_vals = ft_vals[ft_vals["PatID"] == pat]
        pat_ft_vals = pat_ft_vals.merge(df_labels, left_on="Feature", right_on="FeatureName")

        # output is feature values w/ cluster labels
        pat_ft_vals.to_csv(out_dir + pat + "_" + tag + ".csv")

####################################################

def ClusterCount(root, Norm, output, tag):
    '''
    Summarises clustering results
    '''
    dir = root + "\\Aaron\\ProstateMRL\\Data\\Paper1\\" + Norm + "\\Longitudinal\\ClusterLabels\\"
    patIDs = UF.SABRPats()
    df_result = pd.DataFrame()

    for pat in patIDs:

        df = pd.read_csv(root + "\\Aaron\\ProstateMRL\\Data\\Paper1\\" + Norm + "\\Longitudinal\\ClusterLabels\\" + pat + "_" + tag + ".csv")
        df = df[["Feature", "Cluster"]]
        df = df.drop_duplicates()
        # sort by cluster
        df = df.sort_values(by=["Cluster"])
        # turn value counts into a dataframe
        df = df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Counts")
        # set PatID as index
        df["PatID"] = f[3:-4]
        # set PatID as index
        df.set_index("PatID", inplace=True)
            
        # append to result
        df_result = df_result.append(df, ignore_index=False)
    #get number of clusters with more than 3 features
    df_stable = df_result[df_result["Counts"] > 3]
    df_stable = df_stable.groupby("PatID")["Cluster"].count()
    # get mean number of stable clusters
    meanstable = df_stable.mean()
    #print(df_result)
    df_numclust= df_result.groupby("PatID")["Cluster"].count()
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

    if output == True:
        print("Mean number of stable clusters per patient: ", meanstable)
        print("Mean number of clusters per patient: ", df_numclust["NumClusters"].mean())
        print("Mean features per cluster per patient: ", df_numfts["MeanFeaturesperCluster"].mean())

####################################################

def ClusterCC(Cluster_ft_df):
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
            ft_vals = ft_df.FeatureChange.values
            vals[f] = ft_vals
        
        for v in vals:
            ft_1 = vals[v]
            ccfs[v] = v
            ccfs_vals = []

            for u in vals:
                ft_2 = vals[u]
                corr = sts.ccf(ft_1, ft_2)[0] # cross correlation value, index [0] for for 0 lag in csc function
                ccfs_vals.append(corr)
            
            mean_ccfs[v] = np.array(ccfs_vals).mean() # get mean across all cc values for each ft

        s_mean_ccfs = sorted(mean_ccfs.items(), key=lambda x:x[1], reverse=True)
        sorted_temp = s_mean_ccfs[0:int(num_sel)]
        ft_selected = [seq[0] for seq in sorted_temp]

    else: 
        ft_selected = 0

    return ft_selected

####################################################

def ClusterSelection(DataRoot, Norm, tag, output):
    '''
    Loops through each patient  to select the 'best' feature for each cluster by performing cross-correlation
    Discards clusters with less than 3 features
    Selects features which are ranked in top 10 across all patients
    '''
    root = DataRoot
    patIDs = UF.SABRPats()

    labels_dir = root + "\\Aaron\\ProstateMRL\\Data\\Paper1\\" + Norm + "\\Longitudinal\\ClusterLabels\\"
    out_dir = root + "\\Aaron\\ProstateMRL\\Data\\Paper1\\"+ Norm +"\\Features\\"
    
    df_result = pd.DataFrame()
    for pat in tqdm(patIDs):
        # read in feature vals and associated cluster labels
        df_pat = pd.read_csv(labels_dir + pat + "_" + tag + ".csv")

        cluster_num = df_pat["Cluster"].unique()
        fts_selected = []
        df_result_pat = pd.DataFrame()

        # for each patient loop through each cluster to get 'best' feature
        for c in cluster_num:
            df_cluster = df_pat[df_pat["Cluster"] == c]

            # function loops through each cluster and gets feature values
            # performs cross-correlation and returns feature with highest mean correlation to all other features
            # returns NULL if < 3 features in cluster 
            ft_selected = ClusterCC(df_cluster)

            if ft_selected != 0:
                for f in ft_selected:
                    fts_selected.append(f)
        
        # filter through all feature values and select only new features
            row = {}

        for f in fts_selected:
            row["patID"] = pat
            row["Feature"] = f
            df_result_pat = df_result_pat.append(row, ignore_index=True)
        
        df_result = df_result.append(df_result_pat, ignore_index=True)

    df_result = df_result.Feature.value_counts().rename_axis("Feature").reset_index(name="Counts")
    # get number of counts at 10th row
    counts = df_result.iloc[10]["Counts"]
    #print(df_result)
    # get features with counts >= counts
    fts = df_result[df_result["Counts"] >= counts]["Feature"].values
    if output == True:
        print("\nSelected Features: ({})".format(len(fts)))
        for f in fts:
            print(f)
    df_result = df_result[df_result["Counts"] >= counts]

    # drop counts
    df_result.drop(columns=["Counts"], inplace=True)
    df_result.to_csv(out_dir + "Longitudinal_SelectedFeatures_" + tag + ".csv")

####################################################

def LongitudinalModel(DataRoot, Norm, Extract, t_val, tag, output=False):
    # Make Directories if they don't exist
    #print("------------------------------------")
    #print("------------------------------------")
    # print("Checking Directories...")
    print("Root: {} Norm: {}, Tag: {}".format(DataRoot, Norm, tag))
    UF.CD(DataRoot, Extract, Norm, tag)
    
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
    #ClusterCount(DataRoot, Norm, output)
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
    ClusterCount(DataRoot, Norm, tag, output)
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