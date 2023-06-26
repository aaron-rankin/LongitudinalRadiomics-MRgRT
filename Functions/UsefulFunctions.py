from asyncio.windows_events import NULL
from cmath import nan
import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
import pandas as pd
import statsmodels.tsa.stattools as sts
from datetime import datetime

####################################################

def DataRoot(root):
    '''
    Returns the root directory of the data if on
    server or local
    '''
    if root == 1:
        # server
        root = "D:\\data\\"
    elif root == 2:
        # local
        root = "E:\\"
    
    return root

#####################################################

def CD(DataRoot, Extract, Norm, tag):
    '''
    Checks if folders exist and creates them if not
    '''
    root_dir = DataRoot + "Aaron\ProstateMRL\Data\Paper1\\"

    if not os.path.exists(root_dir + Norm):
        os.makedirs(root_dir + Norm)

    root_dir = os.path.join(root_dir, Norm)

    # Delta folder
    if not os.path.exists(root_dir + "\\Delta"):
        os.makedirs(root_dir + "\\Delta")

    # Longitudinal folder
    if not os.path.exists(root_dir + "\\Longitudinal"):
        os.makedirs(root_dir + "\\Longitudinal")
        os.makedirs(root_dir + "\\Longitudinal\\DM")
        os.makedirs(root_dir + "\\Longitudinal\\DM\\csvs")
        os.makedirs(root_dir + "\\Longitudinal\\DM\\Figs")
        os.makedirs(root_dir + "\\Longitudinal\\ClusterLabels")
        os.makedirs(root_dir + "\\Longitudinal\\ClusterPlots")

    # Features folder
    if not os.path.exists(root_dir + "\\Features"):
        os.makedirs(root_dir + "\\Features")
    if Extract == "No":
        if not os.path.exists(root_dir + "\\Features\\Longitudinal_All_fts_" + tag + ".csv"):
            df_fts = pd.read_csv(root_dir + "\\Features\\Longitudinal_All_fts_Baseline.csv")
            df_fts.to_csv(root_dir + "\\Features\\Longitudinal_All_fts_" + tag + ".csv")
            df_l = pd.read_csv(root_dir + "\\Features\\Longitudinal_Limbus_fts_Baseline.csv")
            df_l.to_csv(root_dir + "\\Features\\Longitudinal_Limbus_fts_" + tag + ".csv")
#####################################################


def SABRPats():
    '''
    Returns array of patIDs for SABR 
    '''
    array = ['653', '713', '752', '826', '1088', '1089', '1118', '1303', '1307', '1464', '1029',
 '1302', '1431', '1481', '1540', '1553', '1601', '1642', '829', '955']

    return array

#####################################################

def NormArray():
    """
    Returns array of normalisation factors
    """
    Norms = ["Raw", "HM-FS", "HM-TP", "HM-FSTP", "Med-Pros", "Med-Glute", "Med-Psoas"]
    return Norms

####################################################

def GetNorm(image_name): 
    '''
    Input: image file name
    Output: Normalisation method
    '''
    n = image_name

    Norm = n.split("_")[2]
    Norm = Norm.replace(".nii", "")
    Norm = Norm.replace("v2", "")
    if Norm == "image":
        Norm = "Raw"

    return Norm

####################################################

def GetRegion(mask_name):
    '''
    Input: Mask name
    Output: Mask region
    '''
    n = mask_name

    if "pros" in n:
        Region = "Prostate"
    elif "glute" in n: 
        Region = "Glute"
    elif "psoas" in n:
        Region = "Psoas"
    else:
        Region = "-"
    
    return Region

####################################################

def FixPatID(patID, treatment_group):
    '''
    Patient IDs are not consistent across the data
    '''
    if "new" in treatment_group:
        newID = str(patID)
    else:
        if len(str(patID)) == 3:
            newID = "0000" + str(patID)
        elif len(str(patID)) == 4:
            newID = "000" + str(patID)
        else:
            newID = str(patID)

    return newID 

####################################################

def FixDate(date):
    '''
    Reformats date to YYYYMMDD
    '''
    date_string = str(date)
    if len(date_string) != 8:
        date_string = date_string[:-2]
    date = datetime.strptime(date_string, "%Y%m%d").date()

    return date
####################################################

def GetNiftiPaths(patient_path, treatment):
    """
    Returns the paths to the nifti files
    """
    mask_path = os.path.join(patient_path, "Masks\\")

    if treatment == "20fractions":
        mask_labels = ["_shrunk_pros.nii", "_glute2.nii", "_psoas.nii"] # set glute2 for 20fractions
    else:
        mask_labels = ["_shrunk_pros.nii", "_glute.nii", "_psoas.nii"]

    image_roots = ["RawImages\\", "HM-TP\\", "HM-FS\\", "HM-FSTP\\","Norm-Pros\\", "Norm-Glute\\", "Norm-Psoas\\", "Med-Pros\\", "Med-Glute\\", "Med-Psoas\\"]
    #image_roots = ["BaseImages\\", "HM-TP\\","Norm-Psoas\\", "Med-Psoas\\"]

    image_labels = ["Raw", "HM-TP", "HM-FS",  "HM-FSTP", "Norm-Pros","Norm-Glute", "Norm-Psoas", "Med-Pros", "Med-Glute", "Med-Psoas"]
    #image_labels = ["Raw", "HM-TP", "Norm-Psoas", "Med-Psoas"]
    
    image_paths = []
    
    for b in image_roots:
        image_paths.append(os.path.join(patient_path, b))
    
    return mask_path, mask_labels, image_paths, image_labels

####################################################

def GetNiftiPathsProsSens(patient_path, treatment):
    """
    """
    mask_path = os.path.join(patient_path, "Masks\\")

    masks = os.listdir(mask_path)
    mask_labels = []
    for x in masks:
        if "shrunk_pros" in x[:-7]:
            mask_labels.append(x)

    image_roots = ["RawImages\\", "HM-TP\\", "HM-FS\\", "HM-FSTP\\","Norm-Pros\\", "Norm-Glute\\", "Norm-Psoas\\", "Med-Pros\\", "Med-Glute\\", "Med-Psoas\\"]
    #image_roots = ["BaseImages\\", "HM-TP\\","Norm-Psoas\\", "Med-Psoas\\"]

    image_labels = ["Raw", "HM-TP", "HM-FS",  "HM-FSTP", "Norm-Pros","Norm-Glute", "Norm-Psoas", "Med-Pros", "Med-Glute", "Med-Psoas"]
    #image_labels = ["Raw", "HM-TP", "Norm-Psoas", "Med-Psoas"]
    
    image_paths = []
    
    for b in image_roots:
        image_paths.append(os.path.join(patient_path, b))
    
    return mask_path, mask_labels, image_paths, image_labels

####################################################

def GetImageFile(image_path, patient, scan, image_label):
    """
    Returns the path to the image file
    """
    label = image_label
    #if image_label.__contains__("Raw"):
    #    image_label == "image"
    #if label.__contains__("Norm"):# or label.__contains__("Med"):
    #    label = label + "_v2"
    
    
    file_name = patient + "_" + scan + "_" + label + ".nii"
    file_path = os.path.join(image_path, file_name)

    return file_path, file_name

####################################################

def ClusterFtSelection(Cluster_ft_df):
    '''
    Input - df filtered for norm, patient, cluster
    Output - performs cross-correlation within clustered fts and returns ft most strongly correlated with the rest, if more than 2 fts present
    '''
    fts = Cluster_ft_df.FeatureName.unique()
    num_fts = len(fts)
   
    if num_fts > 2:
        vals = {} # stores fts and values
        ccfs = {} # stores cc values for each feature
        mean_ccfs = {} # stores the mean cc value for every feature

        for f in fts:
            ft_df = Cluster_ft_df[Cluster_ft_df["FeatureName"] == f]
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

        ft_selected = max(mean_ccfs, key=mean_ccfs.get) # get max mean cc value and return the feature

    else: 
        ft_selected = np.nan

    return ft_selected

####################################################

def ClusterFtSelection2(Cluster_ft_df):
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

def ICC_Class(icc_val):
    '''
    Classifies the icc value into poor, moderate, good, excellent
    '''

    if icc_val < 0.5:
        icc_class = "Poor"
    elif icc_val >= 0.5 and icc_val < 0.75:
        icc_class = "Moderate"
    elif icc_val >= 0.75 and icc_val < 0.9:
        icc_class = "Good"
    elif icc_val >= 0.9 and icc_val < 1:
        icc_class = "Excellent"
    else:
        icc_class = np.nan

    return icc_class

####################################################

def CompareFeatureLists(root, Norm, stages, tag):
    '''
    Given feature lists from different methods, returns the features that are in common
    '''
    fts_1 = pd.read_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\Features\\" + stages[0] + "_" + tag + ".csv")
    fts_2 = pd.read_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\Features\\" + stages[1] + "_" + tag + ".csv")

    fts_1 = set(fts_1["Feature"].unique())
    fts_2 = set(fts_2["Feature"].unique())

    common_fts = fts_1.intersection(fts_2)

    return common_fts
    
####################################################

def ModelSummary(root, Norm, tag):
    dir = root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\Features\\"
    out = open(root + "Aaron\ProstateMRL\Data\Paper1\\NormSummary\\" + Norm + "_" + tag + ".txt", "w")

    out.write("Model Summary - " + Norm)
    out.write("\n")
    out.write("-------------------------\n")
    out.write("ICC Reduction\n")
    out.write("Features Before: 105\n")

    L_ICC = pd.read_csv(dir + "Longitudinal_FeaturesRemoved_ICC_" + tag + ".csv")
    L_ICC_fts = len(L_ICC["Feature"].unique())
    out.write("Longitudinal Features Removed: " + str(L_ICC_fts) + "\n")
    out.write("\n")
    D_ICC = pd.read_csv(dir + "Delta_FeaturesRemoved_ICC_" + tag + ".csv")
    D_ICC_fts = len(D_ICC["Feature"].unique())
    out.write("Delta Features Removed: " + str(D_ICC_fts) + "\n")

    out.write("-------------------------\n")
    out.write("Volume Reduction\n")
    out.write("\n")
    out.write("Longitudinal Features Before: " + str(105 - L_ICC_fts) + "\n")
    L_Vol = pd.read_csv(dir + "Longitudinal_FeaturesRemoved_Volume_" + tag + ".csv")
    L_Vol_fts = len(L_Vol["Feature"].unique())
    out.write("Longitudinal Features Removed: " + str(L_Vol_fts)+ "\n")

    out.write("\n")
    out.write("Delta Features Before: " + str(105 - D_ICC_fts)+ "\n")
    D_Vol = pd.read_csv(dir + "Delta_FeaturesRemoved_Volume_" + tag + ".csv")
    D_Vol_fts = len(D_Vol["Feature"].unique())
    out.write("Delta Features Removed: " + str(D_Vol_fts) + "\n")

    out.write("-------------------------\n")
    L_both = CompareFeatureLists(root, Norm, ["Longitudinal_FeaturesRemoved_ICC", "Longitudinal_FeaturesRemoved_Volume"], tag)
    out.write("Longitudinal - Number of features both Vol & ICC redudant: " + str(len(L_both)) + "\n")
    D_both = CompareFeatureLists(root, Norm, ["Delta_FeaturesRemoved_ICC", "Delta_FeaturesRemoved_Volume"], tag)
    out.write("Delta - Number of features both Vol & ICC redudant: " + str(len(D_both)) + "\n")

    out.write("-------------------------\n")

    out.write("Feature Selection\n")
    out.write("\n")
    out.write("Longitudinal Features Before: " + str(105 - L_ICC_fts - L_Vol_fts) + "\n")
    L_Select = pd.read_csv(dir + "Longitudinal_SelectedFeatures_" + tag + ".csv")
    L_Select_fts = L_Select["Feature"].unique()
    out.write("Longitudinal Features Selected: " + str(len(L_Select_fts)) + "\n")

    out.write("\n")
    out.write("Delta Features Before: " + str(105 - D_ICC_fts - D_Vol_fts)+ "\n")
    D_Select = pd.read_csv(dir + "Delta_SelectedFeatures_" + tag + ".csv")
    D_Select_fts = D_Select["Feature"].unique()
    out.write("Delta Features Selected: " + str(len(D_Select_fts))+ "\n")

    # check if any features are selected in both longitudinal and delta
    out.write("\n")
    Selected_both = set(L_Select_fts).intersection(set(D_Select_fts))
    out.write("Features Selected in Both Longitudinal and Delta: " + str(len(Selected_both))+ "\n")
    out.write("-------------------------")

    out.close()

    # print out contents of file
    with open(root + "Aaron\ProstateMRL\Data\Paper1\\NormSummary\\" + Norm + "_" + tag + ".txt", "r") as f:
        print(f.read())

####################################################

