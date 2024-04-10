import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from datetime import datetime
# import radiomics
#from radiomics import featureextractor
import sys
from tqdm import tqdm
# import UsefulFunctions as UF

####################################################

def create_output_dirs(tag):
    '''
    Create output directories for storing extracted features and results
    '''
    cwd = os.getcwd()

    if not os.path.exists(f'{cwd}/Output'):
        os.makedirs(f'{cwd}/Output')
    if not os.path.exists(f'{cwd}/Output/{tag}'):
        os.makedirs(f'{cwd}/Output/{tag}')
        os.makedirs(f'{cwd}/Output/{tag}/features')
        os.makedirs(f'{cwd}/Output/{tag}/Extraction')
        os.makedirs(f'{cwd}/Output/{tag}/Plots')

    output_path = f'{cwd}/Output/{tag}/'

    return output_path

####################################################

def extract_features(output_path, key_path):

    '''
    Extract radiomics features from MR images
    '''
    key_extraction = pd.read_csv(key_path)
    params_extraction = key_path.split('/')[:-1] + 'Default_ExtractionParams.yaml'

    print('Extracting features...')

    df_all = pd.DataFrame()

    for i, row in tqdm(key_extraction.iterrows()):
        patID = row["PatID"]
        fraction = row["Fraction"]
        mask = row["Contour"]
        contour_type = row["ContourType"]
        image_path = row["ImagePath"]
        mask_path = row["MaskPath"]
        
        features = calc_features(patID, fraction, mask, contour_type, image_path, mask_path, params_extraction)
        
        df_all = pd.concat([df_all, features], axis=0)

    df_all.to_csv(f'{output_path}/features/features.csv', index=False)

    print('features extracted!')

    return df_all

####################################################

def get_df_all(output_path):
    '''
    Load extracted features if already extracted
    '''
    if os.path.exists(f'{output_path}/features/features_all.csv'):
        df_all = pd.read_csv(f'{output_path}/features/features_all.csv')
        
        return df_all
    
    elif os.path.exists(f'Output/Submission/features/features_all.csv'):
        df_all = pd.read_csv(f'Output/Submission/features/features_all.csv')
        
        return df_all
    else:
        print('Error loading features!')

        return None

    # return df all if it exists
        

####################################################

def calc_features(PatID, Fraction,  Contour, ContourType, image_path, mask_path, extractor_params):
    """
    PatID: Patient ID
    Mask: Mask name
    Fraction: Fraction number
    tag: Tag for output file
    image_path: path to image
    mask_path: path to mask - mask must be binary
    outDir: output directorsy
    extractor_params: path to parameter file - default is All.yaml (PyRadiomics default)
    """

    # create the feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(extractor_params)

    # extract features
    results = pd.Series(extractor.execute(image_path, mask_path, label=255))

    # convert to dataframe
    features = pd.DataFrame(results).T

    # if diagnostic features are included, remove them
    features = features.drop(columns = [col for col in features.columns if "diagnostic" in col])
    # remove original from column names
    features.columns = [col.replace("original_", "") for col in features.columns]
    features.insert(0, "PatID", PatID)
    features.insert(1, "Fraction", Fraction)
    features.insert(2, "Contour", Contour)
    features.insert(3, "ContourType", ContourType)

    features = features.melt(id_vars = ["PatID", "Fraction", "Contour", "ContourType"], var_name = "Feature", value_name = "FeatureValue")

    return features

####################################################

def rescale_features(df_all, output_path):
    '''
    Rescale features to be between 0 and 1
    '''
    from sklearn.preprocessing import MinMaxScaler

    df_rescaled = df_all.copy()
    fts = df_rescaled['Feature'].unique()

    for ft in fts:
        df_ft = df_rescaled[df_rescaled['Feature'] == ft]
        vals = df_ft['FeatureValue'].values

        vals_scaled = MinMaxScaler(feature_range=(0,1)).fit_transform(vals.reshape(-1, 1))
        df_rescaled.loc[df_rescaled['Feature'] == ft, 'FeatureValue'] = vals_scaled

    df_rescaled.to_csv(f'{output_path}/Features/Features_Rescaled.csv', index=False)
    
    return df_rescaled

####################################################

def format_df_all(df_all, output_path, rescale=False):
    '''
    Format the extracted features dataframe
    '''
    df_all['PatID'] = df_all['PatID'].astype(str)
    df_all['Fraction'] = df_all['Fraction'].astype(str)
    df_all['Contour'] = df_all['Contour'].astype(str)
    df_all['ContourType'] = df_all['ContourType'].astype(str)

    df_all = df_all[['PatID', 'Fraction', 'Contour', 'ContourType', 'Feature', 'FeatureValue']]
    df_all = df_all[~df_all["Feature"].isin(["firstorder_Minimum", "firstorder_Maximum"])]
    
    print(f'Dataframe formatted!')
    print(f'Dataframe shape:{df_all.shape}')
    print(f'Dataframe columns:{df_all.columns}')
    
    if rescale:
        df_all = rescale_features(df_all, output_path)
    
    return df_all

####################################################

def split_df_all(df_all):
    '''
    Split dataframe by contour type
    '''

    df_all_man = df_all[df_all['ContourType'] == 'Manual']
    df_all_auto = df_all[df_all['ContourType'] == 'Auto']

    return df_all_man, df_all_auto

####################################################

def calc_delta(df_pat):
    '''
    Calculate the change in features between first and last fraction
    '''
    # pivot df based on fraction
    df_pat = df_pat.pivot(index=('PatID', 'Contour','ContourType',
                                 'Feature'), columns='Fraction', values='FeatureValue')
    df_pat = df_pat.reset_index()
    
    # calculate delta
    df_pat['Delta'] = (df_pat[1] - df_pat[5]) / df_pat[1]

    df_pat.drop(columns=[1, 5], inplace=True)
    df_pat.rename(columns={'Delta': 'FeatureValue'}, inplace=True)
    df_pat['Fraction'] = 'Delta'
    
    return df_pat



####################################################

def get_delta_data(output_path):
    '''
    Load delta data if already extracted
    '''

    delta_path = output_path

    if os.path.exists(os.path.join(delta_path, 'features', 'features_all.csv')):
        
        df_all = pd.read_csv(os.path.join(delta_path, 'features', 'features_all.csv'))            

    else:
        data_path = output_path.replace('-Delta', '')
        data_path = os.path.join(data_path, 'features', 'features_all.csv')

        df_all = pd.DataFrame()

        if os.path.exists(data_path):
            df_all_frac = pd.read_csv(data_path)
            df_all_frac = df_all_frac[df_all_frac['Fraction'].isin([1,5])]

            patIDs = df_all_frac['PatID'].unique()

            for pat in patIDs:
                df_pat = df_all_frac[df_all_frac['PatID'] == pat]
                df_pat = calc_delta(df_pat)

                df_all = pd.concat([df_all, df_pat], axis=0)
            
            df_all.to_csv(os.path.join(delta_path, 'features', 'features_all.csv'), index=False)
        
        else:
            print('Error loading features!')

            return None

    return df_all

####################################################

    


####################################################

# def All(DataRoot, Norm, tag):

#     root = DataRoot
#     # Patient Key
#     patKey = pd.read_csv(root + "Aaron\\ProstateMRL\\Code\\PatKeys\\AllPatientKey_s.csv")
#     niftiDir = root + "prostateMR_radiomics\\nifti\\"
#     outDir = root + "Aaron\\ProstateMRL\\Data\\Paper1\\" + Norm + "\\features\\"
#     # filter only SABR patients
#     patKey = patKey[patKey["Treatment"] == "SABR"]

#     # loop through all patients
#     patIDs = UF.SABRPats()  
#     results_df = pd.DataFrame()

#     if "Filter" in tag:
#         extractor_params = root + "Aaron\\ProstateMRL\Code\features\\Parameters\\All_Filters.yaml"
#     else:
#         extractor_params = root + "Aaron\\ProstateMRL\Code\features\\Parameters\\All.yaml"
#     extractor = featureextractor.RadiomicsFeatureExtractor(extractor_params)

#     for pat in tqdm(patIDs):
#         p_df = patKey[patKey["PatID"].isin([pat])]

#         p_vals = pd.DataFrame(columns=["PatID", "Scan", "Fraction", "Days", "Mask"])
#         # get file directory for patient
#         patDir = p_df["FileDir"].values[0]

#         # get scans
#         scans = p_df["Scan"].values
#         fractions = p_df["Fraction"].values
#         days = p_df["Days"].values

#         pat = UF.FixPatID(pat, patDir)       
#         patDir = niftiDir + patDir + "\\" + pat + "\\"

#         for j in range(len(scans)):
#             scan = scans[j]
#             frac = fractions[j]
#             day = days[j]

#             # get the scan directory
#             scanDir = patDir + scan + "\\"

#             # image file
#             imgFile = scanDir + Norm + "\\" + pat + "_" + scan + "_" + Norm + ".nii"

#             # mask files
#             RP_mask = scanDir + "Masks\\"  + pat + "_" + scan + "_shrunk_pros.nii"

#             # create a new row for the dataframe
#             new_row = {"PatID": pat, "Scan": scan, "Fraction": frac, "Days": day, "Mask": "RP"}

#             feat_df = pd.DataFrame()
#             # extract features
#             print("Pat: {} Scan: {}".format(pat, scan))
#             temp_results = pd.Series(extractor.execute(imgFile, RP_mask, label=255))
#             feat_df = feat_df.append(temp_results, ignore_index=True)
#             # merge new row with feature dataframe with new row first
#             feat_df = pd.concat([pd.DataFrame(new_row, index=[0]), feat_df], axis=1)
#             # append to the patient dataframe
#             p_vals = p_vals.append(feat_df, ignore_index=True)
    
#         # append to the results dataframe
#         results_df = results_df.append(p_vals, ignore_index=True)
    

#     # save the results and merge to patient key

#     results_df = results_df.drop(columns = [col for col in results_df.columns if "diagnostics" in col])
#     results_df = results_df.drop(columns = [col for col in results_df.columns if "Unnamed" in col])

#     results_df_w = results_df.sort_values(by = ["PatID", "Fraction", "Days"])

#     # save the results
#     #results_df.to_csv(outDir + "Longitudinal_fts_w.csv")
    

#     results_df_l = results_df.melt(id_vars = ["PatID", "Scan", "Days", "Fraction", "Mask"], var_name = "Feature", value_name = "FeatureValue")
#     fts = results_df_l["Feature"].unique()
#     PatIDs = results_df_l["PatID"].unique()
#     df_out = pd.DataFrame()
#     # loop through all patients
#     for pat in PatIDs:
#         df_pat = results_df_l[results_df_l["PatID"].isin([pat])]
#         df_pat = df_pat.sort_values(by = ["Days", "Fraction"])
#         # loop through all features
#         for ft in fts:
#             vals_ft = df_pat[df_pat["Feature"] == ft]["FeatureValue"].values
#             if vals_ft[0] == 0:
#                 ft_change = np.zeros(len(vals_ft))
#             else:
#                 ft_change = (vals_ft - vals_ft[0]) / vals_ft[0]
                        
#             df_pat.loc[df_pat["Feature"] == ft, "FeatureChange"] = ft_change

#         df_out = df_out.append(df_pat)

#     df_out.to_csv(outDir + "\\Longitudinal_All_fts_" + tag + ".csv", index=False)

#     return results_df_w, df_out

# ####################################################

# def Limbus(DataRoot, Norm, tag):
#     # Patient Key
#     root = DataRoot
#     patKey = pd.read_csv(root + "\\Aaron\\ProstateMRL\\Code\\PatKeys\\LimbusKey_s.csv")
#     niftiDir = root + "prostateMR_radiomics\\nifti\\"
#     outDir = root + "Aaron\\ProstateMRL\\Data\\Paper1\\" + Norm + "\\features\\"

#     # filter only SABR patients
#     patKey = patKey[patKey["Treatment"] == "SABR"]

#     # loop through all patients
#     PatIDs = patKey["PatID"].unique()[0:10]
#     results_df = pd.DataFrame()

#     if "Filters" in tag:
#         extractor_params = root + "Aaron\\ProstateMRL\Code\features\\Parameters\\All_Filters.yaml"
#     else:
#         extractor_params = root + "Aaron\\ProstateMRL\Code\features\\Parameters\\All.yaml"
#     extractor = featureextractor.RadiomicsFeatureExtractor(extractor_params)

#     for pat in tqdm(PatIDs):
#         p_df = patKey[patKey["PatID"].isin([pat])]
#         p_vals = pd.DataFrame(columns=["PatID", "Scan", "Fraction", "Days", "Mask"])
#         # get file directory for patient
#         patDir = p_df["FileDir"].values[0]

#         # get scans
#         scans = p_df["Scan"].values
#         fractions = p_df["Fraction"].values
#         days = p_df["Days"].values

#         pat = UF.FixPatID(pat, patDir)       
#         patDir = niftiDir + patDir + "\\" + pat + "\\"

#         for j in range(len(scans)):
#             scan = scans[j]
#             frac = fractions[j]
#             day = days[j]
#             print(pat, j)

#             # get the scan directory
#             scanDir = patDir + scan + "\\"

#             # image file
#             imgFile = scanDir + Norm + "\\" + pat + "_" + scan + "_" + Norm + ".nii"

#             # mask files
#             RP_mask = scanDir + "Masks\\"  + pat + "_" + scan + "_shrunk_pros.nii"
#             Limbus_mask = scanDir + "Masks\\" +  pat + "_" + scan + "_Limbus_shrunk.nii"
#             masks = [RP_mask, Limbus_mask]

#             for m in masks:
#                 # get the mask name
#                 if m == RP_mask:
#                     maskName = "RP"
#                 else:
#                     maskName = "Limbus"
#                 # create a new row for the dataframe
#                 new_row = {"PatID": pat, "Scan": scan, "Fraction": frac, "Days": day, "Mask": maskName}

#                 feat_df = pd.DataFrame()
#                 # extract features
#                 temp_results = pd.Series(extractor.execute(imgFile, m, label=255))
#                 feat_df = feat_df.append(temp_results, ignore_index=True)
#                 # merge new row with feature dataframe with new row first
#                 feat_df = pd.concat([pd.DataFrame(new_row, index=[0]), feat_df], axis=1)
#                 # append to the patient dataframe
#                 p_vals = p_vals.append(feat_df, ignore_index=True)
    
#         # append to the results dataframe
#         results_df = results_df.append(p_vals, ignore_index=True)

#     # save the results

#     results_df = results_df.drop(columns = [col for col in results_df.columns if "diagnostics" in col])
#     results_df = results_df.drop(columns = [col for col in results_df.columns if "Unnamed" in col])

#     results_df_w = results_df.sort_values(by = ["PatID", "Fraction", "Days"])

#     # save the results
#     #results_df_w.to_csv(outDir + "Longtidunial_Limbus_fts_w.csv")

#     results_df_l = results_df.melt(id_vars = ["PatID", "Scan", "Days", "Fraction", "Mask"], var_name = "Feature", value_name = "FeatureValue")
#     fts = results_df_l["Feature"].unique()
#     PatIDs = results_df_l["PatID"].unique()
#     df_out = pd.DataFrame()
#     # loop through all patients
#     for pat in PatIDs:
#         df_pat = results_df_l[results_df_l["PatID"].isin([pat])]
#         df_pat = df_pat.sort_values(by = ["Days", "Fraction"])
#         for m in df_pat["Mask"].unique():
#             df_pat_m = df_pat[df_pat["Mask"] == m]

#             # loop through all features
#             for ft in fts:
#                 vals_ft = df_pat_m[df_pat_m["Feature"] == ft]["FeatureValue"].values
#                 if vals_ft[0] == 0:
#                     ft_change = np.zeros(len(vals_ft))
#                 else:
#                     ft_change = (vals_ft - vals_ft[0]) / vals_ft[0]
                            
#                 df_pat_m.loc[df_pat_m["Feature"] == ft, "FeatureChange"] = ft_change

#             df_out = df_out.append(df_pat_m)

#     df_out.to_csv(outDir + "Longitudinal_Limbus_fts_" + tag + ".csv" , index=False)

#     return results_df_w, df_out

# ####################################################

# def DeltaValues(root, Norm, tag):
#     df_all = pd.read_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\features\Longitudinal_All_fts_" + tag + ".csv")
#     PatIDs = df_all["PatID"].unique()
#     fts = df_all["Feature"].unique()

#     df_out = pd.DataFrame()

#     # loop through all patients
#     for pat in PatIDs:
#         df_pat = df_all[df_all["PatID"] == pat]
#         f1,f2 = df_pat["Fraction"].values[0], df_pat["Fraction"].values[-1]
#         df_pat = df_pat[df_pat["Fraction"].isin([f1,f2])]
        
#         df_out = df_out.append(df_pat)

#     df_out.to_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\features\Delta_All_fts_" + tag + ".csv", index = False)


#     df_lim = pd.read_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\features\\Longitudinal_Limbus_fts_" + tag + ".csv")
#     PatIDs = df_lim["PatID"].unique()
#     fts = df_lim["Feature"].unique()

#     df_out = pd.DataFrame()

#     # loop through all patients
#     for pat in PatIDs:
#         df_pat = df_lim[df_lim["PatID"] == pat]
    
#         f1,f2 = df_pat["Fraction"].values[0], df_pat["Fraction"].values[-1]
#         df_pat = df_pat[df_pat["Fraction"].isin([f1,f2])]
        
#         df_out = df_out.append(df_pat)

#     df_out.to_csv(root + "Aaron\ProstateMRL\Data\Paper1\\" + Norm + "\\features\Delta_Limbus_fts_" + tag + ".csv", index = False)

# ####################################################
