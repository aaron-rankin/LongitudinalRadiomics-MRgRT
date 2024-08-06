import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from datetime import datetime
from radiomics import featureextractor
import sys
from tqdm import tqdm


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
    print('Loading features...')
    print(f'Output path: {output_path}')
    if os.path.exists(f'{output_path}/features/features_all.csv'):
        df_all = pd.read_csv(f'{output_path}/features/features_all.csv')
        
        return df_all
    
    elif os.path.exists(f'Output/Submission/features/features_all.csv'):
        df_all = pd.read_csv(f'Output/Submission/features/features_all.csv')
        
        return df_all
    else:
        print('Error loading features!')
        sys.exit()
        

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

