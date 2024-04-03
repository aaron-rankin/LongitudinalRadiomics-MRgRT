'''
MR-Longitudinal Radiomics
-------------------------
Radiomics pipeline created for analysis of on-treatment MR images of patients with prostate cancer.
Feature extraction is performed using pyRadiomics
Feature reduction done by comparing ICC values and volume correlations
Feature selection done by clustering similar feature trajectories

AGR 03/04/2024
'''

import os
import pandas as pd
from tqdm import tqdm
from Functions import extraction as extr, reduction as red, longitudinal_model as lm


EXTRACT = False

def main():
    '''
    Main function to run the pipeline
    '''
    tag = "Submission"
    output_path = ext.create_output_dirs(tag)

    print('-'*30)
    print('Getting feature data...')

    if EXTRACT:
        df_all = extr.extract_features(output_path, key_path)
    else:
        df_all = extr.get_df_all(output_path)

    df_all = extr.format_df_all(df_all, rescale=True, output_path=output_path)

    df_man, df_auto = extr.split_df_all(df_all)

    print('-'*30)
    print('Feature reduction...')

    # Volume correlation
    print('Manual Contours: ')
    remove_fts_vol = red.volume_corr_tps(df_man, output_path)
    
    print('Auto Contours: ')
    _ = red.volume_corr_tps(df_auto, output_path)


    # ICC
    remove_fts_icc = red.icc_calculation(df_all, output_path)

    # Remove features
    remove_fts_total = list(set(remove_fts_vol) | set(remove_fts_icc))
    df_man = red.remove_fts(df_man, remove_fts_total, output_path)

    print('-'*30)
    print('Feature selection...')








    

    
if __name__ == '__main__':
    main()

