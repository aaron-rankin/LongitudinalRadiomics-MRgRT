'''
MR-Longitudinal Radiomics
-------------------------
Radiomics pipeline created for analysis of on-treatment MR images of patients with prostate cancer.
Feature extraction is performed using pyRadiomics
Feature reduction done by comparing ICC values and volume correlations
Feature selection done by clustering similar feature trajectories

AGR 03/04/2024
'''

from Functions import extraction as extr, reduction as red, longitudinal_model as lm
import numpy as np
import os

EXTRACT = False

# T_VALS = np.arange(1.75, 2.51, 0.25)
T_VALS = [1.75]

def main():
    '''
    Main function to run the pipeline
    '''
    for t_val in T_VALS:
        print('-'*30)
        print(f'Running pipeline for T = {t_val}')
        tag = f"Submission-Long"

        output_path = extr.create_output_dirs(tag)

        print('-'*30)
        print('Getting feature data...')

        if EXTRACT:
            df_all = extr.extract_features(output_path, key_path)
        else:
            df_all = extr.get_df_all(output_path)

        df_all = extr.format_df_all(df_all, rescale=True, output_path=output_path)

        df_man, df_auto = extr.split_df_all(df_all)
        df_man.to_csv(os.path.join(output_path, 'features_rescaled.csv'), index=False)

        print('-'*30)
        print('Feature reduction...')

        # Volume correlation
        print('Manual Contours: ')
        remove_fts_vol_tp = red.volume_corr_tps(df_man, output_path)
        remove_fts_vol_traj  = red.volume_trajectory(df_man, output_path)
        
        print('Auto Contours: ')
        #auto_vol_tp = red.volume_corr_tps(df_auto, output_path)
        #auto_vol_traj = red.volume_trajectory(df_auto, output_path)

        # ICC
        remove_fts_icc = red.icc_calculation(df_all, output_path)

        # Remove features
        remove_fts_total = list(set(remove_fts_vol_tp) | set(remove_fts_vol_traj) | set(remove_fts_icc))
        # remove_fts_total_auto = list(set(auto_vol_tp) | set(auto_vol_traj) | set(remove_fts_icc))
        
        df_man = red.remove_fts(df_man, remove_fts_total, output_path)
        # df_auto = red.remove_fts(df_auto, remove_fts_total_auto, output_path)

        print('-'*30)
        print('Feature selection...')


        # Longitudinal model
        print('Computing Euclidean distances:')
        lm.distance_matrices(df_man, output_path, plot=False)
        
        lm.cluster_features(df_man, output_path, t_val)
        
        print('Selecting features:')
        
        lm.select_features(df_man, output_path)
        lm.count_clusters(output_path)
        print('-'*30)
        print('Pipeline complete.')
        print('-'*30)


main()
