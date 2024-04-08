'''
Delta radiomics main script
---------------------------

Loads in feature files, gets changes in features between first and last fraction 
Performs feature reduction steps
Performs feature selection

AGR 8/4/2024
'''
import seaborn as sns
import matplotlib.pyplot as plt
from Functions import extraction as extr, reduction as red, delta_model as dm

def main():
    '''
    Main function to run delta pipeline
    '''
    tag = 'Submission-Delta' # add -Delta to tag

    output_path = extr.create_output_dirs(tag)

    print('-'*30)
    print('Getting feature data...')

    df_all = extr.get_delta_data(output_path)

    df_all = extr.format_df_all(df_all, rescale=True, output_path=output_path)

    df_man, df_auto = extr.split_df_all(df_all)

    print('-'*30)

    print('Feature reduction...')

    # Volume Correlation
    print('Manual Contours: ')
    remove_fts_vol_man = red.volume_corr_tps(df_man, output_path=output_path)

    # ICC
    remove_fts_icc = red.icc_calculation(df_all, output_path)

    # Remove features
    remove_fts_total = list(set(remove_fts_icc) | set(remove_fts_vol_man))

    df_man = red.remove_fts(df_man, remove_fts_total, output_path)

    print('-'*30)
    print('Feature selection...')

    df_corr = dm.correlation_matrix(df_man, output_path, plot=False)
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_corr)
    plt.show()
    # dm.feature_selection(df_corr, output_path)

    print('Delta Pipeline Complete')

if __name__ == '__main__':
    main()

