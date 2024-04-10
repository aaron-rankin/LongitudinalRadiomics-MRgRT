import pandas as pd
import numpy as np
import os

def compare_ft_list(list1, list2):
    '''
    Compare two lists of features
    '''
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    
    # get common features
    common = set1 & set2

    # get features in list1 but not in list2
    only_list1 = set1 - set2

    # get features in list2 but not in list1
    only_list2 = set2 - set1

    return common, only_list1, only_list2

def get_ft_lists(ft_dir):
    '''
    Get feature lists from two directories
    0 - ICC features
    1 - Volume features
    2 - Selected features
    '''
    fts_icc = pd.read_csv(os.path.join(ft_dir, 'Manual_ICC_Names.csv'))
    fts_vol = pd.read_csv(os.path.join(ft_dir, 'Manual_VolCorr_TP_Names.csv'))
    fts_sel = pd.read_csv(os.path.join(ft_dir, 'Features_Selected.csv'))

    fts_icc = fts_icc['Feature'].tolist()
    fts_vol = fts_vol['Feature'].tolist()
    fts_sel = fts_sel['Feature'].tolist()

    return fts_icc, fts_vol, fts_sel

def output_fts(ft_list):
    '''
    print each ft
    '''
    for ft in ft_list:
        print(ft)

    print('='*30)


def main():
    '''
    Main function to compare feature sets
    '''
    
    tag1 = 'Submission-Delta' # add -Delta to tag
    tag2 = 'Submission' 

    dir1 = os.path.join(os.getcwd(), 'Output', tag1, 'features')
    dir2 = os.path.join(os.getcwd(), 'Output', tag2, 'features')

    fts_icc1, fts_vol1, fts_sel1 = get_ft_lists(dir1)
    fts_icc2, fts_vol2, fts_sel2 = get_ft_lists(dir2)

    print('#'*30)
    print('Comparing ICC features...')
    print('#'*30)
    common_icc, only_icc1, only_icc2 = compare_ft_list(fts_icc1, fts_icc2)

    print(f'Common ICC features: {len(common_icc)}')
    output_fts(common_icc)

    print(f'Only in ICC1: {len(only_icc1)}')
    output_fts(only_icc1)

    print(f'Only in ICC2: {len(only_icc2)}')
    output_fts(only_icc2)


    print('#'*30)
    print('Comparing Volume features...')
    print('#'*30)
    common_vol, only_vol1, only_vol2 = compare_ft_list(fts_vol1, fts_vol2)

    print(f'Common Volume features: {len(common_vol)}')
    output_fts(common_vol)

    print(f'Only in Volume1: {len(only_vol1)}')
    output_fts(only_vol1)

    print(f'Only in Volume2: {len(only_vol2)}')
    output_fts(only_vol2)

    print('#'*30)
    print('Pre-feature Selection...')
    fts_pre1 = fts_icc1 + fts_vol1
    fts_pre2 = fts_icc2 + fts_vol2

    common_pre, only_pre1, only_pre2 = compare_ft_list(fts_pre1, fts_pre2)

    print(f'Common Pre-Selected features: {len(common_pre)}')
    output_fts(common_pre)

    print('#'*30)
    print('Comparing Selected features...')
    print('#'*30)
    common_sel, only_sel1, only_sel2 = compare_ft_list(fts_sel1, fts_sel2)

    print(f'Common Selected features: {len(common_sel)}')
    output_fts(common_sel)

    print(f'Only in Selected1: {len(only_sel1)}')
    output_fts(only_sel1)

    print(f'Only in Selected2: {len(only_sel2)}')
    output_fts(only_sel2)




if __name__ == '__main__':
    main()

    