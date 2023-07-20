import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from scipy import stats
from tqdm import tqdm

####################################################

def ICC(df_all, output_path, plot=False):
    """
    Measure the stability of the feature values over all fractions by calculating the
    intraclass correlation coefficient (ICC) for each feature between manual and automatic contours 
    for each fraction.
    ICC is calculated using the two-way mixed model.
    Requires 2 values for each feature for each fraction.
    Compares the values for each feature depending on the contour used (manual or automatic).
    Then calculates mean ICC over all fractions.
    Features removed if mean ICC < 0.5.

    Returns a list of features to remove to a csv file.
    
    df_all: dataframe containing all features
    output_path: path to save output 
    tag: tag to add to output file
    plot: if True, plot the ICC for each feature against fraction, red if feature removed, green if not
    """
    print("-" * 30)
    print("Stability Test")
    print("Calculating ICC...")
    features = df_all["Feature"].unique()
    fractions = df_all["Fraction"].unique()

    df_res = pd.DataFrame(columns=["Feature", "Fraction", "ICC"])

    for fr in fractions:
        df_fr = df_all.loc[df_all["Fraction"] == fr]
        
        for ft in tqdm(features):
            df_ft = df_fr.loc[df_fr["Feature"] == ft]
            icc = pg.intraclass_corr(data=df_ft, targets="PatID", raters="ContourType", ratings="FeatureValue")
            df_res = df_res.append({"Feature": ft, "Fraction": fr, "ICC": icc["ICC"][0]}, ignore_index=True)

    df_res["ICC_Class"] = df_res["ICC"].apply(lambda x: "Poor" if x < 0.5 else ("Moderate" if x < 0.75 else ("Good" if x < 0.9 else "Excellent")))

    df_res["MeanICC"] = df_res.groupby("Feature")["ICC"].transform("mean")
    df_res["MeanICC_Class"] = df_res["MeanICC"].apply(lambda x: "Poor" if x < 0.5 else ("Moderate" if x < 0.75 else ("Good" if x < 0.9 else "Excellent")))

    df_res = df_res.sort_values(by=["Feature", "Fraction"], ascending=True)
    df_res["Remove"] = ["Yes" if x < 0.75 else "No" for x in df_res["MeanICC"]]

    # return rows with poor ICC
    fts_remove = df_res[df_res["Remove"] == "Yes"]["Feature"].values
    fts_remove = np.unique(fts_remove)

    # save features to csv
    fts_remove_out = pd.DataFrame({"Feature": fts_remove})
    fts_remove_out.to_csv(output_path + "/Features/Rd_ICC_FeatureNames.csv", index=False)
    df_res.to_csv(output_path + "/Features/Rd_ICC_Values.csv", index=False)
    print("ICC redudant features: " + str(len(fts_remove)) + "/" + str(len(features)) )
    print("-" * 30)

    if plot == True:
        print("Plotting ICC Values...")
        if os.path.exists(output_path + "/Plots/ICC/") == False:
            os.mkdir(output_path + "/Plots/ICC/")
        
        pal = sns.color_palette("Set2", 2)
        pal = pal.as_hex()
        pal = pal[0:2]
        
        # plot ICC for each feature
        for ft in tqdm(features):
            plt.figure(figsize=(10,8))
            sns.set_style("darkgrid")
            sns.set_context("paper", font_scale=1.5)
            colour = pal[1] if ft in fts_remove else pal[0]
            sns.lineplot(x="Fraction", y="ICC", data=df_res[df_res["Feature"] == ft], color=colour, legend = False, linewidth=2.5)
            sns.scatterplot(x="Fraction", y="ICC", data=df_res[df_res["Feature"] == ft], color=colour, legend = False, s=100)
            plt.title(ft, fontsize=20)
            plt.xlabel("Fraction", fontsize=16)
            plt.xlim(0.95, 5.05)
            plt.xticks([1, 2, 3, 4, 5])
            plt.ylabel("ICC", fontsize=16)
            plt.ylim(0, 1.01)
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            plt.savefig(output_path + "/Plots/ICC/" + ft + ".png", dpi=300)
            plt.close()
        print("-" * 30)

####################################################

def Volume(df_all, output_path, plot=False):
    """
    Correlate feature values over treatment with volume of manual prostate mask by 
    plotting a scatter plot and calculating the Spearman correlation coefficient over all fractions.
    Calculates the mean rho for each feature and removes features with rho > 0.6.
    
    Returns a list of features to remove to a csv file.
    
    df_all: dataframe containing all features
    output_path: path to save output
    tag: tag to add to output file
    plot: if true, plot the volume correlation for each feature against fraction, red if feature removed, green if not
    """
    print("-" * 30)
    print("Volume Correlation")
    df_vol = df_all[df_all["Feature"] == "shape_MeshVolume"]

    features = df_all["Feature"].unique()
    fractions = df_all["Fraction"].unique()

    df_res = pd.DataFrame()
    print("Correlating features to volume...")

    for fr in fractions:
        # get volume for fraction
        vals_vol_fr = df_vol[df_vol["Fraction"] == fr]
        vals_vol_fr = vals_vol_fr["FeatureValue"].values

        df_fr = df_all[df_all["Fraction"] == fr]

        # loop through features
        for ft in tqdm(features):
            # vals for feature
            vals_ft_fr = df_fr[df_fr["Feature"] == ft]
            vals_ft_fr = vals_ft_fr["FeatureValue"].values

            # get spearman correlation
            rho = stats.spearmanr(vals_vol_fr, vals_ft_fr)[0]
            
            df_temp = pd.DataFrame({"Fraction": [fr], "Feature": [ft], "rho": [np.abs(rho)], })
            df_res = df_res.append(df_temp)

    # get mean rho over all fractions
    df_mean = df_res.groupby("Feature").mean().reset_index()
    df_mean = df_mean.rename(columns={"rho": "rho_mean"})
    df_mean.drop(["Fraction"], axis=1, inplace=True)
    df_mean = df_mean.sort_values(by="rho_mean", ascending=False)

    df_res = df_res.merge(df_mean, on="Feature", how="left")
    # sort by feature then fraction
    df_res = df_res.sort_values(by=["Feature", "Fraction"], ascending=True)
    df_res["Remove"] = ["Yes" if x > 0.6 else "No" for x in df_res["rho_mean"]]
    
    # if rho_mean > 0.6 find feature
    fts_remove = df_mean[df_mean["rho_mean"] > 0.6]["Feature"].values
    fts_remove = np.unique(fts_remove)
    
    # save features to csv
    fts_remove_out = pd.DataFrame({"Feature": fts_remove})
    fts_remove_out.to_csv(output_path + "/Features/Rd_VolCorr_FeatureNames.csv", index=False)
    df_res.to_csv(output_path + "/Features/Rd_VolCorr_Values.csv", index=False)
    print("Volume redundant features: " + str(len(fts_remove)) + "/" + str(len(features)) )
    print("-" * 30)

    if plot == True:
        print("Plotting volume correlation...")
        df_plot = df_all[["Feature", "Fraction", "FeatureValue"]]
        df_plot = df_plot.merge(df_res[["Feature", "Fraction", "rho", "Remove"]], on=["Feature", "Fraction"], how="left")
        if os.path.exists(output_path + "/Plots/VolCorr") == False:
            os.mkdir(output_path + "/Plots/VolCorr/")

        pal = sns.color_palette("Set2", 2)
        pal = pal.as_hex()
        pal = pal[0:2]


        for ft in tqdm(features):
            plt.figure(figsize=(10,8))
            sns.set_style("darkgrid")
            sns.set_context("paper", font_scale=1.5)
            colour = pal[1] if ft in fts_remove else pal[0]
            sns.lineplot(x="Fraction", y="rho", data=df_plot[df_plot["Feature"] == ft], color=colour, legend = False, linewidth=2.5)
            sns.scatterplot(x="Fraction", y="rho", data=df_plot[df_plot["Feature"] == ft], color=colour, legend = False, s=100)
            plt.title(ft, fontsize=20)
            plt.xlabel("Fraction", fontsize=16)
            plt.ylabel("Volume Correlation", fontsize=16)
            plt.ylim(0, 1, 1.01)
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            plt.xlim(0.95, 5.05)
            plt.xticks([1, 2, 3, 4, 5])
            plt.savefig(output_path + "/Plots/VolCorr/" + ft + ".png", dpi=300)
            plt.close()
        print("-" * 30)

####################################################        

def RemoveFts(df_all, output_path):
    """
    Remove features that have been identified as redundant.
    """
    print("-" * 30)
    print("Removing redundant features...")
    fts_ICC = pd.read_csv(output_path + "/Features/Rd_ICC_FeatureNames.csv")
    fts_ICC = fts_ICC["Feature"].values
    fts_Vol = pd.read_csv(output_path + "/Features/Rd_VolCorr_FeatureNames.csv")
    fts_Vol = fts_Vol["Feature"].values

    fts_remove = np.concatenate((fts_ICC, fts_Vol))
    fts_remove = np.unique(fts_remove)
    df_all = df_all[~df_all["Feature"].isin(fts_remove)]
    print("Number of features removed: " + str(len(fts_remove)))
    print("Number of features remaining: " + str(len(df_all["Feature"].unique())))
    print("-" * 30)
    
    return df_all

####################################################