import pandas as pd
import os
from Functions import Extraction as FE
from tqdm import tqdm

# read in key
cwd = os.path.dirname(os.path.abspath(__file__))
output_path = cwd + "/Output/Long-Test/"

# get key
key = pd.read_csv(cwd + "/Input/Default/PepKey_Lim.csv")

extraction_path = output_path + "/Extraction/"
if not os.path.exists(extraction_path):
    os.makedirs(extraction_path)

params_extractor = cwd + "/Input/Default/Default_ExtractionParams.yaml"

# Loop over all patients
print("Extracting features for patients...")
for pat in tqdm(key["PatID"].unique()):

    # Get the patient's key
    key_pat = key[key["PatID"] == pat]
    Features_pat = pd.DataFrame()
    # loop over all rows
    for i, row in key_pat.iterrows():
        PatID = row["PatID"]
        Fraction = row["Fraction"]
        Mask = row["Contour"]
        ContourType = row["ContourType"]
        ImagePath = row["ImagePath"]
        MaskPath = row["MaskPath"]

        # Extract features
        Features = FE.ExtractFeatures(PatID, Fraction, Mask, ContourType, ImagePath, MaskPath, params_extractor)

        Features_pat = Features_pat.append(Features)
    
    # Save the features to parquet
    if os.path.exists(output_path + "/Extraction/") == False:
        os.mkdir(output_path + "/Extraction/")
        
    Features_pat.to_csv(output_path + "/Extraction/" + ContourType + "_" + str(pat) + ".csv", index=False)
