# Longitudinal Radiomics - MRgRT

There is an unmet need for radiomics pipelines to analyse the __full trajectory__ of features acquired during treatment. Here we propose a method for finding representative feature trajectories from prostate cancer patients treated with MRgRT. 

Pipeline
--------
1. __On-treatment__ images were collected
2. Prostate gland was delineated (a manual and auto contour)
3. 105 features were extracted from each image using the contours
4. ICC values between the two masks were computed - _features removed if they had a mean ICC < 0.75_ across treatment.
5. Spearman rank correlation was assessed between all feature values and volume - _features were removed if they had a mean correlation > 0.6_ across treatment.
6. For each patient, the __Euclidean distance__ between remaining feature trajectories was computed.
7. These distance values were then used to group similar feature trajectories using __hierarchical clustering__.
8. The most representative features in each cluster were determined by finding the feature with the highest mean Pearson correlation.
9. Each patient passes through a set of features.
10. Features were ranked based on the number of occurences across the cohort and features found to be representative across 30% of patients were selected.

    ![alt text](https://github.com/aaron-rankin/LongitudinalRadiomics-MRgRT/blob/main/PaperMaterial/Pipeline3%20(Small)%20(Phone).png)
