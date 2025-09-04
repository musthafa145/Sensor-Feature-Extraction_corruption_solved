Bearing Dataset: Corruption Recovery and Feature Extraction
Overview

This repository addresses the challenges posed by corrupted bearing vibration data and demonstrates the application of advanced signal processing techniques to recover and extract meaningful features for machine learning tasks.

📂 Dataset

The dataset used is the Bearing Dataset
, which contains vibration signals from bearings under various conditions. The raw data was found to have corruption issues, necessitating preprocessing and recovery steps.

🛠️ Corruption Issue and Recovery

Upon inspection, it was identified that the dataset contained corrupted entries, leading to erroneous readings. The following steps were undertaken to address this:

Identification of Corrupted Entries: Through exploratory data analysis, rows with missing or anomalous values were flagged.

Data Recovery: Missing values were imputed using statistical methods, and outliers were handled to restore data integrity.

🔧 Preprocessing Techniques

To prepare the data for feature extraction, the following preprocessing steps were applied:

Signal Denoising: Utilized the PyWavelets (pywt) library to perform wavelet transform-based denoising, effectively reducing noise while preserving important signal features.

import pywt
import numpy as np

def denoise_signal(signal, wavelet='db8', level=1):
    coeff = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * (1 / 2)
    coeff[1:] = (pywt.threshold(i, threshold, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet)


Normalization: Applied Min-Max scaling to standardize the range of feature values.

📊 Feature Extraction

Post-preprocessing, the following feature extraction techniques were employed:

Manual Feature Extraction: Computed basic statistical features such as mean, standard deviation, skewness, kurtosis, and energy for each signal channel.

Automated Feature Extraction with tsfresh: Leveraged the tsfresh
 library to automatically extract a wide range of time-series features, including:

Statistical features: mean, variance, skewness, kurtosis

Temporal features: autocorrelation, Fourier coefficients

Entropy measures: permutation entropy, sample entropy

from tsfresh import extract_features

extracted_features = extract_features(df, column_id='id', column_sort='time')


This approach automates the extraction of hundreds of features, facilitating the identification of the most relevant ones for predictive modeling.

🗂️ Outputs

The following output files have been generated:

manual_features.csv: Contains manually extracted statistical features.

tsfresh_features.csv: Contains features extracted using tsfresh.

combined_features.csv: A consolidated file combining both manual and tsfresh features.

These files are available for download in the repository.

🚀 Next Steps

With the preprocessed and feature-engineered dataset, the next steps involve:

Feature Selection: Identifying the most relevant features for predictive modeling.

Model Training: Applying machine learning algorithms to train models for bearing fault classification.

Model Evaluation: Assessing model performance using appropriate metrics.
