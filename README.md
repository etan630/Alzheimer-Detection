# Alzheimers Project

Alzheimer’s disease is a progressive neurodegenerative disorder, and early detection is critical for patient care, yet it is often diagnosed too late due to the difficulty of identifying subtle early-stage brain degeneration.

We propose a machine learning model to analyze images for early-stage Alzheimer’s detection, focusing on subtle patterns of brain degeneration that may be overlooked by doctors.

## Setting up for windows gpu

make new environment with python 3.9

run:

`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

`pip install "tensorflow<2.11" `

`conda install --yes --file requirements.txt`

`pip install numpy --upgrade`

this is because scikit image, tensorflow, and numpy don't really like each other together.
There are a couple of weird version conflicts they have to deal with. If we don't need to use
scikit image, then we can just not have it. 

## Relevant Directories and Files

/Data/ - The directory where the MRI scans are sorted into different classes
1. Mild Dementia
2. Moderate Dementia
3. Non Demented
4. Very mild Dementia

/notebooks/ - The directory containing the different ML models
1. /notebooks/1_CNN.ipynb - The CNN Model notebook
2. /notebooks/2_Logistic_Regression_Balanced.ipynb - The logistic regression model trained on a balanced dataset notebook
3. /notebooks/2_Logistic_Regression_Imbalanced.ipynb - The logistic regression model trained on the original, imbalanced dataset notebook
4. /notebooks/3_SVM.ipynb - The SVM Model notebook
5. /notebooks/preprocessing.py - Where the prepocessing of data for the CNN model lies. 
6. /notebooks/preprocessing.py - Where the prepocessing of data for the SVM model lies. 
7. /notebooks/testmodel.py - Runs the Gradcam algorithm on one image

/environment.yml - The configuration file.

/requirements.txt - The requirements needed to run the notebooks.
