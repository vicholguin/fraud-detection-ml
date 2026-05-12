# Machine Learning Approach to Fraud Detection
CLASS: CS 487 / CS 519 – Applied Machine Learning I
AUTHORS: Mario Rodriguez, Said Ajo-Montano, Emma Jackson, Victoria Holguin
DATE: April 14th, 2025
DESCRIPTION: Machine Learning models trained to detect credit card fraud. 

## FILES INCLUDED:
- main.py
- tune.py
- requirements.txt
- README.md
- credit_card_fraud_dataset.csv 
- synthetic_fraud_dataset.csv

## SYSTEM REQUIREMENTS:
- Python 3.x
- pip (Python package installer)

## HOW TO SETUP AND RUN:
1. Ensure Python 3.x (version 3.10 or higher) is installed on your system.
2. Install required Python libraries: Open a terminal or command prompt and run the following command:
pip install -r requirements.txt
This will install all necessary packages listed in the 'requirements.txt' file.
3. Running the Script: This code was written using Visual Studio Code and assumes that all files above are in the same directory.
4. Navigate to the directory holding the main.py file and use the command py main.py to run the code (on Windows -- may be python3 on other operating systems)
5. To change which dataset is being used, modify this line in the code: file_path = "synthetic_fraud_dataset.csv"  (Update with correct file path)
6. To change the output file name, change this line: results_df.to_csv("model_comparison_results_fixed3.csv", index=False)

## EXPECTED OUTPUT:
                  Model   ROC-AUC  Precision    Recall  F1-Score       MCC    Time
    Logistic Regression  #.######   #.######  #.######  #.######  #.######    #.######
          Random Forest  #.######   #.######  #.######  #.######  #.######    #.######
       Gradient Boosting  #.######   #.######  #.######  #.######  #.######   #.######
       Support Vector Machine  #.######   #.######  #.######  #.######  #.######    #.######
       XGBoost  #.######   #.######  #.######  #.######  #.######    #.######

## tune.py
1. Follow steps 1-3 above, then navigate to the directory with tune.py
2. Use the command python tune.py to run the code
3. To change the dataset, modify this line with the new dataset name: file_path = "credit_card_fraud_dataset.csv"

## tune.py EXPECTED OUTPUT
Decision Tree:  {'max_depth': #, 'min_samples_split': #} -> 0.98995

Random Forest: {'max_depth': #, 'max_features': 'sqrt', 'n_estimators': #} -> 0.99

Gradient Boosting: {'learning_rate': #.##, 'max_depth': #, 'n_estimators': ##} -> 0.99

XGBoost: {'learning_rate': #.##, 'max_depth': #, 'n_estimators': ##, 'subsample': #.#} -> 0.99

Logistic Regression: {'C': #.##, 'max_iter': ###, 'penalty': 'l2', 'solver': 'lbfgs'} -> 0.99

## TROUBLESHOOTING:
- If the script fails to execute, check that all required packages are installed correctly and Python is updated
to the latest version.

### CONTACT: 
For any further assistance, please contact emma13@nmsu.edu, mariorod@nmsu.edu, sajomont@nmsu.edu, and vichol@nmsu.edu.


