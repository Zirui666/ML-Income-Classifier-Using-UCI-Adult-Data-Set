# ML-Income-Classifier-Using-UCI-Adult-Data-Set
## Overview
### This project predicts whether an individual's income exceeds $50,000 based on demographic and occupational attributes. Using lasso regression, ridge regression, and random forest, the models are evaluated for classification accuracy and generalization.

## R Script Overview
### 1. Preprocessing:
### Drops unnecessary columns and converts categorical variables to binary indicators.
### Creates additional features like age_sq (age squared) and standardizes numeric variables.

### 2. Data Splitting:
### Splits data into training (70%) and testing (30%) sets. the best regularization parameter (λ).

### 3. Model Training:
### Trains lasso and ridge regression models with 10-fold cross-validation to find the best λ.
### Trains random forest models with varying hyperparameters.

### 4. Evaluation:
### Compares model performance using classification accuracy and confusion matrices.
### Best Test Accuracy: Ridge Regression (81.07%).
