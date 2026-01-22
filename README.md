# Medical_Insurance_Cost_Prediction

## **Data Loading, Preprocessing, and Feature Selection Workflow**

This project begins by importing essential Python libraries for data manipulation, visualization, and machine learning. The dataset is sourced directly from a public GitHub repository containing medical insurance records, including demographic attributes, lifestyle factors, and corresponding insurance charges. This project builds a complete end‑to‑end machine learning workflow to predict medical insurance charges using the dataset.

The focus is on:
- Clean preprocessing
- Correct handling of categorical variables
- Avoiding data leakage
- Applying multiple feature‑selection techniques
- Training and comparing several regression models
- Selecting the best model based on RMSE and R²

**1. Import Required Libraries**

The workflow uses:
- Pandas, NumPy for data handling
- Matplotlib, Seaborn for visualization
- Scikit‑learn for preprocessing, feature selection, and modeling
- XGBoost for gradient‑boosted regression

**2. Load & Inspect the Dataset**

After loading the dataset:
- Data types and structure are reviewed
- Null values are checked
- First few rows are displayed for initial understanding
  
The dataset is confirmed to be clean and ready for encoding


**3. Encoding Categorical Variables**

Categorical columns (sex, smoker, region) are converted into numerical format using one‑hot encoding with drop_first=True to avoid multicollinearity.
This results in a fully numerical dataset suitable for machine learning models.

**4. Defining Features and Target Variable**

- Target: charges
- Features: All remaining columns after encoding serve as input features.

**5. Standardize the Dataset**
  
All features are scaled using StandardScaler to ensure:
- Mean = 0
- Standard deviation = 1
  
This step is essential for regularized models and gradient‑based algorithms.

**6. Train–Test Split (Before Feature Selection)**

To prevent data leakage, the dataset is split into:
- 80% training data
- 20% testing data
  
Feature selection is performed only on the training set.

**7. Feature Selection Methods**

Three different techniques are applied to identify the most important predictors:
1. Lasso Regression (L1 Regularization)
- Selects features with non‑zero coefficients
  
2. Ridge Regression (L2 Regularization)
- Ranks features by coefficient magnitude
- Top 8 features are selected
  
3. Random Forest Regressor
- Uses impurity‑based feature importance
- Top 8 features are selected
  
The union of all selected features forms the final feature set.

**8. Prepare Final Train/Test Data**

Training and testing datasets are subset to include only the selected features.
This reduces dimensionality and improves interpretability.

**9. Baseline Model (All Features)**

A baseline Linear Regression model is trained using all scaled features.
Its performance metrics (RMSE, MAE, R²) serve as a benchmark for comparison.

**10. Regression Models Using Selected Features**

The following models are trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
  
Each model is evaluated using:
- RMSE
- MAE
- R² Score

**11. Performance Comparison**

Performance is summarized in a results table and visualized using:
- RMSE bar chart
- R² bar chart
- MAE bar chart
  
These visualizations clearly show how each model performs relative to the baseline.

**12. Final Model Selection**

Based on the evaluation:
- XGBoost achieved the lowest RMSE
- XGBoost achieved the highest R² Score
  
**Final Decision:**

XGBoost is selected as the best-performing model for predicting medical insurance costs.
