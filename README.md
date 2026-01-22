# Medical_Insurance_Cost_Prediction

## **Data Loading, Preprocessing, and Feature Selection Workflow**

This project begins by importing essential Python libraries for data manipulation, visualization, and machine learning. The dataset is sourced directly from a public GitHub repository containing medical insurance records, including demographic attributes, lifestyle factors, and corresponding insurance charges.

**1. Data Loading and Initial Inspection**

The dataset is loaded into a Pandas DataFrame and inspected to understand:
- Column names and data types
- Presence of missing values
- Basic structure and sample records
  
The dataset contains no null values, ensuring a clean starting point for preprocessing.

**2. Encoding Categorical Variables**

Categorical features such as sex, smoker, and region are converted into numerical form using one‑hot encoding. To avoid multicollinearity, the first category of each variable is dropped. After encoding, the dataset expands to include binary indicator columns representing each category.

**3. Defining Features and Target Variable**

The target variable for prediction is:
- charges — the medical insurance cost.
  
All remaining columns serve as input features.

**4. Feature Scaling**

To ensure consistent model performance, all numerical features are standardized using StandardScaler, transforming them to have:
- Mean = 0
- Standard deviation = 1
  
This step is crucial for models sensitive to feature magnitude, such as Lasso, Ridge, and gradient‑based algorithms.

**5. Train–Test Split (Before Feature Selection)**

To prevent data leakage, the dataset is split into training and testing sets before performing any feature selection. This ensures that feature selection is based solely on training data and does not inadvertently use information from the test set.

**6. Feature Selection Using Three Methods**

Three complementary techniques are applied to identify the most influential predictors of insurance cost:
Lasso Regression (L1 Regularization)
- Performs aggressive coefficient shrinkage
- Selects only features with non‑zero coefficients
Ridge Regression (L2 Regularization)
- Penalizes large coefficients
- Ranks features by importance
- Top 8 features are selected
Random Forest Regressor
- Captures non‑linear relationships
- Ranks features using impurity‑based importance
- Top 8 features are selected
A union of all selected features from the three methods forms the final feature set used for modeling.

**7. Preparing Final Training and Testing Data**

The training and testing datasets are subset to include only the selected features. This reduces dimensionality and improves model interpretability while retaining predictive power.

**8. Baseline Model (All Features)**

A baseline Linear Regression model is trained using all scaled features. Its performance (RMSE, MAE, R²) serves as a benchmark to compare against models trained on selected features.

**9. Regression Models Trained on Selected Features**

Multiple regression models are trained and evaluated using only the selected features:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
  
Each model is evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

**10. Performance Comparison and Visualization**

Model performance is summarized in a comparison table and visualized using horizontal bar charts for:
- RMSE (lower is better)
- R² Score (higher is better)
- MAE (optional but included for completeness)
These visualizations highlight how each model performs relative to the baseline and to one another.

**11. Final Model Selection**

Based on evaluation metrics:
- XGBoost achieves the lowest RMSE
- XGBoost achieves the highest R² Score
  
**Final Decision**

XGBoost is selected as the best-performing model for predicting medical insurance costs.
