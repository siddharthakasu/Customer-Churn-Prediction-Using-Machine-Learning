
# Customer Churn Prediction using Machine Learning

This project aims to predict customer churn for a telecommunications company using various machine learning models. By analyzing customer data, the model can identify customers who are likely to discontinue their service. This allows the company to proactively engage with these customers to retain their business.

-----

## Dataset

The project uses the **"WA\_Fn-UseC\_-Telco-Customer-Churn.csv"** dataset. This dataset contains information about 7,043 customers and includes 21 features, such as:

  * **Customer Demographics:** gender, SeniorCitizen, Partner, Dependents
  * **Account Information:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
  * **Services Subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  * **Target Variable:** Churn (Yes/No)

-----

## Methodology

The project follows a systematic approach to build and evaluate the churn prediction model:

### 1\. Data Preprocessing

  * **Data Cleaning:** The `customerID` column was dropped as it is not relevant for modeling. Missing values in the `TotalCharges` column, which were represented as empty strings, were replaced with "0.0" and the column was converted to a float data type.
  * **Encoding Categorical Features:** Categorical features were converted into a numerical format using `LabelEncoder`. The encoders for each categorical feature were saved to a file named `encoders.pkl` for later use in the predictive system.
  * **Handling Class Imbalance:** The target variable, `Churn`, exhibited a class imbalance. To address this, the **Synthetic Minority Oversampling Technique (SMOTE)** was applied to the training data to create a balanced dataset for model training.

### 2\. Model Training and Evaluation

  * **Model Selection:** Three different machine learning models were evaluated for their performance in predicting customer churn:
      * Decision Tree Classifier
      * Random Forest Classifier
      * XGBoost Classifier
  * **Cross-Validation:** A 5-fold cross-validation was performed to assess the generalization performance of each model. The **Random Forest Classifier** achieved the highest cross-validation accuracy of **0.84**.

-----

## Results

The **Random Forest Classifier** was selected as the final model due to its superior performance in cross-validation. The model was then trained on the full SMOTE-augmented training dataset and evaluated on the held-out test set.

The performance of the trained Random Forest model on the test set is as follows:

  * **Accuracy:** 77.86%
  * **Confusion Matrix:**
    ```
    [[878 158]
     [154 219]]
    ```
  * **Classification Report:**
    ```
                  precision    recall  f1-score   support

               0       0.85      0.85      0.85      1036
               1       0.58      0.59      0.58       373

        accuracy                           0.78      1409
       macro avg       0.72      0.72      0.72      1409
    weighted avg       0.78      0.78      0.78      1409
    ```

-----

## How to Use

To run this project, you need to have Python and the necessary libraries installed.

### Installation

Clone the repository and install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Running the Code

1.  Place the `"WA_Fn-UseC_-Telco-Customer-Churn.csv"` dataset in the same directory as the notebook.
2.  Open and run the `Customer_Churn_Prediction_using_ML.ipynb` notebook in a Jupyter environment.

### Predictive System

The trained Random Forest model and the label encoders are saved in pickle files (`customer_churn_model.pkl` and `encoders.pkl`). You can load these files to make predictions on new customer data.

Here is an example of how to make a prediction:

```python
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Example new customer data
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

input_data_df = pd.DataFrame([input_data])

# Encode categorical features
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {pred_prob}")
```

-----

## Future Improvements

  * **Hyperparameter Tuning:** Implement techniques like GridSearchCV or RandomizedSearchCV to find the optimal hyperparameters for the models.
  * **Explore Other Models:** Evaluate other classification algorithms like Gradient Boosting, Support Vector Machines, or Neural Networks.
  * **Alternative Sampling Techniques:** Experiment with downsampling or other oversampling methods besides SMOTE.
  * **Address Overfitting:** Investigate and apply techniques to mitigate potential overfitting.
  * **Stratified K-Fold Cross-Validation:** Use stratified k-fold cross-validation to ensure the class distribution is preserved in each fold, which can be more robust for imbalanced datasets.
