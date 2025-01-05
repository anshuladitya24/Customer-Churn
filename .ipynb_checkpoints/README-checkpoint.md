# Customer Churn Prediction

## Project Overview
This project aims to predict customer churn for a telecom company using the Telco Customer Churn dataset from Kaggle. By leveraging machine learning techniques, the goal is to help the company identify factors contributing to churn and implement strategies for customer retention.

---

## Dataset

**Source**: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Key Features:
- **Customer Demographics**: Gender, SeniorCitizen, Partner, Dependents.
- **Account Information**: Tenure, Contract, Payment method, Monthly and Total charges.
- **Service Details**: Phone service, Internet service, Streaming services.
- **Target Variable**: `Churn` (Yes/No).

---

## Tools and Libraries Used
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Data Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`

---

## Workflow

### 1. Data Loading and Exploration
- Load the dataset into a Pandas DataFrame.
- Explore the data structure, identify missing values, and understand feature distributions.

### 2. Data Cleaning
- Drop irrelevant features (e.g., `customerID`).
- Handle missing or incorrect values in numeric columns.
- Convert categorical variables into numerical ones using one-hot encoding.

### 3. Feature Selection
- Define features (`X`) and the target variable (`y`).
- Split the dataset into training and testing subsets (80% train, 20% test).

### 4. Model Building
- Use Logistic Regression for binary classification.
- Train the model on the training dataset.
- Evaluate performance using accuracy, confusion matrix, and classification report.

### 5. Insights and Analysis
- Analyze feature importance to identify key drivers of customer churn.
- Visualize results using bar plots and heatmaps.

---

## Project Highlights

- **Accuracy**: The Logistic Regression model achieved an accuracy of approximately XX%.
- **Top Contributing Features**:
  1. Contract type (Month-to-Month contracts show higher churn rates).
  2. Monthly Charges (Higher charges increase churn likelihood).
  3. Tenure (Short-tenure customers are more prone to churn).

---

## Visualizations
### Examples:
- **Confusion Matrix**: Evaluates true positives, false positives, true negatives, and false negatives.
- **Feature Importance**: Highlights the top 10 features influencing customer churn.

---

## Future Scope
1. Explore advanced models like Random Forest or Gradient Boosting.
2. Perform hyperparameter tuning for optimized performance.
3. Create an interactive dashboard for business decision-making.

---

## How to Run

1. Clone the repository and navigate to the project folder:
    ```bash
    git clone <repository-url>
    cd customer-churn-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project directory.

4. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Customer_Churn_Prediction.ipynb
    ```

---

## Acknowledgments
- The dataset is provided by [Kaggle](https://www.kaggle.com/).
- Thanks to the creators of open-source libraries for enabling efficient data analysis and modeling.
