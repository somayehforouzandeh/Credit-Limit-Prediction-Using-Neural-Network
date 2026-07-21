# 💳 Credit Limit Prediction Using Neural Network

<p align="center">
  <img src="Assets/Credit-Limit-Prediction-Banner.png" alt="Credit Limit Prediction Using Neural Network">
</p>

<h3 align="center">
  Predicting Credit Limits Using Data Preprocessing, Machine Learning, and Neural Networks
</h3>

---

## 📌 Project Overview

This project develops a neural network regression model to predict customer credit limits based on historical financial and customer-related data.

The project demonstrates a complete machine learning workflow, including data preprocessing, missing value handling, statistical outlier detection, categorical feature encoding, feature scaling, neural network modeling, and model evaluation.

---

## 🎯 Objectives

* Predict customer credit limits using machine learning
* Prepare and clean real-world customer data
* Handle missing values
* Detect and remove statistical outliers
* Transform categorical variables using One-Hot Encoding
* Normalize numerical features
* Build a neural network regression model
* Evaluate model performance using statistical metrics
* Visualize the training process and prediction results

---

## 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Outlier Detection
   ↓
One-Hot Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Neural Network Training
   ↓
Credit Limit Prediction
   ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

### Missing Values

Missing values in numerical features were replaced with the mean of their respective columns.

### Outlier Detection

Statistical outliers were identified using Z-scores. Observations with values beyond three standard deviations were filtered from the dataset.

### Categorical Features

Categorical variables were transformed into numerical features using **One-Hot Encoding**.

### Feature Scaling

Numerical features were standardized using **StandardScaler** to improve neural network training.

---

## 🧠 Neural Network Architecture

The prediction model was developed using **TensorFlow and Keras**.

```text
Input Layer
     ↓
Dense Layer — 64 Neurons — ReLU
     ↓
Dense Layer — 32 Neurons — ReLU
     ↓
Output Layer — 1 Neuron
```

### Model Configuration

* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)
* **Evaluation Metric:** Mean Absolute Error (MAE)
* **Epochs:** 30
* **Batch Size:** 32

---

## 📊 Model Evaluation

The model was evaluated using:

* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**

### Final Model Results

```text
MSE  = 19,814,772
RMSE = 4,451.38
```

RMSE represents the typical prediction error in the same unit as the target variable.

---

## 📈 Model Training Performance

<p align="center">
  <img src="Screenshots/Neural%20Network%20Training%20Performance.png" alt="Neural Network Training Performance" width="800">
</p>

The chart shows the training and validation loss during 30 epochs. The decreasing loss indicates that the model progressively improved its learning performance.


---

## 🎯 Actual vs. Predicted Credit Limit

<p align="center">
  <img src="Screenshots/Actual%20vs%20Predicted%20Credit%20Limit.png" alt="Actual vs Predicted Credit Limit" width="800">
</p>

This visualization compares the actual credit limit values with the values predicted by the neural network model.


---

## 📋 Final Model Evaluation Results

<p align="center">
  <img src="Screenshots/Model%20Evaluation%20Results.png" alt="Model Evaluation Results" width="700">
</p>

The final evaluation results summarize the prediction performance of the neural network model using MSE and RMSE metrics.


---

## 🛠️ Technologies & Libraries

* **Python**
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical computing
* **SciPy** — Statistical analysis and outlier detection
* **Scikit-learn** — Data preprocessing, train-test split, and evaluation
* **TensorFlow / Keras** — Neural network development
* **Matplotlib** — Data visualization

---

## 📂 Project Structure

```text
Credit-Limit-Prediction-Deep-Learning/
│
├── Assets/
│   └── Credit-Limit-Prediction-Banner.png
│
├── Data/
│   └── CreditPrediction.csv
│
├── Screenshots/
│   ├── Neural Network Training Performance.png
│   ├── Actual vs Predicted Credit Limit.png
│   └── Model Evaluation Results.png
│
├── credit_limit_prediction.py
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/somayehforouzandeh/Credit-Limit-Prediction-Using-Neural-Network.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

```bash
python credit_limit_prediction.py
```

---

## 💡 Key Takeaway

This project demonstrates how a complete machine learning pipeline can transform raw customer data into predictive insights using neural networks.

The workflow combines data preprocessing, statistical analysis, feature engineering, and deep learning to develop a regression model for credit limit prediction.

---

## 👩‍💻 Author

**Somayeh Forouzandeh**

Industrial Engineer | Business Intelligence | Data Analytics

---

⭐ Explore the code, visualizations, and machine learning workflow to understand how neural networks can be applied to credit limit prediction.
