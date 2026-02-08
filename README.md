# Data Generation using Modelling and Simulation for Machine Learning

## 1. Objective
To generate simulation based data using Cantera, evaluate multiple machine learning models, and compare them using TOPSIS.

---

## 2. Description
This project performs ammonia synthesis simulations by varying temperature and pressure parameters. The generated synthetic dataset is used to train and evaluate multiple models. Performance metrics from these models are then ranked using TOPSIS to identify the most suitable model.

---

## 3. Methodology

### 3.1 Data Generation
- Cantera is used to compute thermodynamic equilibrium for ammonia synthesis.
- Two input parameters are varied:
  - Temperature: 600–900 K
  - Pressure: 50–300 atm
- Feed composition is fixed at N₂:H₂ = 1:3.
- For each parameter combination, the NH₃ mole fraction is recorded.
- A total of 1000 simulations are executed.
- Output dataset is stored as `cantera_data.csv`.

---

### 3.2 Machine Learning Models
The following regression models are used:

- Linear Regression  
- Ridge Regression  
- K-Nearest Neighbors   
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Support Vector Regressor  
- Baseline Neural Network

---

### 3.3 Evaluation Metrics
Each model is evaluated using:
- RMSE
- MAE
- R² Score
- Training Time

The evaluation results are stored in `cantera.csv`.

---

### 3.4 TOPSIS Ranking
TOPSIS is applied to rank the models.

**Weights used:**
- RMSE: 0.35
- MAE: 0.30 
- R²: 0.20 
- Time: 0.15 

The TOPSIS score and rank are appended to the output file.


## 4. Output
The output CSV file contains:
- Original input data
- TOPSIS Score column
- Rank column

---

## 5. Result 
The table compares the various models using RMSE, MAE, R² score and time spent. Where lower error and higher R² scores show better performance, whereas time spent reflects model efficiency. Non-linear models like KNN and ensemble based ones show a generally better result when compared with linear models. The final TOPSIS ranking reflects the balanced tradeoff between accuracy and computation efficiency
