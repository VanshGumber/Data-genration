import cantera as ct
import numpy as np
import pandas as pd
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


np.random.seed(42)

gas = ct.Solution("gri30.yaml")

rows = []

for itera in range(1000):
    print(itera)
    t = np.random.uniform(600, 900)
    p = np.random.uniform(50, 300)

    gas.TP = t, p * ct.one_atm
    gas.X = "N2:0.25, H2:0.75"
    gas.equilibrate("TP")

    nh3 = gas["NH3"].X[0]

    rows.append([t, p, nh3])

df = pd.DataFrame(
    rows,
    columns=["Temp", "Pressure", "Result"]
)

x = df[["Temp", "Pressure"]].values
y = df["Result"].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=10))
])

svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10))
])

nn = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPRegressor(
        hidden_layer_sizes=(16,),
        activation="relu",
        max_iter=500,
        random_state=42
    ))
])

models = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(alpha=1.0),
    "KNN": knn,
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "SVR_RBF": svr,
    "NeuralNet": nn
}

results = []

for name, model in models.items():
    print(name)
    start = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start

    preds = model.predict(x_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "TrainTime_sec": train_time
    })

results_df = pd.DataFrame(results).sort_values("RMSE")

df.to_csv("cantera_data.csv", index=False)
results_df.to_csv("cantera.csv", index=False)


inp = "cantera.csv"
w = "0.35, 0.25, 0.2, 0.15"
imp = "-,-,+,-"
out = "cantera.csv"

try:
    df = pd.read_csv(inp)
except FileNotFoundError:
    print("Input file not found")
    sys.exit(1)

if df.shape[1] < 3:
    print("Input file must contain three or more columns")
    sys.exit(1)

data = df.iloc[:, 1:]

try:
    data = data.astype(float)
except:
    print("From 2nd to last columns must contain numeric values only")
    sys.exit(1)

weights = w.split(",")
impacts = imp.split(",")

if len(weights) != data.shape[1] or len(impacts) != data.shape[1]:
    print("Number of weights, impacts and criteria columns must be same")
    sys.exit(1)

try:
    weights = [float(i) for i in weights]
except:
    print("Weights must be numeric and comma separated")
    sys.exit(1)

for i in impacts:
    if i not in ["+", "-"]:
        print("Impacts must be either + or -")
        sys.exit(1)

norm = data.copy()
for i in range(data.shape[1]):
    d = np.sqrt(sum(data.iloc[:, i] ** 2))
    for j in range(data.shape[0]):
        norm.iat[j, i] = data.iat[j, i] / d

for i in range(norm.shape[1]):
    for j in range(norm.shape[0]):
        norm.iat[j, i] = norm.iat[j, i] * weights[i]

ideal_best = []
ideal_worst = []

for i in range(norm.shape[1]):
    if impacts[i] == "+":
        ideal_best.append(norm.iloc[:, i].max())
        ideal_worst.append(norm.iloc[:, i].min())
    else:
        ideal_best.append(norm.iloc[:, i].min())
        ideal_worst.append(norm.iloc[:, i].max())

s_plus = []
s_minus = []

for i in range(norm.shape[0]):
    s1 = 0
    s2 = 0
    for j in range(norm.shape[1]):
        s1 += (norm.iat[i, j] - ideal_best[j]) ** 2
        s2 += (norm.iat[i, j] - ideal_worst[j]) ** 2
    s_plus.append(np.sqrt(s1))
    s_minus.append(np.sqrt(s2))

score = []
for i in range(len(s_plus)):
    score.append(s_minus[i] / (s_plus[i] + s_minus[i]))

df["Topsis Score"] = score
df["Rank"] = df["Topsis Score"].rank(ascending=False, method="max").astype(int)

df.to_csv(out, index=False)
print("TOPSIS result saved to", out)
