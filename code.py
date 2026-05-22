
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

n_samples = 1000
TVL = np.linspace(100000, 500000, n_samples)
Historical_APY = np.linspace(0.02, 0.3, n_samples)
Volatility = np.linspace(0.01, 0.15, n_samples)
Audit_Passed = np.tile([0, 1], n_samples // 2)
Liquidity_Score = np.linspace(0, 1, n_samples)
Chain_Score = np.linspace(0.5, 1, n_samples)

X = pd.DataFrame({
    "TVL": TVL,
    "Historical_APY": Historical_APY,
    "Volatility": Volatility,
    "Audit_Passed": Audit_Passed,
    "Liquidity_Score": Liquidity_Score,
    "Chain_Score": Chain_Score
})


y_regression = 0.8 * X["Historical_APY"] + 0.05 * X["Liquidity_Score"] - 0.1 * X["Volatility"]
y_classification = ((X["Volatility"] > 0.1) | (X["Audit_Passed"] == 0)).astype(int)
y_health = ((X["Historical_APY"] > 0.15) & (X["Volatility"] < 0.08)).astype(int)


conditions = [
    (X["Historical_APY"] < 0.1),
    (X["Historical_APY"] >= 0.1) & (X["Historical_APY"] < 0.2),
    (X["Historical_APY"] >= 0.2)
]
choices = [0, 1, 2]
y_strategy = np.select(conditions, choices)


X_anomaly = X.copy()
X_anomaly.loc[:19, "TVL"] = 500  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_scaled, y_regression, test_size=0.2)
X_train2, X_test2, y_train_cls, y_test_cls = train_test_split(X_scaled, y_classification, test_size=0.2)
X_train4, X_test4, y_train_health, y_test_health = train_test_split(X_scaled, y_health, test_size=0.2)
X_train5, X_test5, y_train_strat, y_test_strat = train_test_split(X_scaled, y_strategy, test_size=0.2)


reg_model = LinearRegression()
reg_model.fit(X_train, y_train_reg)
reg_mse = mean_squared_error(y_test_reg, reg_model.predict(X_test))


cls_model = RandomForestClassifier(n_estimators=100)
cls_model.fit(X_train2, y_train_cls)
cls_acc = accuracy_score(y_test_cls, cls_model.predict(X_test2))
cls_report = classification_report(y_test_cls, cls_model.predict(X_test2), output_dict=True)


anomaly_model = IsolationForest(contamination=0.02)
anomaly_model.fit(X_scaled)
anomaly_preds = anomaly_model.predict(scaler.transform(X_anomaly))
anomaly_result = pd.Series(anomaly_preds).value_counts()

health_model = LogisticRegression()
health_model.fit(X_train4, y_train_health)
health_acc = accuracy_score(y_test_health, health_model.predict(X_test4))
health_report = classification_report(y_test_health, health_model.predict(X_test4), output_dict=True)


strategy_model = DecisionTreeClassifier()
strategy_model.fit(X_train5, y_train_strat)
strategy_acc = accuracy_score(y_test_strat, strategy_model.predict(X_test5))
strategy_report = classification_report(y_test_strat, strategy_model.predict(X_test5), output_dict=True)


sample_input = scaler.transform([X.iloc[0].values])

print("=== SAMPLE INPUT PREDICTIONS ===")
print("Predicted Future APY:", round(reg_model.predict(sample_input)[0], 5))
print("Risk Classification (0=Low, 1=High):", cls_model.predict(sample_input)[0])
print("Anomaly Detection (1=Normal, -1=Anomaly):", anomaly_model.predict(sample_input)[0])
print("Protocol Health (0=Unhealthy, 1=Healthy):", health_model.predict(sample_input)[0])
print("Strategy Recommendation (0=Staking, 1=Lending, 2=LP):", strategy_model.predict(sample_input)[0])


print("\n=== MODEL PERFORMANCE SUMMARY ===")
print("Yield Prediction MSE:", reg_mse)
print("Risk Classification Accuracy:", cls_acc, "| Precision:", cls_report['1']['precision'])
print("Anomalies Detected (Out of 1000):", anomaly_result.to_dict())
print("Protocol Health Accuracy:", health_acc, "| Recall:", health_report['1']['recall'])
print("Strategy Recommendation Accuracy:", strategy_acc, "| F1 Score:", strategy_report['1']['f1-score'])
