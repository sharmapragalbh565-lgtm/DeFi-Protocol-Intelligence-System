# DeFi-Protocol-Intelligence-System
Multi-Model Machine Learning for Yield Prediction, Risk Assessment & Strategy Recommendation

This project simulates a DeFi analytics engine using multiple machine learning models to evaluate protocol performance, risk exposure, anomalies, and optimal strategy selection.

It demonstrates how different ML paradigms (Regression, Classification, Tree Models, and Anomaly Detection) can work together to analyze decentralized finance metrics.

# Dataset Features

Synthetic dataset of 1000 samples with:

TVL – Total Value Locked

Historical_APY – Past yield

Volatility – Risk measure

Audit_Passed – Security audit indicator (0/1)

Liquidity_Score – Market liquidity quality

Chain_Score – Blockchain reliability metric

# System Architecture

Synthetic DeFi Data
        ↓
Feature Scaling (StandardScaler)
        ↓
Train/Test Split
        ↓
------------------------------------------
| Regression Model (APY Prediction)     |
| Risk Classification (Random Forest)   |
| Health Model (Logistic Regression)    |
| Strategy Model (Decision Tree)        |
| Anomaly Detection (Isolation Forest)  |
------------------------------------------
        ↓
Performance Evaluation + Sample Prediction

# Target Variables
1️. Future Yield (Regression)
y_regression = 0.8 * Historical_APY + 0.05 * Liquidity - 0.1 * Volatility

Predicts expected APY based on performance and risk indicators.

2️. Risk Classification

High Risk (1) if:

Volatility > 0.1

OR Audit not passed

Otherwise Low Risk (0).

3️. Protocol Health

Healthy (1) if:

APY > 0.15

AND Volatility < 0.08

4️. Strategy Recommendation
APY Range	Strategy
< 10%	Staking
10% – 20%	Lending
> 20%	Liquidity Providing (LP)

5️. Anomaly Injection

First 20 records are manipulated:

X_anomaly.loc[:19, "TVL"] = 500

This simulates abnormal liquidity collapse for anomaly detection testing.

# Models Used
1. Linear Regression

Used for continuous yield prediction.

Metric:

Mean Squared Error (MSE)

2. Random Forest Classifier

Used for risk detection.

Metrics:

Accuracy

Precision

Classification Report

3. Isolation Forest

Unsupervised anomaly detection.

Detects:

Abnormal TVL patterns

Outlier liquidity behavior

4. Logistic Regression

Binary classification for protocol health.

Metric:

Accuracy

Recall

5. Decision Tree Classifier

Multi-class strategy recommendation.

Metric:

Accuracy

F1 Score

# Model Performance Summary

Outputs include:

Yield Prediction MSE

Risk Classification Accuracy & Precision

Number of Anomalies Detected

Protocol Health Accuracy & Recall

Strategy Model Accuracy & F1 Score

# Installation
git clone https://github.com/yourusername/defi-intelligence-system.git
cd defi-intelligence-system
pip install -r requirements.txt
python main.py

# Requirements
numpy
pandas
scikit-learn

# Key Learnings Demonstrated

Multi-model architecture design

Feature engineering for DeFi analytics

Supervised vs Unsupervised ML comparison

Synthetic anomaly injection

Performance metric interpretation

Multi-target modeling from shared feature space

# Potential Improvements

Replace synthetic data with real DeFi protocol data (e.g., The Graph, Dune)

Add cross-validation

Add hyperparameter tuning (GridSearchCV)

Introduce XGBoost or LightGBM

Add SHAP explainability

Convert to real-time dashboard (Streamlit)

Deploy as API (FastAPI)

# Real-World Use Cases

DeFi Yield Optimization Engines
