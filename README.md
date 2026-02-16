Heart Disease Prediction using Machine Learning

📋 Overview
Heart disease is one of the leading causes of death worldwide. This project uses Machine Learning to predict the presence of heart disease in patients based on clinical parameters, achieving 95% accuracy with Random Forest algorithm.

"Early detection through data-driven systems can save lives by enabling timely intervention and prevention."

🎯 Objective
Build a robust classification model that predicts heart disease presence to:

Enable early intervention for at-risk patients
Support data-driven clinical decisions
Reduce healthcare costs through preventive care

📊 Dataset
Source: UCI Heart Disease Dataset

Dataset Stats:
Total Patients: 303
With Disease: 165 (54.5%)
Without Disease: 138 (45.5%)
Features: 13 medical attributes

🛠️ Tech Stack
Category Libraries
Data Processing	---> pandas, numpy
Visualization ---> matplotlib, seaborn
Machine Learning ---> scikit-learn, xgboost
Deep Learning ---> keras, tensorflow

🤖 Models Implemented
Algorithm	Accuracy
Random Forest	95% 🏆
XGBoost	92%
SVM	91%
Logistic Regression	89%
Neural Network	88%
KNN	87%
Decision Tree	85%
Naive Bayes	83%

🏆 Winner: Random Forest with 95% Accuracy!

🔍 Key Findings
Top 5 Predictive Features:
thalach - Maximum heart rate achieved
cp - Chest pain type
oldpeak - ST depression
ca - Number of major vessels
thal - Thalassemia

Interesting Insights:
Patients with typical angina show higher disease probability
Higher heart rate during exercise correlates with lower risk
ST depression is a strong indicator of heart disease

🚀 Quick Start
1. Clone Repository
bash
git clone https://github.com/Jeevithagowda18/-Heart-Disease-Prediction-using-Machine-Learning
cd Heart-Disease-Prediction
2. Install Dependencies
bash
pip install -r requirements.txt
3. Run Jupyter Notebook
bash
jupyter notebook Heart_Disease_Prediction.ipynb
💻 Usage Example
python
import pandas as pd
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')

# Sample patient data
patient = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]

# Predict
prediction = model.predict(patient)
probability = model.predict_proba(patient)

if prediction[0] == 1:
    print(f"⚠️ High risk ({probability[0][1]:.2%} probability)")
else:
    print(f"✅ Low risk ({probability[0][0]:.2%} probability)")

📈 Results Visualization
The project includes:

Confusion Matrices for all models
Feature Importance charts
ROC Curves for comparison
Learning Curves for bias-variance analysis

🎯 Conclusion
✅ Random Forest is the best model with 95% accuracy
✅ Ensemble methods outperform individual classifiers
✅ Model can assist doctors in early diagnosis
✅ Ready for clinical decision support deployment