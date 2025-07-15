# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# โหลดข้อมูลจาก path บนเครื่องของคุณ
file_path = r"C:\Users\26009648\Desktop\NNI Line\PYTHON_NNI.xlsx"
df = pd.read_excel(file_path)

# นิยาม features และ target
X = df[["LC", "MFR_S205", "MFR_S206", "MFR_S402C"]]
y = df["NNI"]

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ทำ normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้าง Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# กำหนดค่าที่ต้องการ tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

# ใช้ Grid Search เพื่อหาค่าที่ดีที่สุด
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# ใช้โมเดลที่ดีที่สุดในการทำนาย
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# ประเมินโมเดล
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# รวมผลลัพธ์ลงใน DataFrame
results = X_test.copy()
results["Actual_NNI"] = y_test.values
results["Predicted_NNI"] = y_pred

# Export เป็น CSV
output_path = r"C:\Users\26009648\Desktop\NNI Line\NNI_Prediction_Results.csv"
results.to_csv(output_path, index=False)

# แสดงผลใน console
print("Best Parameters:", grid_search.best_params_)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
print(f"\n✅ Results exported to: {output_path}")




# %%
import joblib

# บันทึกโมเดลและตัวแปลง scaler ลงเป็น .pkl
joblib.dump(best_model, r"C:\Users\26009648\Desktop\NNI Line\best_model.pkl")
joblib.dump(scaler, r"C:\Users\26009648\Desktop\NNI Line\scaler.pkl")

print("✅ โมเดลและ scaler ถูกบันทึกเรียบร้อยแล้ว")

# %%
import streamlit as st
import numpy as np
import joblib

# โหลดโมเดล
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("NNI Predictor App")

a = st.number_input("Input LC")
b = st.number_input("Input MFR_S205")
c = st.number_input("Input MFR_S206")
d = st.number_input("Input MFR_S402C")

if st.button("Predict"):
    X = np.array([[a, b, c, d]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    st.success(f"Predicted E: {prediction[0]:.2f}")



