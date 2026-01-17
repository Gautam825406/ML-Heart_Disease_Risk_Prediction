# ❤️ Heart Disease Prediction System

## 📌 Overview
The **Heart Disease Prediction System** is an end-to-end Machine Learning project designed to predict the likelihood of heart disease using patient clinical data.  
This project demonstrates the complete machine learning lifecycle, including **data analysis, preprocessing, model training, model persistence, and real-time deployment using a Flask web application**.

The system aims to support **early diagnosis and preventive healthcare** by providing quick and reliable predictions based on key medical parameters.

---

## 🎯 Objective
- Predict the risk of heart disease using patient health indicators  
- Assist in early detection and preventive decision-making  
- Demonstrate real-world machine learning implementation and deployment  
- Build an interpretable and reproducible ML pipeline  

---

## 🧠 Model & Approach
- **Machine Learning Algorithm:** K-Nearest Neighbors (KNN)  
- **Feature Scaling:** StandardScaler  
- **Model Persistence:** Pickle (`.pkl`)  
- **Deployment Framework:** Flask  

The KNN model was selected due to its simplicity and effectiveness in classification tasks after proper feature scaling.

---

## 📂 Project Structure
HeartDiseasePrediction/
│
├── HeartdiseaseFinal.ipynb      # EDA, preprocessing, feature engineering & model training
├── app.py                       # Flask application for real-time prediction
├── knn_heart_model.pkl          # Trained KNN model
├── heart_scaler.pkl             # StandardScaler used during training
├── heart_columns.pkl            # Feature order reference for inference
├── requirements.txt             # Project dependencies




---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Deployment:** Flask  
- **Environment:** Jupyter Notebook  

---

## 📈 Key Highlights
- End-to-end machine learning pipeline  
- Clean data preprocessing and feature scaling  
- Real-time heart disease prediction using Flask  
- Reproducible and modular project structure  
- Healthcare-focused ML use case  

---

## 🧠 Inference
The analysis revealed that clinical features such as **age, chest pain type, exercise-induced angina, Oldpeak, ST_Slope, and maximum heart rate** play a significant role in predicting heart disease.  
Both statistical analysis and machine learning models showed strong alignment with medical knowledge, validating the reliability of the approach.

The trained KNN model demonstrated effective classification performance after scaling, making it suitable as a **decision-support tool** for early risk assessment rather than a standalone diagnostic system.

---

## 🔮 Future Improvements
- Implement advanced models such as **Random Forest** and **XGBoost**  
- Add **model explainability** using SHAP or LIME  
- Improve UI/UX of the web application  
- Deploy the application on cloud platforms (Render / AWS / Azure)  
- Integrate database support for patient history tracking  

---

## 👨‍💻 Author
**Gautam Keshri**  
B.Sc. (Hons) – Data Science & Artificial Intelligence  
Indian Institute of Technology (IIT), Guwahati  
