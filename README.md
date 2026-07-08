# 😴 SleepSense AI

> **AI-Powered Sleep Disorder Risk Assessment Platform**
https://sleepsense-ai.streamlit.app/

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Model-Random%20Forest-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Premium%20Dashboard-blueviolet?style=for-the-badge" />
</p>

---

## 📖 Overview

**SleepSense AI** is an intelligent clinical decision-support system that predicts **Normal Sleep**, **Insomnia**, and **Sleep Apnea** using Machine Learning.

The application combines healthcare analytics, an interactive dashboard, and a modern Streamlit interface to assist healthcare professionals in evaluating patient sleep health based on physiological and lifestyle information.

---


<p align="center">
<img src="images/dashboard.png" width="95%">
</p>

---

# 🚀 Features

### 🎨 Premium Dashboard
- Modern glassmorphism UI
- Dark & Light themes
- Responsive layout
- Interactive cards

### 🤖 AI Prediction
- Random Forest Classifier
- Real-time prediction
- Multi-class classification
- Confidence score

### 📊 Analytics Dashboard
- Interactive Plotly visualizations
- Correlation Heatmap
- Feature Importance
- Distribution Analysis
- Class Balance Visualization

### 📂 Batch Prediction
- Upload CSV
- Predict multiple patients
- Download predictions

### 📄 Reports
- PDF Report
- Excel Export
- JSON Export

---

# 🧠 Machine Learning Pipeline

```
Patient Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Random Forest Classifier
      │
      ▼
Prediction
      │
      ▼
Probability Score
      │
      ▼
Clinical Recommendation
```

---

# 📋 Input Features

| Category | Features |
|-----------|----------|
| 👤 Personal | Age, Gender, Occupation |
| ❤️ Health | BMI Category, Blood Pressure, Heart Rate |
| 🌙 Lifestyle | Sleep Duration, Daily Steps, Stress Level |
| 🏃 Activity | Physical Activity Level, Quality of Sleep |

---

# 🎯 Prediction Classes

| Class | Description |
|--------|-------------|
| 🟢 Normal | Healthy sleep pattern |
| 🟡 Insomnia | Difficulty initiating or maintaining sleep |
| 🔴 Sleep Apnea | Sleep-disordered breathing |

---

# ⚙️ Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Frontend | Streamlit |
| Styling | Custom CSS |
| Model | Random Forest Classifier |

---

# 📁 Project Structure

```text
SleepSense-AI/
│
├── app.py
├── model.pkl
├── encoder.pkl
├── requirements.txt
├── README.md
│
├── assets/
│   ├── logo.png
│   ├── background.jpg
│   └── icons/
│
├── images/
│   ├── dashboard.png
│   ├── prediction.png
│   └── analytics.png
│
├── notebooks/
│   └── Model_Training.ipynb
│
└── data/
    └── Sleep_health_dataset.csv
```

---

# 🚀 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Pavan-kalyan-k/Sleep-Disorder-Prediction.git

cd Sleep-Disorder-Prediction
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
streamlit run app.py
```

The application will open at:

```
http://localhost:8501
```

---

# 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **95%** |
| Precision | **92%** |
| Recall | **93%** |
| F1 Score | **92%** |
| ROC-AUC | **98%** |
| Cross Validation | **90%** |

---

# 🔮 Future Improvements

- SHAP Explainability
- User Authentication
- Cloud Deployment
- REST API
- Mobile-Friendly UI
- Doctor Dashboard
- Real-time Wearable Integration
- AI Chat Assistant

---

# 👨‍💻 Developer

## Pavan Kalyan


🌐 GitHub:
https://github.com/Pavan-kalyan-k

💼 LinkedIn:
https://www.linkedin.com/in/pavankalyan16/

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.
