# 📱 Students Social Media Addiction Analysis

## 🧠 Project Overview

In the modern digital era, social media has become an integral part of students' daily lives. While it offers connectivity and learning opportunities, excessive usage can negatively impact mental health, sleep, and academic performance.

This project aims to analyze the patterns of social media usage among students, predict the level of addiction, and evaluate which machine learning models perform best for this task.

---

## 🎯 Objectives

- Collect and preprocess student social media usage data.
- Perform exploratory data analysis (EDA) to uncover patterns and correlations.
- Build and compare multiple machine learning models to predict social media addiction scores.
- Provide a streamlined pipeline for inference through an API (`app.py` and `inference.py`).
- Visualize results to support data-driven recommendations.

---

## 📁 Repository Structure

```
Students-Social-Media-Addiction/
│
├── notebooks/                # Jupyter Notebooks for EDA, preprocessing, modeling
│   ├── BestFitModel.ipynb    # Gradient Boosting Regressor tuning
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── FeatureSelection.ipynb# Feature importance using Random Forest
│   ├── GradientBoostingRegressor.ipynb
│   ├── Inference_best_model.ipynb # Inference using best model
│   ├── LogisticRegression.ipynb
│   ├── LinearRegression.ipynb
│   ├── ModelTrainingWithoutFeatureSelection.ipynb
│   ├── PreProcessing.ipynb
│   ├── XGBoost.ipynb
│
├── sources/                  # Source code for deployment & inference
│   ├── app.py                # FastAPI app for model inference
│   └── inference.py          # Script for running predictions
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore

```

---

## 📊 Dataset

The dataset includes:

- **Demographics:** Age, gender, academic level.
- **Social Media Usage:** Average daily hours on different platforms.
- **Mental Health Metrics:** Stress, anxiety, sleep quality scores.
- **Addiction Score:** Target variable representing the degree of social media addiction.

### **Key Features:**

- **Student_ID**: Unique identifier for each student
- **Demographic Information**:
    - **`Age`** (18-24 years)
    - **`Gender`** (Male/Female)
    - **`Academic_Level`** (High School/Undergraduate/Graduate)
    - **`Country`** (Multiple countries represented)
- **Social Media Usage Patterns**:
    - **`Avg_Daily_Usage_Hours`** (1.5-8.5 hours)
    - **`Most_Used_Platform`** (12 platforms including Instagram, TikTok, Facebook, Twitter, YouTube, etc.)
- **Impact Assessment Metrics**:
    - **`Affects_Academic_Performance`** (Yes/No)
    - **`Sleep_Hours_Per_Night`** (3.8-9.6 hours)
    - **`Mental_Health_Score`** (4-9 scale, likely 0-10)
    - **`Relationship_Status`** (Single/In Relationship/Complicated)
    - **`Conflicts_Over_Social_Media`** (0-5 scale)
- **Target Variable**:
    - **`Addicted_Score`** (2-9 scale, where higher scores indicate stronger addiction)
    

## **📝**   Notebooks

### **1. EDA.ipynb**

- Performs data exploration and visualization.
- Identifies distributions, correlations, and trends between features and `Addicted_Score`.

### **2. PreProcessing.ipynb**

- Handles missing values, outliers, and categorical encoding.
- Scales numeric features for model training.

### **3. FeatureSelection(RandomForestRegressor).ipynb**

- Uses Random Forest to compute feature importance.
- Selects top features for model optimization.

### **4. LinearRegression.ipynb**

- Implements baseline Linear Regression.
- Evaluates model using MSE, RMSE, and R² metrics.

### **5. LogisticRegression.ipynb**

- Converts addiction scores into categories (low, medium, high).
- Evaluates classification performance using confusion matrices.

### **6. GradientBoostingRegressor.ipynb**

- Trains Gradient Boosting Regressor.
- Performs hyperparameter tuning for improved predictions.

### **7. BestFitModel(GradientBoostingRegressor).ipynb**

- Fine-tunes the best Gradient Boosting model.
- Prepares pipeline for deployment.

### **8. XGBoost.ipynb**

- Trains XGBoost Regressor and compares performance with other models.

### **9. ModelTrainingWithoutFeatureSelection.ipynb**

- Trains models using all features.
- Serves as baseline to evaluate feature selection impact.

### **10. Inference_best_model.ipynb**

- Demonstrates how to load the best-trained model and run predictions on new data.

## 🚀 API & Inference

### **1. FastAPI Application (`app.py`)**

- Serves the trained model via REST API.
- Accepts JSON input and returns predicted addiction scores.
- Provides interactive **Swagger UI** and **Redoc** documentation.

### **2. Running the API**

```bash
uvicorn sources.app:app --reload

```

- **Swagger UI:** `http://127.0.0.1:8000/docs`
    - Test API endpoints interactively.
    - Enter input features and receive predictions instantly.
- **Redoc:** `http://127.0.0.1:8000/redoc`
    - Alternative API documentation view.

### **3. Inference Script (`inference.py`)**

- Load the saved model pipeline.
- Run predictions on single or batch data without Jupyter.
- Useful for production or integrating into other applications.

## 🛠️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/AaditiG1/Students-Social-Media-Addiction.git
cd Students-Social-Media-Addiction

```

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

1. Install dependencies:

```bash
pip install -r requirements.txt

```

1. Launch Jupyter Notebook:

```bash
jupyter notebook

```

1. Run API (optional):

```bash
uvicorn sources.app:app --reload

```

## 🧪 Machine Learning Models

Several models were trained and compared:

| Model | MSE | RMSE | R² |
| --- | --- | --- | --- |
| Random Forest | 0.0388 | 0.1969 | 0.9846 |
| Gradient Boosting | 0.0480 | 0.2191 | 0.9808 |
| XGBoost | 0.0493 | 0.2219 | 0.9803 |

## 🏁 Conclusion

This project provides a comprehensive analysis and prediction framework for **students’ social media addiction**. Through careful data preprocessing, feature selection, and model training, we identified key patterns affecting addiction levels. The **Gradient Boosting Regressor** emerged as the best-performing model, achieving high accuracy and reliable predictions. Combined with a **FastAPI backend** and **Swagger UI**, this project enables real-time prediction of **Addicted_Score**, offering actionable insights for early interventions.

---

## 📚 Key Learnings

- Proper **data preprocessing and feature selection** significantly improve model performance.
- **Gradient Boosting Regressor** outperforms other models for this dataset.
- **Interactive APIs** with Swagger UI simplify testing and real-time predictions.
- **Visualizations** help in understanding feature relationships and model performance.

---

## 🔮 Future Plans

- Expand the dataset for broader representation and improved generalization.
- Explore **advanced ensemble and deep learning models** for potentially higher accuracy.
- Develop **dashboards** to monitor student addiction trends.
- Integrate the API with **web or mobile applications** for wider accessibility.


## 📬 Connect with Me

I’m always happy to collaborate, share knowledge, or discuss ideas! You can reach out through the following channels:

- **GitHub:** https://github.com/AaditiG1
- **LinkedIn:** [https://www.linkedin.com/in/aaditi-ghimire](https://www.linkedin.com/in/aaditi-ghimire-5697342bb/)
- **Email:** aaditighimire21@gmail.com
