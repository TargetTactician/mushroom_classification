## 🍄 Mushroom Classification Project

### 🔍 Overview

This project focuses on classifying mushrooms as **edible** or **poisonous** based on various physical attributes using supervised machine learning. It includes:

* Full **EDA and preprocessing pipeline**
* Comparison of **multiple classification models**
* A deployed **Streamlit web application** for real-time prediction

---

### 📁 Project Structure

```
├── Mushroom_done.ipynb         # Jupyter Notebook with complete ML workflow
├── app.py                      # Streamlit app for interactive predictions
├── model_with_encoders.pkl     # Trained model with label encoders
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies (to be added if deploying)
```

---

### 📊 Dataset

* Source: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
* Rows: \~8,000 samples
* Features: 22 categorical attributes
* Target: `edible` (0) or `poisonous` (1)

---

### 🧠 Models Used

The notebook (`Mushroom_done.ipynb`) includes training and evaluation of the following models:

| Model                  | Description                              |
| ---------------------- | ---------------------------------------- |
| Logistic Regression    | Baseline model for classification        |
| Decision Tree          | Tree-based classifier for feature splits |
| Random Forest          | Ensemble method of multiple trees        |
| K-Nearest Neighbors    | Instance-based classifier                |
| Naive Bayes            | Probabilistic classifier                 |
| Support Vector Machine | Hyperplane-based binary classifier       |

Each model is evaluated using:

* **Accuracy**
* **Precision, Recall, F1-score**
* **Confusion Matrix**

---

### ✅ Best Model

After comparing multiple models, the best-performing model was **Random Forest**, achieving the highest accuracy and robust performance across all metrics. This model is saved and used in the web app.

---

### 🌐 Web Application

A **Streamlit-based web app** is included for interactive predictions.

#### Features:

* Manual input using dropdowns
* CSV batch prediction
* Visual summary of predictions
* Downloadable result CSV

#### Run locally:

```bash
create python venv
activate venv (source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

---

### 📸 Screenshots

*<img width="1920" height="1003" alt="Screenshot from 2025-07-12 01-14-36" src="https://github.com/user-attachments/assets/ed6cfaf2-c15e-4933-951c-315c6bc44fce" />*

*<img width="1920" height="1003" alt="Screenshot from 2025-07-12 01-14-54" src="https://github.com/user-attachments/assets/38b3e9dd-5f50-4489-b25e-5f0a8a3db71e" />*

*<img width="1920" height="1003" alt="Screenshot from 2025-07-12 01-15-05" src="https://github.com/user-attachments/assets/e08ce398-cd9f-4280-9295-41d01229a701" />*

*<img width="1920" height="1003" alt="Screenshot from 2025-07-12 01-15-31" src="https://github.com/user-attachments/assets/d496f5b9-4916-42a0-a9f6-1ba9f6257ea2" />*
### Edible Summary
*<img width="1920" height="1003" alt="Screenshot from 2025-07-12 01-15-39" src="https://github.com/user-attachments/assets/957aed91-6767-46f2-895a-c133b232763c" />*



### ✍️ Author

* **Your Name:** Parthiban G
