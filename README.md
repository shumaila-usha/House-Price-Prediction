# 🏠 House Price Prediction

A Machine Learning project that predicts house prices based on different features of residential properties.

## 📌 Project Overview

The **House Price Prediction** project uses Machine Learning to estimate the selling price of a house based on property-related features.

The project was developed as part of an **AI/ML learning project** and demonstrates the complete Machine Learning workflow, including:

* Data loading
* Data preprocessing
* Feature selection
* Model training
* Model evaluation
* Model saving
* House price prediction

---

## 🎯 Project Objective

The main objective of this project is to build a Machine Learning model that can predict house prices accurately using historical housing data.

The model learns relationships between house features and their corresponding sale prices and then uses those patterns to predict prices for new houses.

---

## 📊 Dataset

The project uses a housing dataset containing:

* **1460 rows**
* **81 columns**

The dataset contains different features describing residential properties, such as:

* Overall quality
* Living area
* Number of rooms
* Garage information
* Basement information
* Year built
* Neighborhood
* Property condition
* Sale price

The target variable is:

**SalePrice**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* VS Code

---

## 📂 Project Structure

```text
House_Price_Prediction/
│
├── data/
│   └── train.csv
│
├── models/
│   └── house_price_model.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── venv/
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Machine Learning Workflow

### 1. Load Dataset

The housing dataset is loaded using Pandas.

### 2. Data Preprocessing

The data is prepared for Machine Learning by handling the required features and preparing the dataset for model training.

### 3. Feature Selection

Important house-related features are selected as inputs for the model.

### 4. Train the Model

The Machine Learning model is trained using the prepared housing data.

### 5. Evaluate the Model

The trained model is evaluated using:

* Mean Absolute Error (MAE)
* R² Score

### 6. Save the Model

The trained model is saved as:

```text
models/house_price_model.pkl
```

### 7. Make Predictions

The saved model can be loaded and used to predict the price of a house.

---

## 📈 Model Performance

The trained model achieved the following results:

| Metric              |    Result |
| ------------------- | --------: |
| Mean Absolute Error | 17,386.11 |
| R² Score            |    0.8941 |

### 🏆 R² Score

An **R² score of 0.8941** means that the model explains approximately **89.41% of the variation** in the house prices within the evaluated data.

---

## 💰 Example Prediction

After training and saving the model, the prediction script produced:

```text
House Price Prediction System
--------------------------------

Predicted House Price:
$205,994.95
```

This demonstrates that the trained model can be loaded and used to make predictions for new property information.

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project

Open the project folder in **VS Code**.

### Step 3: Create a Virtual Environment

```bash
py -m venv venv
```

### Step 4: Activate the Virtual Environment

On Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Train the Model

```bash
python src/train.py
```

### Step 7: Make a Prediction

```bash
python src/predict.py
```

---

## 📦 Model File

The trained Machine Learning model is stored at:

```text
models/house_price_model.pkl
```

The model can be loaded using Joblib and used to make predictions on new house data.

---

## 🚀 Future Improvements

This project can be improved by:

* Adding a web interface using Streamlit
* Adding more advanced feature engineering
* Testing additional Machine Learning algorithms
* Hyperparameter tuning
* Adding data visualizations
* Deploying the prediction system online
* Allowing users to enter house details through a web form

---

## 👩‍💻 Author

**Shumaila Kiani**

AI/ML Project

---

## ⭐ Project Highlights

* ✅ Complete Machine Learning workflow
* ✅ Real-world housing dataset
* ✅ Model evaluation using MAE and R²
* ✅ Trained model saved with Joblib
* ✅ Prediction script implemented
* ✅ Example house price prediction generated

---

## 📜 License

This project is created for **educational and learning purposes**.
