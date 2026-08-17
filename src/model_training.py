# ==========================================
# House Price Prediction
# Model Training
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib
from pathlib import Path


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("data/train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]


# Remove ID because it is not useful for prediction
X = X.drop("Id", axis=1)


# ==========================================
# 3. Identify Column Types
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# ==========================================
# 4. Preprocessing
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 5. Create Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. Create Complete Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 7. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 8. Train Model
# ==========================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 9. Make Predictions
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# 10. Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))


# ==========================================
# 11. Save Model
# ==========================================

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

model_path = models_folder / "house_price_model.pkl"

joblib.dump(pipeline, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)