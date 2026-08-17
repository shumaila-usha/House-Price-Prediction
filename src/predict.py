# ==========================================
# House Price Prediction
# Prediction Script
# ==========================================

import joblib
import pandas as pd


# ==========================================
# 1. Load Trained Model
# ==========================================

model = joblib.load("models/house_price_model.pkl")

print("House Price Prediction System")
print("--------------------------------")


# ==========================================
# 2. Load Dataset Columns
# ==========================================

data = pd.read_csv("data/train.csv")

# Get the exact feature columns used during training
feature_columns = data.drop(
    columns=["SalePrice", "Id"]
).columns


# ==========================================
# 3. Create Example House
# ==========================================

house = {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotFrontage": 65,
    "LotArea": 8450,
    "Street": "Pave",
    "Alley": None,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "CollgCr",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "2Story",
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "BrkFace",
    "MasVnrArea": 196,
    "ExterQual": "Gd",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "Gd",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ",
    "BsmtFinSF1": 706,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 150,
    "TotalBsmtSF": 856,
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 856,
    "2ndFlrSF": 854,
    "LowQualFinSF": 0,
    "GrLivArea": 1710,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "Gd",
    "TotRmsAbvGrd": 8,
    "Functional": "Typ",
    "Fireplaces": 0,
    "FireplaceQu": None,
    "GarageType": "Attchd",
    "GarageYrBlt": 2003,
    "GarageFinish": "RFn",
    "GarageCars": 2,
    "GarageArea": 548,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 0,
    "OpenPorchSF": 61,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": None,
    "Fence": None,
    "MiscFeature": None,
    "MiscVal": 0,
    "MoSold": 2,
    "YrSold": 2008,
    "SaleType": "WD",
    "SaleCondition": "Normal"
}


# ==========================================
# 4. Create DataFrame
# ==========================================

house_data = pd.DataFrame([house])

# Make sure columns are in exactly the same order
house_data = house_data.reindex(columns=feature_columns)


# ==========================================
# 5. Make Prediction
# ==========================================

predicted_price = model.predict(house_data)[0]


# ==========================================
# 6. Display Result
# ==========================================

print("\nPredicted House Price:")
print(f"${predicted_price:,.2f}")