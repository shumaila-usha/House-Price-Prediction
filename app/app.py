from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__, template_folder="../templates")

# Load trained model
model = joblib.load("models/house_price_model.pkl")
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Default values for the remaining model features
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

    # Replace default values with values entered by the user
    house["MSSubClass"] = float(request.form["MSSubClass"])
    house["MSZoning"] = request.form["MSZoning"]
    house["LotFrontage"] = float(request.form["LotFrontage"])
    house["LotArea"] = float(request.form["LotArea"])
    house["Street"] = request.form["Street"]
    house["LotShape"] = request.form["LotShape"]
    house["LandContour"] = request.form["LandContour"]
    house["Utilities"] = request.form["Utilities"]
    house["LotConfig"] = request.form["LotConfig"]
    house["LandSlope"] = request.form["LandSlope"]
    house["Neighborhood"] = request.form["Neighborhood"]
    house["Condition1"] = request.form["Condition1"]
    house["Condition2"] = request.form["Condition2"]
    house["BldgType"] = request.form["BldgType"]
    house["HouseStyle"] = request.form["HouseStyle"]
    house["OverallQual"] = float(request.form["OverallQual"])
    house["OverallCond"] = float(request.form["OverallCond"])
    house["YearBuilt"] = float(request.form["YearBuilt"])
    house["YearRemodAdd"] = float(request.form["YearRemodAdd"])
    house["GrLivArea"] = float(request.form["GrLivArea"])
    house["BedroomAbvGr"] = float(request.form["BedroomAbvGr"])
    house["FullBath"] = float(request.form["FullBath"])
    house["HalfBath"] = float(request.form["HalfBath"])
    house["GarageCars"] = float(request.form["GarageCars"])
    house["GarageArea"] = float(request.form["GarageArea"])

    # Convert to DataFrame
    house_df = pd.DataFrame([house])

    # Make prediction
    predicted_price = model.predict(house_df)[0]

    return render_template(
        "index.html",
        prediction=f"${predicted_price:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)