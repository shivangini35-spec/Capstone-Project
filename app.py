import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

APP_VERSION = "2026-08-30-dtype-fix-final"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Property Price Prediction")
st.caption(
    "Streamlit deployment of the capstone notebook using the final "
    "Tuned Gradient Boosting model."
)

st.caption(f"App version: {APP_VERSION}")


# =========================================================
# FILES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Propertydata.csv"
MODEL_PATH = BASE_DIR / "tuned_gradient_boosting_model.pkl"


# =========================================================
# COLUMN RENAME MAP — SAME AS NOTEBOOK
# =========================================================

RENAME_MAP = {
    "PropertyClass": "Property Class",
    "PropertyZone": "Property Zone",
    "PropertyFrontage": "Lot Frontage",
    "PropertySize": "Property Size",
    "PropertyShape": "Property[Lot] Shape",
    "Elevation": "Land Contour[Elevation]",
    "Orientation": "Lot Configuration",
    "Grade": "Land Slope",
    "Condition1": "Condition 1",
    "Condition2": "Condition 2",
    "BldgType": "Building Type",
    "PropertyStyle": "Property[House] Style",
    "OverallQual": "Overall Quality",
    "OverallCond": "Overall Condition",
    "YearBuilt": "Year Built",
    "YearRemodAdd": "Year Remodeled/Addition",
    "RoofStyle": "Roof Style",
    "RoofMatl": "Roof Material",
    "Roof1Material": "Roof Material 1",
    "Roof2Material": "Roof Material 2",
    "ExterQual": "Exterior Quality",
    "ExterCond": "Exterior Condition",
    "PropertyFooting": "Property Footing [Foundation] Type",
    "BsmntFinish": "Basement Finish Rating",
    "BsmntMaintenance": "Basement Maintenance",
    "BsmntVisibility": "Basement Visibility",
    "BsmntFinRat1": "Basement Finish Rating 1",
    "BsmntFinSty1": "Basement Finish Style 1",
    "BsmntFinQual1": "Basement Finish Quality 1",
    "BsmtFinSF2": "Basement Finish Square Footage 2",
    "BsmtUnfSF": "Basement Unfinished Square Footage",
    "BsmntSqFtage": "Basement Square Footage",
    "Heating": "Heating Type",
    "HeatingEfficiency": "Heating Quality and Condition",
    "CentralAir": "Central Air Conditioning",
    "Electrical": "Electrical System",
    "1stFlrSF": "First Floor Square Footage",
    "2ndFlrSF": "Second Floor Square Footage",
    "LowQualFinSF": "Low Quality Finished Square Footage",
    "GrLivArea": "Above Ground Living Area",
    "BsmtFullBath": "Basement Full Bathrooms",
    "BsmtHalfBath": "Basement Half Bathrooms",
    "Bath1": "Bathroom 1",
    "Bath2": "Bathroom 2",
    "BedroomUpLev": "Bedrooms Upstairs",
    "KitchenUpLev": "Kitchens Upstairs",
    "KitchenQual": "Kitchen Quality",
    "CntRmsUpLev": "Total Rooms Above Ground",
    "CntFireplaces": "Number of Fireplaces",
    "QualFireplace": "Fireplace Quality",
    "BasementType": "Basement Type",
    "BasementYrBlt": "Basement Year Built",
    "BasementFinish": "Basement Finish",
    "BasementCars": "Basement Cars",
    "BasementSqFootage": "Basement Area",
    "BasementQual": "Basement Quality",
    "BasementCond": "Basement Condition",
    "PavedDrive": "Paved Driveway",
    "WoodDeckSF": "Wood Deck Square Footage",
    "OpenPorchSF": "Open Porch Square Footage",
    "EnclosedPorch": "Enclosed Porch",
    "3SsnPorch": "Three Season Porch",
    "ScreenPorch": "Screen Porch",
    "PoolArea": "Pool Area",
    "PoolQC": "Pool Quality",
    "BoundaryFeatures": "Boundary Features",
    "AddFeatures": "Additional Features",
    "AddVal": "Additional Value",
    "SaleMon": "Sale Month",
    "SaleYr": "Sale Year",
    "SaleType": "Sale Type",
    "SaleCondn": "Sale Condition",
    "PropPrice": "Property Price",
}


ORDINAL_COLS = [
    "Property[Lot] Shape",
    "Land Slope",
    "Exterior Quality",
    "Exterior Condition",
    "Basement Finish Rating",
    "Basement Maintenance",
    "Basement Visibility",
    "Basement Finish Rating 1",
    "Basement Finish Quality 1",
    "Heating Quality and Condition",
    "Kitchen Quality",
    "Functional",
    "Fireplace Quality",
    "Basement Finish",
    "Basement Quality",
    "Basement Condition",
    "Paved Driveway",
    "Pool Quality",
]

QUALITY_ORDER = {
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
}

CATEGORICAL_MISSING_COLUMNS = [
    "Pool Quality",
    "Additional Features",
    "Alley",
    "Boundary Features",
    "ExteriorCladdingType",
    "Fireplace Quality",
    "Basement Visibility",
    "Basement Finish",
    "Basement Maintenance",
    "Basement Finish Rating 1",
    "Basement Finish Quality 1",
    "Basement Type",
    "Basement Quality",
    "Basement Condition",
    "Basement Finish Rating",
]

NUMERICAL_ZERO_COLUMNS = [
    "ExteriorCladdingArea",
    "Basement Year Built",
]


# =========================================================
# NOTEBOOK PREPROCESSING
# =========================================================

def clean_and_rename(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning/renaming steps used in the notebook."""
    df = raw_df.copy()

    df.drop(columns=["PropertyID"], errors="ignore", inplace=True)
    df.rename(columns=RENAME_MAP, inplace=True)

    if "Property Class" in df.columns:
        df["Property Class"] = df["Property Class"].astype("category")

    if "Sale Year" in df.columns:
        df["Sale Year"] = pd.to_numeric(
            df["Sale Year"], errors="coerce"
        )

    for col in CATEGORICAL_MISSING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in NUMERICAL_ZERO_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0)

    if "Lot Frontage" in df.columns:
        df["Lot Frontage"] = pd.to_numeric(
            df["Lot Frontage"], errors="coerce"
        )
        df["Lot Frontage"] = df["Lot Frontage"].fillna(
            df["Lot Frontage"].median()
        )

    if "Electrical System" in df.columns:
        mode = df["Electrical System"].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "None"
        df["Electrical System"] = (
            df["Electrical System"].fillna(fill_value)
        )

    for col in [
        "Basement Year Built",
        "Lot Frontage",
        "ExteriorCladdingArea",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0)

    return df


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering copied from the capstone notebook."""
    out = frame.copy()

    out["HouseAge"] = (
        out["Sale Year"] - out["Year Built"]
    ).clip(lower=0)

    out["RemodAge"] = (
        out["Sale Year"] - out["Year Remodeled/Addition"]
    ).clip(lower=0)

    out["IsRemodeled"] = (
        out["Year Remodeled/Addition"] > out["Year Built"]
    ).astype(int)

    out["TotalSF"] = (
        out["Basement Square Footage"]
        + out["First Floor Square Footage"]
        + out["Second Floor Square Footage"]
    )

    out["TotalPorchSF"] = (
        out["Wood Deck Square Footage"]
        + out["Open Porch Square Footage"]
        + out["Enclosed Porch"]
        + out["Three Season Porch"]
        + out["Screen Porch"]
    )

    out["TotalBathrooms"] = (
        out["Bathroom 1"]
        + 0.5 * out["Bathroom 2"]
        + out["Basement Full Bathrooms"]
        + 0.5 * out["Basement Half Bathrooms"]
    )

    out["LivingAreaPerRoom"] = (
        out["Above Ground Living Area"]
        / out["Total Rooms Above Ground"].replace(0, np.nan)
    )

    out["LivingAreaPerRoom"] = (
        out["LivingAreaPerRoom"]
        .fillna(out["Above Ground Living Area"])
    )

    out["HasBasement"] = (
        out["Basement Square Footage"] > 0
    ).astype(int)

    out["HasGarage"] = (
        out["Basement Cars"] > 0
    ).astype(int)

    out["HasFireplace"] = (
        out["Number of Fireplaces"] > 0
    ).astype(int)

    out["HasPool"] = (
        out["Pool Area"] > 0
    ).astype(int)

    out["Has2ndFloor"] = (
        out["Second Floor Square Footage"] > 0
    ).astype(int)

    out["HasPorch"] = (
        out["TotalPorchSF"] > 0
    ).astype(int)

    out["HasMasonry"] = (
        out["ExteriorCladdingArea"].fillna(0) > 0
    ).astype(int)

    out["QualLivArea"] = (
        out["Overall Quality"]
        * out["Above Ground Living Area"]
    )

    out["QualCond"] = (
        out["Overall Quality"]
        * out["Overall Condition"]
    )

    out["GarageScore"] = (
        out["Basement Cars"]
        * out["Basement Area"]
    )

    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    sale_month_num = (
        out["Sale Month"]
        .astype(str)
        .str.strip()
        .str[:3]
        .map(month_map)
    )

    sale_month_num = sale_month_num.fillna(
        pd.to_numeric(
            out["Sale Month"], errors="coerce"
        )
    )

    out["SaleMonSin"] = np.sin(
        2 * np.pi * sale_month_num / 12
    )
    out["SaleMonCos"] = np.cos(
        2 * np.pi * sale_month_num / 12
    )

    return out


def fit_encoding_metadata(engineered_train: pd.DataFrame):
    """
    Reproduce the notebook's ordinal + one-hot encoding and return
    metadata needed to transform new rows identically.
    """
    categorical_cols = engineered_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if "Property Price" in categorical_cols:
        categorical_cols.remove("Property Price")

    ordinal_cols = [
        col for col in ORDINAL_COLS
        if col in categorical_cols
    ]
    nominal_cols = [
        col for col in categorical_cols
        if col not in ordinal_cols
    ]

    encoded = engineered_train.copy()
    ordinal_meta = {}

    for col in ordinal_cols:
        values = set(
            engineered_train[col]
            .dropna()
            .astype(str)
            .unique()
        )

        if values.issubset(set(QUALITY_ORDER.keys())):
            encoded[col] = (
                engineered_train[col]
                .map(QUALITY_ORDER)
                .fillna(0)
                .astype(int)
            )
            ordinal_meta[col] = {
                "type": "quality",
                "mapping": QUALITY_ORDER,
            }
        else:
            categories = (
                engineered_train[col]
                .astype("category")
                .cat.categories
                .tolist()
            )
            mapping = {
                value: idx
                for idx, value in enumerate(categories)
            }
            encoded[col] = (
                engineered_train[col]
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )
            ordinal_meta[col] = {
                "type": "category_codes",
                "mapping": mapping,
            }

    encoded = pd.get_dummies(
        encoded,
        columns=nominal_cols,
        drop_first=True,
        dtype=int,
    )

    target = "Property Price"
    feature_columns = [
        c for c in encoded.columns if c != target
    ]

    return (
        encoded,
        ordinal_cols,
        nominal_cols,
        ordinal_meta,
        feature_columns,
    )


def transform_new_rows(
    rows: pd.DataFrame,
    ordinal_cols,
    nominal_cols,
    ordinal_meta,
    feature_columns,
) -> pd.DataFrame:
    """Transform new cleaned+engineered rows to the model's 238 columns."""
    encoded = rows.copy()

    for col in ordinal_cols:
        meta = ordinal_meta[col]
        mapping = meta["mapping"]

        if col not in encoded.columns:
            encoded[col] = 0
            continue

        if meta["type"] == "quality":
            encoded[col] = (
                encoded[col]
                .map(mapping)
                .fillna(0)
                .astype(int)
            )
        else:
            encoded[col] = (
                encoded[col]
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

    available_nominal = [
        c for c in nominal_cols if c in encoded.columns
    ]

    encoded = pd.get_dummies(
        encoded,
        columns=available_nominal,
        drop_first=True,
        dtype=int,
    )

    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    # GradientBoosting requires numeric input only.
    encoded = encoded.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    return encoded


# =========================================================
# HELPERS FOR STREAMLIT INPUTS
# =========================================================

def default_value(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        value = pd.to_numeric(
            series, errors="coerce"
        ).median()
        return float(value) if pd.notna(value) else 0.0

    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]

    return "None"


def numeric_bounds(series: pd.Series, fallback_min=0.0):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return fallback_min, fallback_min + 1.0

    low = float(numeric.min())
    high = float(numeric.max())

    if low == high:
        high = low + 1.0

    return low, high


def options_for(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return []

    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def build_default_row(clean_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = clean_df.drop(
        columns=["Property Price"],
        errors="ignore",
    )

    defaults = {
        col: default_value(feature_df[col])
        for col in feature_df.columns
    }

    return pd.DataFrame([defaults])


# =========================================================
# LOAD DATA + MODEL
# =========================================================

@st.cache_resource(show_spinner="Preparing the prediction model...")
def load_artifacts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Propertydata.csv was not found. Add Propertydata.csv "
            "to the same GitHub folder as app.py."
        )

    raw = pd.read_csv(DATA_PATH)
    clean = clean_and_rename(raw)
    engineered = engineer_features(clean)

    (
        encoded,
        ordinal_cols,
        nominal_cols,
        ordinal_meta,
        feature_columns,
    ) = fit_encoding_metadata(engineered)

    X = encoded.drop(columns=["Property Price"])
    y = encoded["Property Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

        if getattr(model, "n_features_in_", len(feature_columns)) != len(feature_columns):
            raise ValueError(
                "The saved model does not match the current training data "
                f"({getattr(model, 'n_features_in_', '?')} model features vs "
                f"{len(feature_columns)} encoded features)."
            )
    else:
        # Final tuned parameters from the notebook.
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=4,
            subsample=0.9,
            random_state=42,
        )
        model.fit(X_train, np.log1p(y_train))

    # Holdout metrics, calculated with the loaded/fallback model.
    test_pred = np.expm1(model.predict(X_test))
    metrics = {
        "r2": r2_score(y_test, test_pred),
        "mae": mean_absolute_error(y_test, test_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
    }

    return {
        "model": model,
        "clean_df": clean,
        "ordinal_cols": ordinal_cols,
        "nominal_cols": nominal_cols,
        "ordinal_meta": ordinal_meta,
        "feature_columns": feature_columns,
        "metrics": metrics,
    }


try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(f"Could not prepare the app: {exc}")
    st.stop()


model = artifacts["model"]
clean_df = artifacts["clean_df"]
ordinal_cols = artifacts["ordinal_cols"]
nominal_cols = artifacts["nominal_cols"]
ordinal_meta = artifacts["ordinal_meta"]
feature_columns = artifacts["feature_columns"]
metrics = artifacts["metrics"]


# =========================================================
# SIDEBAR — MODEL INFORMATION
# =========================================================

with st.sidebar:
    st.header("Model Information")
    st.write("**Final model:** Tuned Gradient Boosting")
    st.metric("Holdout R²", f"{metrics['r2'] * 100:.2f}%")
    st.metric("Holdout MAE", f"${metrics['mae']:,.0f}")
    st.metric("Holdout RMSE", f"${metrics['rmse']:,.0f}")
    st.caption(
        f"The model uses {len(feature_columns)} encoded/engineered features. "
        "The target is trained on log1p(Property Price), and predictions are "
        "converted back to USD using expm1()."
    )


# =========================================================
# TABS
# =========================================================

predict_tab, batch_tab, about_tab = st.tabs(
    ["🔮 Predict a Property", "📄 Batch Prediction", "ℹ️ About"]
)


# =========================================================
# SINGLE PROPERTY PREDICTION
# =========================================================

with predict_tab:
    st.subheader("Enter Property Details")

    st.info(
        "For a simple capstone demo, the app asks for the most meaningful "
        "property inputs. Any fields not shown are automatically filled using "
        "the training dataset's median (numeric) or most common value "
        "(categorical)."
    )

    default_row = build_default_row(clean_df)

    # ----- Core size & quality -----
    st.markdown("#### Property size and quality")
    c1, c2, c3 = st.columns(3)

    with c1:
        overall_quality = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=int(round(default_row["Overall Quality"].iloc[0])),
        )

        overall_condition = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=int(round(default_row["Overall Condition"].iloc[0])),
        )

        property_size = st.number_input(
            "Property / Lot Size (sq ft)",
            min_value=0.0,
            value=float(default_row["Property Size"].iloc[0]),
            step=100.0,
        )

    with c2:
        living_area = st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=0.0,
            value=float(default_row["Above Ground Living Area"].iloc[0]),
            step=50.0,
        )

        basement_sf = st.number_input(
            "Basement Square Footage",
            min_value=0.0,
            value=float(default_row["Basement Square Footage"].iloc[0]),
            step=50.0,
        )

        first_floor_sf = st.number_input(
            "First Floor Square Footage",
            min_value=0.0,
            value=float(default_row["First Floor Square Footage"].iloc[0]),
            step=50.0,
        )

    with c3:
        second_floor_sf = st.number_input(
            "Second Floor Square Footage",
            min_value=0.0,
            value=float(default_row["Second Floor Square Footage"].iloc[0]),
            step=50.0,
        )

        rooms = st.number_input(
            "Total Rooms Above Ground",
            min_value=1,
            value=max(
                1,
                int(round(
                    default_row["Total Rooms Above Ground"].iloc[0]
                )),
            ),
            step=1,
        )

        lot_frontage = st.number_input(
            "Lot Frontage",
            min_value=0.0,
            value=float(default_row["Lot Frontage"].iloc[0]),
            step=1.0,
        )

    # ----- Age -----
    st.markdown("#### Age and remodeling")
    c1, c2, c3 = st.columns(3)

    with c1:
        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2035,
            value=int(round(default_row["Year Built"].iloc[0])),
            step=1,
        )

    with c2:
        year_remodeled = st.number_input(
            "Year Remodeled / Addition",
            min_value=1800,
            max_value=2035,
            value=int(round(
                default_row["Year Remodeled/Addition"].iloc[0]
            )),
            step=1,
        )

    with c3:
        sale_year = st.number_input(
            "Sale Year",
            min_value=1900,
            max_value=2035,
            value=int(round(default_row["Sale Year"].iloc[0])),
            step=1,
        )

    # ----- Bathrooms / garage / amenities -----
    st.markdown("#### Bathrooms, garage and amenities")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        full_bath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            value=int(round(default_row["Bathroom 1"].iloc[0])),
            step=1,
        )

        half_bath = st.number_input(
            "Half Bathrooms",
            min_value=0,
            value=int(round(default_row["Bathroom 2"].iloc[0])),
            step=1,
        )

    with c2:
        basement_full_bath = st.number_input(
            "Basement Full Bathrooms",
            min_value=0,
            value=int(round(
                default_row["Basement Full Bathrooms"].iloc[0]
            )),
            step=1,
        )

        basement_half_bath = st.number_input(
            "Basement Half Bathrooms",
            min_value=0,
            value=int(round(
                default_row["Basement Half Bathrooms"].iloc[0]
            )),
            step=1,
        )

    with c3:
        garage_cars = st.number_input(
            "Garage Capacity (Cars)",
            min_value=0,
            value=int(round(default_row["Basement Cars"].iloc[0])),
            step=1,
        )

        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0.0,
            value=float(default_row["Basement Area"].iloc[0]),
            step=25.0,
        )

    with c4:
        fireplaces = st.number_input(
            "Number of Fireplaces",
            min_value=0,
            value=int(round(
                default_row["Number of Fireplaces"].iloc[0]
            )),
            step=1,
        )

        pool_area = st.number_input(
            "Pool Area (sq ft)",
            min_value=0.0,
            value=float(default_row["Pool Area"].iloc[0]),
            step=25.0,
        )

    # ----- Quality categorical inputs -----
    st.markdown("#### Finish quality")
    c1, c2, c3 = st.columns(3)

    quality_options = ["Po", "Fa", "TA", "Gd", "Ex"]

    def quality_default(column, fallback="TA"):
        value = str(default_row[column].iloc[0])
        return value if value in quality_options else fallback

    with c1:
        kitchen_quality = st.selectbox(
            "Kitchen Quality",
            quality_options,
            index=quality_options.index(
                quality_default("Kitchen Quality")
            ),
        )

    with c2:
        exterior_quality = st.selectbox(
            "Exterior Quality",
            quality_options,
            index=quality_options.index(
                quality_default("Exterior Quality")
            ),
        )

    with c3:
        heating_quality = st.selectbox(
            "Heating Quality and Condition",
            quality_options,
            index=quality_options.index(
                quality_default("Heating Quality and Condition")
            ),
        )

    # ----- Location / sale -----
    st.markdown("#### Location and sale details")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        neighborhood_options = options_for(clean_df, "Neighborhood")
        default_neighborhood = str(default_row["Neighborhood"].iloc[0])
        neighborhood = st.selectbox(
            "Neighborhood",
            neighborhood_options,
            index=(
                neighborhood_options.index(default_neighborhood)
                if default_neighborhood in neighborhood_options
                else 0
            ),
        )

    with c2:
        zone_options = options_for(clean_df, "Property Zone")
        default_zone = str(default_row["Property Zone"].iloc[0])
        property_zone = st.selectbox(
            "Property Zone",
            zone_options,
            index=(
                zone_options.index(default_zone)
                if default_zone in zone_options
                else 0
            ),
        )

    with c3:
        central_air_options = options_for(
            clean_df,
            "Central Air Conditioning",
        )
        default_air = str(
            default_row["Central Air Conditioning"].iloc[0]
        )
        central_air = st.selectbox(
            "Central Air Conditioning",
            central_air_options,
            index=(
                central_air_options.index(default_air)
                if default_air in central_air_options
                else 0
            ),
        )

    with c4:
        month_options = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        default_month = str(default_row["Sale Month"].iloc[0])[:3].title()
        sale_month = st.selectbox(
            "Sale Month",
            month_options,
            index=(
                month_options.index(default_month)
                if default_month in month_options
                else 5
            ),
        )

    # ----- Porch / masonry -----
    with st.expander("Optional structural details"):
        c1, c2, c3 = st.columns(3)

        with c1:
            wood_deck = st.number_input(
                "Wood Deck Square Footage",
                min_value=0.0,
                value=float(default_row["Wood Deck Square Footage"].iloc[0]),
                step=10.0,
            )
            open_porch = st.number_input(
                "Open Porch Square Footage",
                min_value=0.0,
                value=float(default_row["Open Porch Square Footage"].iloc[0]),
                step=10.0,
            )

        with c2:
            enclosed_porch = st.number_input(
                "Enclosed Porch",
                min_value=0.0,
                value=float(default_row["Enclosed Porch"].iloc[0]),
                step=10.0,
            )
            screen_porch = st.number_input(
                "Screen Porch",
                min_value=0.0,
                value=float(default_row["Screen Porch"].iloc[0]),
                step=10.0,
            )

        with c3:
            three_season = st.number_input(
                "Three Season Porch",
                min_value=0.0,
                value=float(default_row["Three Season Porch"].iloc[0]),
                step=10.0,
            )
            masonry_area = st.number_input(
                "Exterior Cladding / Masonry Area",
                min_value=0.0,
                value=float(default_row["ExteriorCladdingArea"].iloc[0]),
                step=10.0,
            )

    if st.button(
        "Predict Property Price",
        type="primary",
        use_container_width=True,
    ):
        # Convert the default one-row DataFrame to a plain dictionary first.
        # This avoids Pandas dtype errors when a user-selected string value
        # (for example Sale Month or a categorical option) replaces a value
        # in a column that Pandas originally inferred as numeric/category.
        row_data = default_row.iloc[0].to_dict()

        updates = {
            "Overall Quality": overall_quality,
            "Overall Condition": overall_condition,
            "Property Size": property_size,
            "Above Ground Living Area": living_area,
            "Basement Square Footage": basement_sf,
            "First Floor Square Footage": first_floor_sf,
            "Second Floor Square Footage": second_floor_sf,
            "Total Rooms Above Ground": rooms,
            "Lot Frontage": lot_frontage,
            "Year Built": year_built,
            "Year Remodeled/Addition": year_remodeled,
            "Sale Year": sale_year,
            "Bathroom 1": full_bath,
            "Bathroom 2": half_bath,
            "Basement Full Bathrooms": basement_full_bath,
            "Basement Half Bathrooms": basement_half_bath,
            "Basement Cars": garage_cars,
            "Basement Area": garage_area,
            "Number of Fireplaces": fireplaces,
            "Pool Area": pool_area,
            "Kitchen Quality": kitchen_quality,
            "Exterior Quality": exterior_quality,
            "Heating Quality and Condition": heating_quality,
            "Neighborhood": neighborhood,
            "Property Zone": property_zone,
            "Central Air Conditioning": central_air,
            "Sale Month": sale_month,
            "Wood Deck Square Footage": wood_deck,
            "Open Porch Square Footage": open_porch,
            "Enclosed Porch": enclosed_porch,
            "Screen Porch": screen_porch,
            "Three Season Porch": three_season,
            "ExteriorCladdingArea": masonry_area,
        }

        for col, value in updates.items():
            if col in row_data:
                row_data[col] = value

        # Build a fresh DataFrame after all user inputs are applied.
        # Pandas can now infer compatible dtypes from the final values.
        row = pd.DataFrame([row_data])

        engineered_row = engineer_features(row)

        model_row = transform_new_rows(
            engineered_row,
            ordinal_cols,
            nominal_cols,
            ordinal_meta,
            feature_columns,
        )

        pred_log = model.predict(model_row)[0]
        predicted_price = float(np.expm1(pred_log))

        st.success(
            f"Estimated Property Price: ${predicted_price:,.0f}"
        )

        st.caption(
            "This is a machine-learning estimate, not a formal appraisal. "
            "Unshown property fields use typical values from the training data."
        )

        result_cols = st.columns(3)
        result_cols[0].metric(
            "Predicted Price",
            f"${predicted_price:,.0f}",
        )
        result_cols[1].metric(
            "Model Holdout R²",
            f"{metrics['r2'] * 100:.2f}%",
        )
        result_cols[2].metric(
            "Model MAE",
            f"${metrics['mae']:,.0f}",
        )


# =========================================================
# BATCH PREDICTION
# =========================================================

with batch_tab:
    st.subheader("Batch Prediction")

    st.write(
        "Upload a CSV containing property rows. It may use either the original "
        "dataset column names (for example `OverallQual`, `GrLivArea`) or the "
        "renamed notebook columns. Missing columns are filled with the training "
        "data's median/mode defaults."
    )

    uploaded = st.file_uploader(
        "Upload property CSV",
        type=["csv"],
    )

    if uploaded is not None:
        incoming_raw = pd.read_csv(uploaded)

        # Rename if the CSV uses original names.
        incoming_raw = incoming_raw.rename(
            columns=RENAME_MAP
        )

        defaults = build_default_row(clean_df)
        expected_raw_cols = defaults.columns.tolist()

        # Fill missing raw columns from training defaults.
        for col in expected_raw_cols:
            if col not in incoming_raw.columns:
                incoming_raw[col] = defaults[col].iloc[0]

        incoming_raw = incoming_raw[expected_raw_cols].copy()

        # Apply notebook-style missing value rules for uploaded rows.
        for col in CATEGORICAL_MISSING_COLUMNS:
            if col in incoming_raw.columns:
                incoming_raw[col] = incoming_raw[col].fillna("None")

        for col in NUMERICAL_ZERO_COLUMNS:
            if col in incoming_raw.columns:
                incoming_raw[col] = pd.to_numeric(
                    incoming_raw[col],
                    errors="coerce",
                ).fillna(0)

        for col in expected_raw_cols:
            if incoming_raw[col].isna().any():
                incoming_raw[col] = incoming_raw[col].fillna(
                    defaults[col].iloc[0]
                )

        engineered_batch = engineer_features(incoming_raw)

        model_batch = transform_new_rows(
            engineered_batch,
            ordinal_cols,
            nominal_cols,
            ordinal_meta,
            feature_columns,
        )

        batch_pred = np.expm1(
            model.predict(model_batch)
        )

        output_df = incoming_raw.copy()
        output_df["Predicted Property Price (USD)"] = (
            batch_pred.round(0)
        )

        st.success(
            f"Predictions generated for {len(output_df)} properties."
        )

        st.dataframe(
            output_df[
                ["Predicted Property Price (USD)"]
            ].head(50),
            use_container_width=True,
        )

        csv_bytes = output_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="property_price_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# ABOUT
# =========================================================

with about_tab:
    st.subheader("About This Capstone App")

    st.markdown(
        """
        This application follows the preprocessing and modeling logic used in
        the uploaded capstone notebook:

        - missing-value treatment without dropping amenity columns
        - renamed, readable property variables
        - engineered features such as **HouseAge**, **TotalSF**,
          **TotalBathrooms**, **QualLivArea**, **QualCond**, and
          **GarageScore**
        - separate ordinal and nominal encoding
        - log transformation of the target price using `log1p`
        - **Tuned Gradient Boosting** as the final prediction model
        - predictions converted back to dollars using `expm1`

        The final notebook found that quality and usable living space were
        especially important drivers of predicted property price.
        """
    )

    st.warning(
        "For deployment, keep `app.py` and `Propertydata.csv` in the same "
        "repository folder. You may also include "
        "`tuned_gradient_boosting_model.pkl`; if it is absent, this app "
        "re-trains the final tuned Gradient Boosting configuration from "
        "`Propertydata.csv` and caches it."
    )
