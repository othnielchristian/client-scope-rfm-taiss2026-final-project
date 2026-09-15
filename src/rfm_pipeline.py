"""
Reusable RFM pipeline extracted from notebooks 01 and 02.

This module lets the Streamlit app run the same cleaning -> RFM feature
engineering -> clustering steps that the notebooks perform manually, on
whatever transactional file a visitor uploads.

Expected raw columns (same schema as the UCI "Online Retail II" dataset):
    Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
    CustomerID (or "Customer ID"), Country
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.clustering import fit_segmentation, summarize_segments

REQUIRED_COLUMNS = {
    "CustomerID", "Invoice", "InvoiceDate", "Quantity", "Price",
    "StockCode", "Description", "Country",
}

VALID_SCALED_COLUMNS = [
    "recence_jours_scaled",
    "frequence_scaled",
    "montant_total_scaled",
]
GLOBAL_SCALED_COLUMNS = [
    "recence_globale_jours_scaled",
    "frequence_globale_scaled",
    "montant_global_scaled",
]


class PipelineError(ValueError):
    """Raised when the uploaded file doesn't match the expected schema."""


def _get_stock_code(code: object):
    """Same helper as src/cleaning.py: keep digits only, drop non-numeric codes."""
    if isinstance(code, str):
        digits = "".join(ch for ch in code if ch.isnumeric())
        if len(digits) == 0:
            return np.nan
        return int(digits)
    if isinstance(code, int):
        return code
    return np.nan


def load_raw(uploaded_file) -> pd.DataFrame:
    """Read an uploaded .csv or .xlsx into a single raw dataframe."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(uploaded_file)

    df = df.rename(columns={"Customer ID": "CustomerID"})

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise PipelineError(
            "Missing required column(s): " + ", ".join(sorted(missing))
            + f". Found columns: {list(df.columns)}"
        )
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror notebook 01: dedupe, drop missing customers, normalise types."""
    df = df.drop_duplicates(subset=df.columns)
    df = df.dropna(subset=["CustomerID"]).reset_index(drop=True)
    df["CustomerID"] = df["CustomerID"].astype(int)

    df["StockCode"] = df["StockCode"].apply(_get_stock_code)
    df = df.dropna(subset=["StockCode"]).reset_index(drop=True)
    df["StockCode"] = df["StockCode"].astype(int)

    df["Invoice"] = df["Invoice"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def build_features(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Mirror notebook 02: RFM + global RFM + profile + log/scale transforms."""
    df = df_cleaned.copy()
    df["TotalAmount"] = df["Quantity"] * df["Price"]

    is_cancellation = df["Invoice"].str.startswith("C")
    is_valid_purchase = (
        ~is_cancellation
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & df["CustomerID"].notna()
    )
    df_purchases = df.loc[is_valid_purchase].copy()
    df_customer_transactions = df.loc[df["CustomerID"].notna()].copy()

    if df_purchases.empty:
        raise PipelineError("No valid (non-cancelled, positive amount) purchases found in this file.")

    date_reference = df_customer_transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm_columns = ["recence_jours", "frequence", "montant_total"]
    global_rfm_columns = ["recence_globale_jours", "frequence_globale", "montant_global"]

    rfm = df_purchases.groupby("CustomerID").agg(
        recence_jours=("InvoiceDate", lambda dates: (date_reference - dates.max()).days),
        frequence=("Invoice", "nunique"),
        montant_total=("TotalAmount", "sum"),
    )

    rfm_global = df_customer_transactions.groupby("CustomerID").agg(
        recence_globale_jours=("InvoiceDate", lambda dates: (date_reference - dates.max()).days),
        frequence_globale=("Invoice", "nunique"),
        montant_global=("TotalAmount", "sum"),
    ).reindex(rfm.index)

    profil_client = df_purchases.groupby("CustomerID").agg(
        pays_residence=("Country", lambda values: values.mode().iat[0]),
        nb_produits_distincts=("StockCode", "nunique"),
    )

    annulations = (
        df.loc[is_cancellation & df["CustomerID"].notna()]
        .groupby("CustomerID")
        .size()
        .rename("nb_annulations")
    )

    produit_top = (
        df_purchases.groupby(["CustomerID", "StockCode", "Description"], dropna=False)["Quantity"]
        .sum()
        .rename("quantite_achetee")
        .reset_index()
        .sort_values(["CustomerID", "quantite_achetee", "StockCode"], ascending=[True, False, True])
        .drop_duplicates("CustomerID")
        .set_index("CustomerID")
        .rename(columns={"StockCode": "produit_top_code", "Description": "produit_top_description"})
        [["produit_top_code", "produit_top_description"]]
    )

    features_clients = (
        rfm.join(rfm_global)
        .join(profil_client)
        .join(annulations)
        .join(produit_top)
        .fillna({"nb_annulations": 0})
        .reset_index()
        .astype({"nb_annulations": "int64"})
    )

    # Log-transform + standardise, exactly as in notebook 02.
    log_groups = {"valid": rfm_columns, "global": global_rfm_columns}
    for _, columns in log_groups.items():
        log_columns = [f"{c}_log" for c in columns]
        scaled_columns = [f"{c}_scaled" for c in columns]
        values = features_clients[columns]

        if "montant_global" in columns:
            transformed = values.copy()
            for column in columns:
                if column == "montant_global":
                    transformed[column] = np.sign(values[column]) * np.log1p(values[column].abs())
                else:
                    transformed[column] = np.log1p(values[column])
        else:
            transformed = np.log1p(values)

        features_clients[log_columns] = transformed.to_numpy()
        features_clients[scaled_columns] = StandardScaler().fit_transform(features_clients[log_columns])

    return features_clients


def run_pipeline(uploaded_file, k_valid: int = 4, k_global: int = 4):
    """End-to-end: raw file -> cleaned -> features -> clustered -> summaries.

    Returns a dict with, for each approach ("valid" and "global"):
      - "segmented": per-customer dataframe with a "segment" column
      - "summary": one row per segment with size, value, and recommendation
    """
    raw = load_raw(uploaded_file)
    cleaned = clean_transactions(raw)
    features = build_features(cleaned)

    n_customers = len(features)
    results = {}
    approaches = {
        "valid": (VALID_SCALED_COLUMNS, min(k_valid, max(2, n_customers - 1))),
        "global": (GLOBAL_SCALED_COLUMNS, min(k_global, max(2, n_customers - 1))),
    }
    for approach, (feature_columns, k) in approaches.items():
        _, segmented = fit_segmentation(features, feature_columns, k)
        summary = summarize_segments(segmented, approach)
        results[approach] = {"segmented": segmented, "summary": summary}

    return results