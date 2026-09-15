"""
Client-Scope RFM — Streamlit dashboard
TAISS 2026 final project (othnielchristian/client-scope-rfm-taiss2026-final-project)

Two tabs:
  1. Segment Dashboard  -> shows the finished analysis (reports/tableau_synthese_segments.csv)
  2. Upload Your Data   -> lets a visitor run the RFM + K-means pipeline on their own file
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.rfm_pipeline import PipelineError, run_pipeline

st.set_page_config(
    page_title="Client-Scope RFM Dashboard",
    page_icon="📊",
    layout="wide",
)

REPORT_PATH = Path(__file__).parent / "reports" / "tableau_synthese_segments.csv"

st.title("📊 Client-Scope RFM — Customer Segmentation")
st.caption("TAISS 2026 final project · Online Retail II · RFM + K-means")

tab_dashboard, tab_upload = st.tabs(["📈 Segment Dashboard", "📤 Upload Your Own Data"])

# ----------------------------------------------------------------------
# TAB 1 — Finished results
# ----------------------------------------------------------------------
with tab_dashboard:
    if not REPORT_PATH.exists():
        st.warning(
            "Couldn't find `reports/tableau_synthese_segments.csv`. "
            "Make sure that file is committed to the repo."
        )
    else:
        df = pd.read_csv(REPORT_PATH)

        total_clients = int(df["Nb clients"].sum())
        total_revenue_share = df["% du CA"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Segments", len(df))
        col2.metric("Total customers", f"{total_clients:,}")
        col3.metric("Highest-value segment", f"#{df.loc[df['Indice de valeur'].idxmax(), 'Segment']}")

        st.divider()

        left, right = st.columns(2)
        with left:
            fig_size = px.bar(
                df, x="Segment", y="Nb clients", color="Segment",
                title="Customers per segment", text="Nb clients",
            )
            st.plotly_chart(fig_size, use_container_width=True)
        with right:
            fig_value = px.pie(
                df, names="Segment", values="% du CA",
                title="Share of revenue by segment",
            )
            st.plotly_chart(fig_value, use_container_width=True)

        fig_scatter = px.scatter(
            df, x="Récence moy. (j)", y="Panier moyen (£)",
            size="Nb clients", color="Segment", text="Segment",
            title="Recency vs. average basket, sized by segment size",
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Segment summary & recommended action")
        st.dataframe(df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# TAB 2 — Live pipeline on an uploaded file
# ----------------------------------------------------------------------
with tab_upload:
    st.markdown(
        "Upload a transactional file with the **same schema as UCI Online Retail II**: "
        "`Invoice, StockCode, Description, Quantity, InvoiceDate, Price, "
        "CustomerID (or 'Customer ID'), Country`."
    )

    uploaded_file = st.file_uploader("Transaction file (.csv or .xlsx)", type=["csv", "xlsx", "xls"])

    col_k1, col_k2 = st.columns(2)
    k_valid = col_k1.slider("Number of segments — valid purchases (k)", 2, 8, 4)
    k_global = col_k2.slider("Number of segments — global activity (k)", 2, 8, 4)

    run_clicked = st.button("Run segmentation", type="primary", disabled=uploaded_file is None)

    if run_clicked and uploaded_file is not None:
        with st.spinner("Cleaning data, building RFM features, and running K-means..."):
            try:
                results = run_pipeline(uploaded_file, k_valid=k_valid, k_global=k_global)
            except PipelineError as exc:
                st.error(str(exc))
                results = None
            except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user
                st.error(f"Something went wrong while processing this file: {exc}")
                results = None

        if results is not None:
            st.success("Segmentation complete.")
            approach_tabs = st.tabs(["Valid purchases only", "Global activity"])

            for approach, tab in zip(["valid", "global"], approach_tabs):
                with tab:
                    summary = results[approach]["summary"]
                    segmented = results[approach]["segmented"]

                    st.dataframe(summary, use_container_width=True, hide_index=True)

                    fig = px.bar(
                        summary, x="segment", y="effectif", color="segment",
                        title=f"Customers per segment ({approach} approach)",
                        text="effectif",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        f"Download segmented customers ({approach}).csv",
                        data=segmented.to_csv(index=False).encode("utf-8"),
                        file_name=f"clients_segmentes_{approach}.csv",
                        mime="text/csv",
                    )

st.divider()
st.caption(
    "Source code: "
    "[GitHub repo](https://github.com/othnielchristian/client-scope-rfm-taiss2026-final-project)"
)