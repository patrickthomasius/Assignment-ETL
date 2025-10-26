# interactive_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import os

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "start"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "postgres"

# -----------------------------
# Load data from Postgres
# -----------------------------

engine = create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


@st.cache_data
def load_table(table_name):
    with engine.connect() as conn:
        df = pd.read_sql_table(table_name, conn)
    return df

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="ETL Data Distribution Explorer", layout="wide")

st.title("📊 ETL Data Distribution Dashboard")

tables = ["diagnoses", "patients", "encounters", "logs"]
table_choice = st.selectbox("Select table to explore:", tables)

df = load_table(table_choice)

st.write(f"### Preview of `{table_choice}`")
st.dataframe(df.head())

# -----------------------------
# Interactive Data Distribution
# -----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

tab1, tab2, tab3, tab4 = st.tabs(["📈 Numeric", "📆 Date/Time", "🔤 Categorical", "Data Quality"])
if table_choice != "etl_logs":

    with tab1:
        if numeric_cols:
            st.write("### Numeric Column Visualization")

            plot_type = st.radio(
                "Choose plot type:",
                ["Histogram", "Scatterplot"],
                horizontal=True,
            )

            if plot_type == "Histogram":
                num_col = st.selectbox("Select numeric column:", numeric_cols)
                fig = px.histogram(df, x=num_col, nbins=30, title=f"Distribution of {num_col}")
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Scatterplot":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Select X-axis column:", numeric_cols, key="xcol")
                with col2:
                    y_col = st.selectbox("Select Y-axis column:", numeric_cols, key="ycol")

                color_col = st.selectbox(
                    "Optional: Color by (categorical or boolean):",
                    [None] + categorical_cols,
                    index=0,
                    key="colorcol",
                )

                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col if color_col else None,
                    title=f"{y_col} vs {x_col}" + (f" colored by {color_col}" if color_col else ""),
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No numeric columns in this table.")

    with tab2:
        if datetime_cols:
            date_col = st.selectbox("Select date/time column:", datetime_cols)
            df["_date"] = pd.to_datetime(df[date_col])
            df["_count"] = 1
            fig = px.histogram(
                df,
                x="_date",
                y="_count",
                title=f"Records over time by {date_col}",
                nbins=50,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No datetime columns in this table.")

    with tab3:
        if categorical_cols:
            cat_col = st.selectbox("Select categorical column:", categorical_cols)
            top_n = st.slider("Show top N categories:", 5, 50, 10)
            cat_counts = df[cat_col].value_counts().nlargest(top_n).reset_index()
            cat_counts.columns = [cat_col, "count"]
            fig = px.bar(cat_counts, x=cat_col, y="count", title=f"Top {top_n} values of {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns in this table.")
else:
    # ---------------------------
    # 🧹 TAB: Data Quality Logs
    # ---------------------------
    st.header("🧹 Data Quality Overview (from etl_logs table)")

    if df.empty:
        st.warning("The etl_logs table is empty.")
    else:
        # 1️⃣ Frequency of Data Quality Issues
        issue_counts = df["reason"].value_counts().reset_index()
        issue_counts.columns = ["reason", "count"]
        fig1 = px.bar(
            issue_counts,
            x="reason",
            y="count",
            text="count",
            title="Frequency of Data Quality Issues",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2️⃣ Issues per Source File
        if "filename" in df.columns:
            file_issue_counts = df.groupby(["filename", "reason"]).size().reset_index(name="count")
            fig2 = px.bar(
                file_issue_counts,
                x="filename",
                y="count",
                color="reason",
                title="Data Quality Issues per Source File",
                barmode="group",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 3️⃣ Affected Patients
        if "patient_id" in df.columns:
            total_patients = df["patient_id"].nunique()
            issues_by_patient = df.groupby("patient_id")["reason"].nunique().reset_index()
            affected_patients = len(issues_by_patient)
            st.metric(
                "🧍 Patients with Data Quality Issues",
                f"{affected_patients} ({affected_patients/total_patients:.0%})",
            )

        # 4️⃣ Drilldown: View by Reason
        st.write("### 🔍 Inspect Specific Issue Type")
        selected_reason = st.selectbox("Select reason to inspect:", df["reason"].unique())
        st.dataframe(
            df[df["reason"] == selected_reason][
                ["patient_id", "filename", "original_value", "cleaned_value", "reason"]
            ]
        )

        # 5️⃣ Optional: Allow CSV download
        st.download_button(
            label="📥 Download Filtered Log as CSV",
            data=df[df["reason"] == selected_reason].to_csv(index=False),
            file_name=f"log_{selected_reason}.csv",
            mime="text/csv",
        )
