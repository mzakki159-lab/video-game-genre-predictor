import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Video Game Genre AI Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #d1d8e0 !important;
        margin-top: 5px;
        font-size: 16px;
    }
    .result-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 22px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2a5298;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. LOAD & PREPROCESS DATASET
# ==========================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("vgsales.csv")

    # Handling missing values
    df["Year"] = df["Year"].fillna(df["Year"].median())
    df["Publisher"] = df["Publisher"].fillna(df["Publisher"].mode()[0])

    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Capping Outliers (Winsorizing)
    df_clean = df.copy()
    for col in num_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.where(
            df_clean[col] < lower,
            lower,
            np.where(df_clean[col] > upper, upper, df_clean[col]),
        )

    return df, df_clean


try:
    df_raw, df_clean = load_and_clean_data()
except Exception as e:
    st.error(
        f"❌ Gagal memuat file `vgsales.csv`. Pastikan file berada di folder yang sama! Detail: {e}"
    )
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/686/686589.png", width=90
    )
    st.title("🎮 Navigasi Menu")
    menu = st.radio(
        "Pilih Halaman:",
        ["Dashboard & Analytics", "Model Training Center", "Prediksi Interaktif"],
        index=0,
    )
    st.divider()
    st.caption("Powered by Scikit-Learn & Streamlit")

# ==========================================
# MENU 1: DASHBOARD & ANALYTICS
# ==========================================
if menu == "Dashboard & Analytics":
    st.markdown(
        """
        <div class="main-header">
            <h1>🎮 Video Game Sales Analytics</h1>
            <p>Eksplorasi visual interaktif dataset penjualan video game di seluruh dunia</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Game", f"{len(df_raw):,}")
    m2.metric("Total Platform", f"{df_raw['Platform'].nunique()}")
    m3.metric("Total Publisher", f"{df_raw['Publisher'].nunique()}")
    m4.metric("Global Sales (M)", f"${df_raw['Global_Sales'].sum():,.2f}")

    st.divider()

    st.subheader("📊 Visualisasi Data Interaktif")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        genre_counts = df_raw["Genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        fig_pie = px.pie(
            genre_counts,
            values="Count",
            names="Genre",
            title="Distribusi Genre Game",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        top_platforms = (
            df_raw["Platform"].value_counts().head(10).reset_index()
        )
        top_platforms.columns = ["Platform", "Count"]
        fig_bar = px.bar(
            top_platforms,
            x="Platform",
            y="Count",
            title="Top 10 Platform Populer",
            color="Count",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("🔍 Lihat Detail Data Tabular"):
        st.dataframe(df_raw, use_container_width=True)

# ==========================================
# MENU 2: MODEL TRAINING CENTER
# ==========================================
elif menu == "Model Training Center":
    st.markdown(
        """
        <div class="main-header">
            <h1>⚙️ Machine Learning Model Hub</h1>
            <p>Latih model Klasifikasi (Random Forest) secara cepat dan akurat</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.info(
        "💡 Klik tombol di bawah untuk melatih model Random Forest Classifier pada data yang sudah di-clean."
    )

    if st.button(
        "🚀 Mulai Training Model", type="primary", use_container_width=True
    ):
        with st.spinner("Sedang memproses data dan melatih model..."):
            # Prepare features & target
            X = df_clean[
                [
                    "Platform",
                    "Year",
                    "Publisher",
                    "NA_Sales",
                    "EU_Sales",
                    "JP_Sales",
                    "Other_Sales",
                    "Global_Sales",
                ]
            ]
            y = df_clean["Genre"]

            # One-Hot Encoding for categorical features
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_cols = ["Platform", "Publisher"]
            X_encoded = pd.DataFrame(
                encoder.fit_transform(X[cat_cols]),
                columns=encoder.get_feature_names_out(cat_cols),
            )

            num_cols = [
                "Year",
                "NA_Sales",
                "EU_Sales",
                "JP_Sales",
                "Other_Sales",
                "Global_Sales",
            ]
            X_final = pd.concat([X[num_cols].reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

            # Train Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y, test_size=0.2, random_state=123
            )

            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=100, random_state=123, n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Save objects in session_state
            st.session_state["rf_model"] = model
            st.session_state["encoder"] = encoder
            st.session_state["feature_cols"] = X_final.columns.tolist()

        st.balloons()
        st.success(f"🎉 Model Berhasil Dilatih! Akurasi Model: **{acc * 100:.2f}%**")

# ==========================================
# MENU 3: PREDIKSI INTERAKTIF
# ==========================================
elif menu == "Prediksi Interaktif":
    st.markdown(
        """
        <div class="main-header">
            <h1>🔮 Prediksi Genre Video Game</h1>
            <p>Input detail parameter game untuk memprediksi genre secara real-time</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if "rf_model" not in st.session_state:
        st.warning(
            "⚠️ **Model Belum Dilatih!** Silakan masuk ke menu **Model Training Center** terlebih dahulu dan klik tombol **Mulai Training Model**."
        )
    else:
        with st.form("prediction_form"):
            st.subheader("📝 Parameter Input Game")

            c1, c2 = st.columns(2)

            with c1:
                rank = st.number_input("Rank Game", min_value=1, value=100, step=1)
                name = st.text_input("Nama Game", value="Super Mario Odyssey")
                platform = st.selectbox(
                    "Platform Konsol",
                    options=sorted(df_raw["Platform"].unique()),
                )
                year = st.slider(
                    "Tahun Rilis",
                    min_value=1980,
                    max_value=2026,
                    value=2017,
                )
                publisher = st.selectbox(
                    "Publisher", options=sorted(df_raw["Publisher"].unique())
                )

            with c2:
                st.write("💰 **Penjualan (dalam Juta Unit):**")
                na_sales = st.number_input(
                    "North America Sales (NA)",
                    min_value=0.0,
                    value=0.5,
                    step=0.1,
                )
                eu_sales = st.number_input(
                    "Europe Sales (EU)", min_value=0.0, value=0.4, step=0.1
                )
                jp_sales = st.number_input(
                    "Japan Sales (JP)", min_value=0.0, value=0.3, step=0.1
                )
                other_sales = st.number_input(
                    "Other Region Sales", min_value=0.0, value=0.1, step=0.1
                )

                global_sales = na_sales + eu_sales + jp_sales + other_sales
                st.info(
                    f"💡 Total Global Sales (Otomatis): **{global_sales:.2f} M**"
                )

            submit_btn = st.form_submit_button(
                "✨ Prediksi Genre Sekarang",
                type="primary",
                use_container_width=True,
            )

        if submit_btn:
            model = st.session_state["rf_model"]
            encoder = st.session_state["encoder"]
            feature_cols = st.session_state["feature_cols"]

            # Prepare single input dataframe
            input_df = pd.DataFrame(
                [
                    {
                        "Platform": platform,
                        "Publisher": publisher,
                        "Year": year,
                        "NA_Sales": na_sales,
                        "EU_Sales": eu_sales,
                        "JP_Sales": jp_sales,
                        "Other_Sales": other_sales,
                        "Global_Sales": global_sales,
                    }
                ]
            )

            # Transform input
            input_cat = pd.DataFrame(
                encoder.transform(input_df[["Platform", "Publisher"]]),
                columns=encoder.get_feature_names_out(["Platform", "Publisher"]),
            )
            input_num = input_df[
                [
                    "Year",
                    "NA_Sales",
                    "EU_Sales",
                    "JP_Sales",
                    "Other_Sales",
                    "Global_Sales",
                ]
            ]
            input_final = pd.concat([input_num, input_cat], axis=1)

            # Ensure all feature columns match training columns
            for col in feature_cols:
                if col not in input_final.columns:
                    input_final[col] = 0
            input_final = input_final[feature_cols]

            # Predict
            predicted_genre = model.predict(input_final)[0]

            st.toast("Prediksi Berhasil!", icon="🎯")

            st.markdown(
                f"""
                <div class="result-box">
                    Hasil Prediksi Genre: <b>{str(predicted_genre).upper()}</b>
                </div>
            """,
                unsafe_allow_html=True,
            )
