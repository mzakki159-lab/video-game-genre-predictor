import numpy as np
import pandas as pd
import plotly.express as px
from pycaret.classification import compare_models, load_model, predict_model, save_model, setup
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

# Custom CSS Styling
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

    # Handling missing values (Sesuai eksplorasi notebook)
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
        f"❌ Gagal memuat file `vgsales.csv`. Pastikan file berada di folder yang sama! Detail error: {e}"
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
    st.caption("Powered by PyCaret & Streamlit")

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
            <p>Latih dan bandingkan algoritma terbaik menggunakan PyCaret AutoML</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.info(
        "💡 **Perbaikan Model:** Fitur `Name` dan `Rank` diabaikan (`ignore_features`), data training dinaikkan menjadi 80% (`train_size=0.8`), serta diaktifkan penyeimbangan kelas (`fix_imbalance=True`) agar hasil prediksi tidak selalu didominasi oleh satu genre tertentu."
    )

    if st.button(
        "🚀 Mulai Auto-ML Training", type="primary", use_container_width=True
    ):
        with st.spinner(
            "Sedang melatih dan mengevaluasi model PyCaret... (Membutuhkan waktu 1-2 menit)"
        ):
            # SETUP PYCARET TERBARU (BEBAS BIAS)
            clf_setup = setup(
                data=df_clean,
                target="Genre",
                ignore_features=[
                    "Name",
                    "Rank",
                ],  # Mengabaikan Nama & Rank agar tidak bias/overfit
                train_size=0.8,  # Menggunakan 80% data untuk training
                fix_imbalance=True,  # Menyeimbangkan distribusi genre
                session_id=123,
                verbose=False,
            )

            best_model = compare_models()
            save_model(best_model, "best_game_genre_model")

        st.balloons()
        st.success(
            "🎉 Training Selesai! Model baru yang bervariasi dan akurat berhasil disimpan."
        )

        st.subheader("🏆 Model Terbaik yang Dipilih:")
        st.code(str(best_model))

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

    model_loaded = False
    try:
        model = load_model("best_game_genre_model")
        model_loaded = True
    except:
        st.warning(
            "⚠️ **Model Belum Ada / Belum Dilatih!** Silakan masuk ke menu **Model Training Center** terlebih dahulu dan klik tombol **Mulai Auto-ML Training**."
        )

    if model_loaded:
        with st.form("prediction_form"):
            st.subheader("📝 Parameter Input Game")

            c1, c2 = st.columns(2)

            with c1:
                rank = st.number_input(
                    "Rank Game (Diabaikan Model)", min_value=1, value=100, step=1
                )
                name = st.text_input(
                    "Nama Game (Diabaikan Model)", value="Call of Duty"
                )
                platform = st.selectbox(
                    "Platform Konsol",
                    options=sorted(df_raw["Platform"].unique()),
                )
                year = st.slider(
                    "Tahun Rilis",
                    min_value=1980,
                    max_value=2026,
                    value=2015,
                )
                publisher = st.selectbox(
                    "Publisher", options=sorted(df_raw["Publisher"].unique())
                )

            with c2:
                st.write("💰 **Penjualan (dalam Juta Unit):**")
                na_sales = st.number_input(
                    "North America Sales (NA)",
                    min_value=0.0,
                    value=9.5,
                    step=0.1,
                )
                eu_sales = st.number_input(
                    "Europe Sales (EU)", min_value=0.0, value=5.8, step=0.1
                )
                jp_sales = st.number_input(
                    "Japan Sales (JP)", min_value=0.0, value=0.2, step=0.1
                )
                other_sales = st.number_input(
                    "Other Region Sales", min_value=0.0, value=1.5, step=0.1
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
            input_data = pd.DataFrame(
                [
                    {
                        "Rank": rank,
                        "Name": name,
                        "Platform": platform,
                        "Year": year,
                        "Publisher": publisher,
                        "NA_Sales": na_sales,
                        "EU_Sales": eu_sales,
                        "JP_Sales": jp_sales,
                        "Other_Sales": other_sales,
                        "Global_Sales": global_sales,
                    }
                ]
            )

            with st.spinner("Menganalisis data..."):
                prediction = predict_model(model, data=input_data)

                if "prediction_label" in prediction.columns:
                    predicted_genre = prediction["prediction_label"].iloc[0]
                else:
                    predicted_genre = prediction["Label"].iloc[0]

            st.toast("Prediksi Berhasil!", icon="🎯")

            st.markdown(
                f"""
                <div class="result-box">
                    Hasil Prediksi Genre: <b>{str(predicted_genre).upper()}</b>
                </div>
            """,
                unsafe_allow_html=True,
            )