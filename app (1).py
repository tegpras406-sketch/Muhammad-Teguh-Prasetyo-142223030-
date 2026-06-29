import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superconductor Critical Temperature Predictor",
    page_icon="🧲",
    layout="wide"
)

# ─── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("superconduct_train.csv")
    return df

df = load_data()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🧲 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", [
    "📊 Eksplorasi Data",
    "📈 Visualisasi",
    "🤖 Prediksi ML",
    "🔍 Prediksi Manual"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** `superconduct_train.csv`")
st.sidebar.markdown(f"**Jumlah baris:** {df.shape[0]:,}")
st.sidebar.markdown(f"**Jumlah fitur:** {df.shape[1] - 1}")

# ─── Feature Groups ─────────────────────────────────────────────────────────────
PROPERTIES = ["atomic_mass", "fie", "atomic_radius", "Density",
               "ElectronAffinity", "FusionHeat", "ThermalConductivity", "Valence"]
STATS = ["mean", "wtd_mean", "gmean", "wtd_gmean", "entropy",
         "wtd_entropy", "range", "wtd_range", "std", "wtd_std"]

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EKSPLORASI DATA
# ════════════════════════════════════════════════════════════════════════════════
if menu == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data Superkonduktor")
    st.markdown("""
    Dataset ini berisi **21.263 material superkonduktor** dengan **81 fitur fisikokimia**
    yang digunakan untuk memprediksi **suhu kritis (critical temperature)** dalam Kelvin.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sampel", f"{df.shape[0]:,}")
    col2.metric("Jumlah Fitur", df.shape[1] - 1)
    col3.metric("Suhu Kritis Rata-rata", f"{df['critical_temp'].mean():.2f} K")
    col4.metric("Suhu Kritis Maks.", f"{df['critical_temp'].max():.2f} K")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗂️ Preview Data", "📋 Statistik", "❓ Missing Values"])

    with tab1:
        st.subheader("5 Baris Pertama")
        st.dataframe(df.head(), use_container_width=True)

    with tab2:
        st.subheader("Statistik Deskriptif")
        cols_show = st.multiselect(
            "Pilih kolom untuk ditampilkan:",
            df.columns.tolist(),
            default=["number_of_elements", "mean_atomic_mass", "mean_Valence", "critical_temp"]
        )
        if cols_show:
            st.dataframe(df[cols_show].describe().round(4), use_container_width=True)

    with tab3:
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ Tidak ada missing values dalam dataset ini!")
        else:
            st.dataframe(missing[missing > 0])

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VISUALISASI
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "📈 Visualisasi":
    st.title("📈 Visualisasi Data")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Distribusi Target",
        "🔗 Korelasi Fitur",
        "📉 Scatter Plot",
        "📦 Box Plot per Elemen"
    ])

    with tab1:
        st.subheader("Distribusi Suhu Kritis (critical_temp)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df["critical_temp"], bins=60, color="#4C72B0", edgecolor="white")
        axes[0].set_title("Distribusi Suhu Kritis")
        axes[0].set_xlabel("Suhu Kritis (K)")
        axes[0].set_ylabel("Frekuensi")

        axes[1].hist(np.log1p(df["critical_temp"]), bins=60, color="#DD8452", edgecolor="white")
        axes[1].set_title("Distribusi Log(1 + Suhu Kritis)")
        axes[1].set_xlabel("log(1 + critical_temp)")
        axes[1].set_ylabel("Frekuensi")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(f"""
        **Statistik:**
        - Minimum: {df['critical_temp'].min():.3f} K
        - Maksimum: {df['critical_temp'].max():.1f} K  
        - Rata-rata: {df['critical_temp'].mean():.2f} K
        - Median: {df['critical_temp'].median():.2f} K
        - Std. Dev: {df['critical_temp'].std():.2f} K
        """)

    with tab2:
        st.subheader("Top 15 Fitur dengan Korelasi Tertinggi terhadap Suhu Kritis")
        corr = df.corr()["critical_temp"].drop("critical_temp").abs().sort_values(ascending=False)
        top_corr = corr.head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ecc71" if df.corr()["critical_temp"][c] > 0 else "#e74c3c" for c in top_corr.index]
        top_corr.plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel("Korelasi Absolut dengan critical_temp")
        ax.set_title("Top 15 Fitur Paling Berkorelasi")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("🟢 Hijau = korelasi positif | 🔴 Merah = korelasi negatif")

    with tab3:
        st.subheader("Scatter Plot Fitur vs Suhu Kritis")
        feature_scatter = st.selectbox(
            "Pilih fitur:", 
            [c for c in df.columns if c != "critical_temp"],
            index=list(df.columns).index("wtd_mean_atomic_mass") if "wtd_mean_atomic_mass" in df.columns else 0
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(df[feature_scatter], df["critical_temp"], alpha=0.15, s=5, color="#4C72B0")
        ax.set_xlabel(feature_scatter)
        ax.set_ylabel("critical_temp (K)")
        ax.set_title(f"{feature_scatter} vs critical_temp")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.subheader("Distribusi Suhu Kritis berdasarkan Jumlah Elemen")
        fig, ax = plt.subplots(figsize=(12, 5))
        df_elem = df[df["number_of_elements"] <= 10].copy()
        df_elem["number_of_elements"] = df_elem["number_of_elements"].astype(int)
        df_elem.boxplot(column="critical_temp", by="number_of_elements", ax=ax,
                        patch_artist=True)
        ax.set_xlabel("Jumlah Elemen")
        ax.set_ylabel("Suhu Kritis (K)")
        ax.set_title("Distribusi Suhu Kritis per Jumlah Elemen")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MACHINE LEARNING
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "🤖 Prediksi ML":
    st.title("🤖 Model Machine Learning")
    st.markdown("Latih dan bandingkan model regresi untuk memprediksi **suhu kritis superkonduktor**.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("Ukuran Data Uji (%)", 10, 40, 20) / 100
    with col2:
        model_choice = st.selectbox("Pilih Model", [
            "Random Forest",
            "Gradient Boosting",
            "Ridge Regression",
            "Linear Regression"
        ])
    with col3:
        n_features = st.slider("Jumlah Top Fitur yang digunakan", 10, 81, 30)

    run = st.button("🚀 Latih Model", type="primary", use_container_width=True)

    if run:
        with st.spinner("Melatih model..."):
            # Pilih top N fitur berdasarkan korelasi
            corr = df.corr()["critical_temp"].drop("critical_temp").abs()
            top_features = corr.sort_values(ascending=False).head(n_features).index.tolist()

            X = df[top_features]
            y = df["critical_temp"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # Model
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "Ridge Regression": Ridge(alpha=1.0),
                "Linear Regression": LinearRegression(),
            }
            model = models[model_choice]

            if model_choice in ["Ridge Regression", "Linear Regression"]:
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

        # ── Metrics ──────────────────────────────────────────────────────────
        st.markdown("### 📊 Hasil Evaluasi Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r2:.4f}", help="Semakin mendekati 1 semakin baik")
        c2.metric("RMSE", f"{rmse:.3f} K", help="Root Mean Squared Error")
        c3.metric("MAE", f"{mae:.3f} K", help="Mean Absolute Error")

        # ── Scatter Actual vs Predicted ───────────────────────────────────────
        st.markdown("### 🎯 Actual vs Predicted")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].scatter(y_test, y_pred, alpha=0.3, s=8, color="#4C72B0")
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal")
        axes[0].set_xlabel("Suhu Kritis Aktual (K)")
        axes[0].set_ylabel("Suhu Kritis Prediksi (K)")
        axes[0].set_title(f"Actual vs Predicted — {model_choice}")
        axes[0].legend()

        residuals = y_test - y_pred
        axes[1].hist(residuals, bins=60, color="#DD8452", edgecolor="white")
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_title("Distribusi Residual (Error)")
        axes[1].set_xlabel("Residual (K)")
        axes[1].set_ylabel("Frekuensi")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Feature Importance (RF/GB only) ──────────────────────────────────
        if model_choice in ["Random Forest", "Gradient Boosting"]:
            st.markdown("### 🔑 Feature Importance")
            importance = pd.Series(model.feature_importances_, index=top_features)
            importance = importance.sort_values(ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            importance.plot(kind="bar", ax=ax, color="#4C72B0")
            ax.set_ylabel("Importance")
            ax.set_title("Top 15 Feature Importance")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Store model in session ────────────────────────────────────────────
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["top_features"] = top_features
        st.session_state["model_choice"] = model_choice
        st.success(f"✅ Model **{model_choice}** berhasil dilatih dan disimpan!")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDIKSI MANUAL
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "🔍 Prediksi Manual":
    st.title("🔍 Prediksi Manual")
    st.markdown("""
    Masukkan nilai fitur material superkonduktor secara manual untuk mendapatkan
    prediksi **suhu kritis** menggunakan model yang sudah dilatih.
    """)

    if "model" not in st.session_state:
        st.warning("⚠️ Silakan latih model terlebih dahulu di halaman **🤖 Prediksi ML**.")
    else:
        features = st.session_state["top_features"]
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        model_choice = st.session_state["model_choice"]

        st.info(f"Model aktif: **{model_choice}** | Menggunakan **{len(features)} fitur**")

        st.markdown("### ✏️ Input Nilai Fitur")
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            med = float(df[feat].median())
            mn = float(df[feat].min())
            mx = float(df[feat].max())
            with cols[i % 3]:
                input_vals[feat] = st.number_input(
                    feat, value=round(med, 4),
                    min_value=round(mn - abs(mn)*0.5, 4),
                    max_value=round(mx + abs(mx)*0.5, 4),
                    format="%.4f"
                )

        if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_vals])

            if model_choice in ["Ridge Regression", "Linear Regression"]:
                input_sc = scaler.transform(input_df)
                pred = model.predict(input_sc)[0]
            else:
                pred = model.predict(input_df)[0]

            pred = max(0, pred)  # suhu tidak boleh negatif

            st.markdown("---")
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 16px;
                padding: 32px;
                text-align: center;
                border: 2px solid #4C72B0;
                margin-top: 16px;
            '>
                <h2 style='color:#aad4f5; margin:0;'>Prediksi Suhu Kritis</h2>
                <h1 style='color:#ffffff; font-size:3.5em; margin:8px 0;'>{pred:.2f} K</h1>
                <p style='color:#8ab4d4; margin:0; font-size:1.1em;'>
                    ({pred - 273.15:.2f} °C &nbsp;|&nbsp; {(pred - 273.15)*9/5 + 32:.2f} °F)
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            ctx_col1, ctx_col2 = st.columns(2)
            with ctx_col1:
                avg = df["critical_temp"].mean()
                diff = pred - avg
                sign = "+" if diff >= 0 else ""
                st.metric("vs Rata-rata Dataset", f"{sign}{diff:.2f} K",
                          delta=f"{sign}{diff:.2f} K")
            with ctx_col2:
                pct = (df["critical_temp"] <= pred).mean() * 100
                st.metric("Persentil dalam Dataset", f"{pct:.1f}%",
                          help="Berapa persen material dalam dataset memiliki suhu kritis ≤ prediksi ini")

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📌 Dataset: Superconduct Train | Aplikasi analisis suhu kritis material superkonduktor")
