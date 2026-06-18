import streamlit as st
import pandas as pd

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Dashboard Data Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# ==========================
# JUDUL
# ==========================
st.title("🎓 Dashboard Data Mahasiswa")
st.write("Analisis Data Mahasiswa Menggunakan Streamlit")

# ==========================
# UPLOAD FILE CSV
# ==========================
uploaded_file = st.file_uploader(
    "Upload File CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # Membaca CSV
    df = pd.read_csv(uploaded_file)

    st.success("✅ Data berhasil dimuat")

    # ==========================
    # INFORMASI DATA
    # ==========================
    st.header("📋 Ringkasan Data")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", len(df))
    col2.metric("Jumlah Kolom", len(df.columns))

    if "Provinsi" in df.columns:
        col3.metric("Jumlah Provinsi", df["Provinsi"].nunique())

    st.divider()

    # ==========================
    # TAMPILKAN DATA
    # ==========================
    st.header("📄 Data Mahasiswa")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # ==========================
    # FILTER PROVINSI
    # ==========================
    if "Provinsi" in df.columns:

        st.header("🔍 Filter Data")

        daftar_provinsi = ["Semua"] + sorted(
            df["Provinsi"].dropna().unique().tolist()
        )

        provinsi = st.selectbox(
            "Pilih Provinsi",
            daftar_provinsi
        )

        if provinsi == "Semua":
            data_filter = df
        else:
            data_filter = df[
                df["Provinsi"] == provinsi
            ]

        st.write(
            f"Jumlah data setelah filter : {len(data_filter)}"
        )

        st.dataframe(
            data_filter,
            use_container_width=True
        )

    st.divider()

    # ==========================
    # STATISTIK DESKRIPTIF
    # ==========================
    st.header("📊 Statistik Deskriptif")

    numeric_cols = df.select_dtypes(include="number")

    if len(numeric_cols.columns) > 0:
        st.dataframe(
            numeric_cols.describe(),
            use_container_width=True
        )

    st.divider()

    # ==========================
    # GRAFIK PROVINSI
    # ==========================
    if "Provinsi" in df.columns:

        st.header("📈 Distribusi Mahasiswa per Provinsi")

        provinsi_count = (
            df["Provinsi"]
            .value_counts()
            .sort_values(ascending=False)
        )

        st.bar_chart(provinsi_count)

    st.divider()

    # ==========================
    # GRAFIK JENIS KELAMIN
    # ==========================
    if "JK" in df.columns:

        st.header("👨‍🎓 Distribusi Jenis Kelamin")

        jk_count = (
            df["JK"]
            .value_counts()
        )

        st.bar_chart(jk_count)

    st.divider()

    # ==========================
    # GRAFIK IP SEMESTER 1
    # ==========================
    if "IP Sem 1" in df.columns:

        st.header("📉 Grafik IP Semester 1")

        st.line_chart(df["IP Sem 1"])

    # ==========================
    # GRAFIK IP SEMESTER 2
    # ==========================
    if "IP Sem 2" in df.columns:

        st.header("📉 Grafik IP Semester 2")

        st.line_chart(df["IP Sem 2"])

    st.divider()

    # ==========================
    # KORELASI NUMERIK
    # ==========================
    st.header("📌 Korelasi Data Numerik")

    if len(numeric_cols.columns) > 1:

        corr = numeric_cols.corr()

        st.dataframe(
            corr,
            use_container_width=True
        )

else:
    st.info("Silakan upload file CSV terlebih dahulu.")