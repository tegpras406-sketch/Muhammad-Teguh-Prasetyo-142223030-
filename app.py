import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Data Mahasiswa",page_icon="🎓",layout="wide")
st.title("🎓 Dashboard Data Mahasiswa")
uploaded_file=st.file_uploader("Upload File CSV",type=["csv"])
if uploaded_file:
    df=pd.read_csv(uploaded_file)
    menu=st.sidebar.selectbox("Menu",["Dataset","Informasi Data","Statistik","Visualisasi","Korelasi"])
    if menu=="Dataset":
        st.dataframe(df,use_container_width=True)
    elif menu=="Informasi Data":
        st.write(df.dtypes.astype(str))
        st.write(df.isnull().sum())
    elif menu=="Statistik":
        st.dataframe(df.describe())
    elif menu=="Visualisasi":
        nums=df.select_dtypes(include="number").columns
        if len(nums):
            col=st.selectbox("Kolom",nums)
            fig,ax=plt.subplots()
            df[col].hist(ax=ax)
            st.pyplot(fig)
    elif menu=="Korelasi":
        num=df.select_dtypes(include="number")
        if num.shape[1]>1:
            fig,ax=plt.subplots()
            im=ax.imshow(num.corr())
            ax.set_xticks(range(len(num.columns)))
            ax.set_xticklabels(num.columns,rotation=90)
            ax.set_yticks(range(len(num.columns)))
            ax.set_yticklabels(num.columns)
            plt.colorbar(im)
            st.pyplot(fig)
