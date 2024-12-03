import streamlit as st
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import umap.umap_ as umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Set the page configuration
st.set_page_config(page_title="Réduction de dimension", layout="wide", page_icon="📊", initial_sidebar_state='expanded')

# Title and Introduction
st.title("📊 Réduction de dimension - Web App")
st.markdown("""
This app allows you to upload your data, visualize reduc dim at target and index columns.
""")

with st.sidebar:
    st.header("Upload and Select Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,sep=";")
        st.success("File successfully uploaded!")

if uploaded_file is not None:
    
    options = ["VisuData", "UMAP", "ACP"]
    selection = st.segmented_control(
        "Directions", options, selection_mode="single"
    )
    # Preprocess the dataset: Convert dates to numerical features and encode categorical variables
    for col in df.columns:
        if col=="index":
            df.index=df[col]
            df.drop(columns=col,inplace=True)
            continue
        if is_string_dtype(df[col]):
            try:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
            except Exception:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
    # This scales each column to have mean=0 and standard deviation=1
    SS=StandardScaler()
    # Apply scaling
    X=pd.DataFrame(SS.fit_transform(df), columns=df.columns)
    if selection == "VisuData":
        # Data Preview Section
        st.subheader("Data Preview")
        preview_rows = st.slider("How many rows to display?", 5, 100, 20)
        st.dataframe(df.head(preview_rows))
    if selection == "UMAP": 
        n_components = st.slider("n_components ?", 1, 50, 2)
        n_neighbors = st.slider("n_neighbors ?", 1, 50, 4)
        min_dist= st.slider("min_dist / 10 ?", 0, 9, 1)/10
        metric=st.selectbox(f"metric ? n_components {n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}", ["euclidean","manhattan","chebyshev","minkowski","canberra","braycurtis","haversine","mahalanobis","wminkowski","seuclidean","cosine","correlation","hamming","jaccard","dice","russellrao","kulsinski","rogerstanimoto","sokalmichener","sokalsneath","yule"])
        with st.spinner('Wait for load...'):
            umap_model_2d = umap.UMAP(n_components=n_components,n_neighbors=n_neighbors,min_dist=min_dist,metric=metric, random_state=42)
            X_reduced_2d=umap_model_2d.fit_transform(X)
        dfig=pd.DataFrame({"x":X_reduced_2d[:, 0],"y": X_reduced_2d[:, 1],"target":df.target})
        fig = px.scatter(dfig,x='x',y='y',color="target",hover_data=[df.index],width=1400,height=1000)
        event = st.plotly_chart(fig, key="iris", on_select="rerun",theme=None)
    if selection == "ACP": 
        n_components = st.slider("n_components ?", 1, 50, 2)
        with st.spinner('Wait for load...'):
            pca = PCA(n_components=n_components)
            X_reduced_2d=pca.fit_transform(X)
        dfig=pd.DataFrame({"x":X_reduced_2d[:, 0],"y": X_reduced_2d[:, 1],"target":df.target})
        fig = px.scatter(dfig,x='x',y='y',color="target",hover_data=[df.index],width=1400,height=1000)
        event = st.plotly_chart(fig, key="iris", on_select="rerun",theme=None)
        # PCA
        pca_var = PCA()
        pca_var.fit(X)
        fig=plt.figure(figsize=(10,5))
        xi = np.arange(1, 1+X.shape[1], step=1)
        yi = np.cumsum(pca_var.explained_variance_ratio_)
        plt.plot(xi, yi, marker='o', linestyle='--', color='b')

        # Aesthetics
        plt.ylim(0.0,1.1)
        plt.xlim(0,100)
        plt.xlabel('Number of Components')
        #plt.xticks(np.arange(1, 1+X.shape[1], step=1))
        plt.ylabel('Cumulative variance (%)')
        plt.title('Explained variance by each component')
        plt.axhline(y=1, color='r', linestyle='-')
        plt.gca().xaxis.grid(False)
        st.pyplot(fig)