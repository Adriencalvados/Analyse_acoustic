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
from sklearn.cluster import DBSCAN

# Set the page configuration
st.set_page_config(page_title="Model", layout="wide", page_icon="📊", initial_sidebar_state='expanded')

# Title and Introduction
st.title("📊 Model - Web App")
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
    
    options = ["VisuData", "UMAP + MODEL", "ACP + MODEL"]
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
    if selection == "UMAP + MODEL": 
        n_components = st.slider("UMAP - n_components ?", 1, 50, 2)
        n_neighbors = st.slider("UMAP - n_neighbors ?", 1, 50, 4)
        min_dist= st.slider("UMAP - min_dist / 10 ?", 0, 9, 1)/10
        metric=st.selectbox(f"UMAP - metric ? n_components {n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}", ["euclidean","manhattan","chebyshev","minkowski","canberra","braycurtis","haversine","mahalanobis","wminkowski","seuclidean","cosine","correlation","hamming","jaccard","dice","russellrao","kulsinski","rogerstanimoto","sokalmichener","sokalsneath","yule"])
        eps = st.slider("DBSCAN - eps / 100 ?", 1, 99, 42)/100
        min_samples = st.slider("DBSCAN - min_samples ?", 1, 999, 240)
        with st.spinner('Wait for load...'):
            umap_model_2d = umap.UMAP(n_components=n_components,n_neighbors=n_neighbors,min_dist=min_dist,metric=metric, random_state=42)
            X_reduced_2d=umap_model_2d.fit_transform(X)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_reduced_2d)#pour la 2d eps=0.42, min_samples=240 10d
            y_pred = db.labels_

         # Number of clusters in labels, ignoring noise if present.
        st.write(f"Number of clusters = {len(set(y_pred)) - (1 if -1 in y_pred else 0)}")
        st.write(f"Number of anomalies = {list(y_pred).count(-1)}")
        dfig=pd.DataFrame({"x":X_reduced_2d[:, 0],"y": X_reduced_2d[:, 1],"prediction":y_pred})
        fig = px.scatter(dfig,x='x',y='y',color="prediction",hover_data={"index": df.index,"target":df.target},width=1400,height=1000,
                 template="plotly")
        event = st.plotly_chart(fig, key="iris", on_select="rerun",theme=None)
        # Download the HTML file
        st.download_button(
            label="Download as HTML",
            data=fig.to_html(full_html=True, include_plotlyjs='cdn'),
            file_name="data.html",
            mime="text/html"
        )
    if selection == "ACP + MODEL": 
        n_components = st.slider("n_components ?", 1, 50, 2)
        eps = st.slider("DBSCAN - eps / 100 ?", 1, 99, 42)/100
        min_samples = st.slider("DBSCAN - min_samples ?", 1, 999, 240)
        with st.spinner('Wait for load...'):
            pca = PCA(n_components=n_components)
            X_reduced_2d=pca.fit_transform(X)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_reduced_2d)#pour la 2d eps=0.42, min_samples=240 10d
            y_pred = db.labels_

         # Number of clusters in labels, ignoring noise if present.
        st.write(f"Number of clusters = {len(set(y_pred)) - (1 if -1 in y_pred else 0)}")
        st.write(f"Number of anomalies = {list(y_pred).count(-1)}")
        dfig=pd.DataFrame({"x":X_reduced_2d[:, 0],"y": X_reduced_2d[:, 1],"prediction":y_pred})
        fig = px.scatter(dfig,x='x',y='y',color="prediction",hover_data={"index": df.index,"target":df.target},width=1400,height=1000,
                 template="plotly")
        event = st.plotly_chart(fig, key="iris", on_select="rerun",theme=None)
        # Download the HTML file
        st.download_button(
            label="Download as HTML",
            data=fig.to_html(full_html=True, include_plotlyjs='cdn'),
            file_name="data.html",
            mime="text/html"
        )