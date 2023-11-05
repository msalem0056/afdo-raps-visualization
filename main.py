import pandas as pd
from sklearn.datasets import load_breast_cancer
import streamlit as st
import streamlit.components.v1 as components
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import plotly.express as px




def app(title=None)-> None:
    """Creates the streamlit app

    Args:
        title (string, optional): The App name. Defaults to None.
    """
    st.title(title)
    col1, col2, col3 = st.columns([0.1,.015,.1])
    col1.markdown("### Developer: Mike Salem")
    col2.image("LI-In-Bug.jpg", width=48)
    col3.markdown("### [LinkedIn](https://www.linkedin.com/in/mike-salem/)")    
    st.write("Description: The following app was created to showcase exploratory data analysis as a component of model transparency. Both UMAP and PCA are showcased below for the breast cancer dataset.")

    dataframe = load_breast_cancer(as_frame= True)
    dataframe = dataframe.data
    df_complete = dataframe.copy()
    df_complete['target'] = load_breast_cancer(as_frame= True).target.astype(str).replace("0", "malignant").replace("1","benign")

    st.write("Data Sample")
    st.dataframe(df_complete.head(30))

    tab1, tab2 = st.tabs(["PCA", "UMAP"])

    with tab1:
        df_diag= load_breast_cancer(as_frame=True).target
        df_features = dataframe
        standardized = StandardScaler()
        standardized.fit(df_features)
        scaled_data = standardized.transform(df_features)
        pca = PCA(n_components=3)
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)
        fig = px.scatter_3d(x_pca, x= x_pca[:,0], y = x_pca[:,1], z = x_pca[:,2], size= [0.2] * len(x_pca[:,0]), color=load_breast_cancer(as_frame= True).target.astype(str).replace("0", "malignant").replace("1","benign"), title="First 3 Components of Breast Cancer Dataset")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        df_pc = pd.DataFrame(pca.components_, columns = df_features.columns)
        fig = plt.figure(figsize=(15, 8))
        sns.heatmap(df_pc, cmap='viridis')
        plt.title('Principal Components correlation with the features')
        plt.xlabel('Features')
        plt.ylabel('Principal Components')
        st.pyplot(fig)
        st.markdown("Source: https://www.kaggle.com/code/jahirmorenoa/pca-to-the-breast-cancer-data-set")

    with tab2:
        reducer = umap.UMAP()
        reducer = umap.UMAP(random_state=42)
        reducer.fit(dataframe)
        embedding = reducer.fit_transform(dataframe)
        fig = px.scatter(embedding, x= embedding[:,0], y = embedding[:,1], color=load_breast_cancer(as_frame= True).target.astype(str).replace("0", "malignant").replace("1","benign"), title="UMAP Visualization of Breast Cancer Dataset")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


if __name__ == "__main__":
    app(title='Exploratory Data Analysis by Visualization')