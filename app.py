import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.title("K-Means vs DBSCAN Clustering")

with open("models.pkl", "rb") as f:
    models = pickle.load(f)

X = models["X"]
kmeans_labels = models["kmeans_labels"]
dbscan_labels = models["dbscan_labels"]

c1, c2 = st.columns(2)

with c1:
    st.subheader("K-Means")
    st.write("Silhouette Score:", round(silhouette_score(X, kmeans_labels), 3))
    st.write("Clusters:", np.unique(kmeans_labels))

with c2:
    st.subheader("DBSCAN")
    if len(set(dbscan_labels)) > 1 and -1 not in set(dbscan_labels):
        st.write("Silhouette Score:", round(silhouette_score(X, dbscan_labels), 3))
    else:
        st.write("Silhouette Score: N/A")
    st.write("Clusters:", set(dbscan_labels))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "KMeans": kmeans_labels,
    "DBSCAN": dbscan_labels
})

c3, c4 = st.columns(2)

with c3:
    st.subheader("K-Means Visualization")
    st.scatter_chart(plot_df, x="PC1", y="PC2", color="KMeans")

with c4:
    st.subheader("DBSCAN Visualization")
    st.scatter_chart(plot_df, x="PC1", y="PC2", color="DBSCAN")
