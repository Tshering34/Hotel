import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import folium
from geopy.geocoders import Nominatim

# Application title
st.title("Project: Machine Learning for Bhutan Tourism")
st.title("Tourist Satisfaction Classification [Classification]")


# Load the dataset using the new caching command
# Use st.cache_data instead of st.cache
@st.cache_data 
def load_data():
    dataset = pd.read_csv('File_Point.csv')
    return dataset

# Loading and displaying the dataset
dataset = load_data()
st.subheader("Dataset preview")
st.dataframe(dataset)

# Extracting features for clustering
X = dataset.iloc[:, [0, 1]].values 

# KMeans clustering, where cluster = 2.
num_clusters = st.slider("2", min_value=2, max_value=10, value=2)
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Drawing clusters
st.subheader("K-Means Clustering Based on Frequently Made Visits by Tourists in Bhutan")
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(num_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i % len(colors)], label=f'Cluster {i + 1}') 

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='orange', label='Centroids') 
plt.title('Clusters of Visited Places by Tourists') 
plt.xlabel('Longitude')
plt.ylabel(') Latitude') 
plt.legend() 
st.pyplot(plt.gcf()) # Render plot in Streamlit 

###----------------------------- For MAP---------------------------------------------###

# Get the geolocation of Bhutan 
address = 'Bhutan' 
geolocator = Nominatim(user_agent="to_explorer") 
location = geolocator.geocode(address) 
latitude = location.latitude 
longitude = location.longitude 

# Display map
st.subheader("Places Visited by Tourists in Bhutan.")
map = folium.Map(location=[latitude, longitude], zoom_start=9, control_scale=True)

# Add points to the Bhutan map
for lat, long in zip(dataset['lat'], dataset['long']):
    folium.CircleMarker(
        [lat, long],
        radius=3,
        color='darkgreen',
        fill=True,
        fill_color='darkgreen',
        fill_opacity=0.7,
        parse_html=False
    ).add_to(map)

# Save the map to HTML and render it
map_html = "map.html"
map.save(map_html)

# Render the Folium map
st.components.v1.html(map._repr_html_(), height=500) 
