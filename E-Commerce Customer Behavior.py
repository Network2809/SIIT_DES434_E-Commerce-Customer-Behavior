# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 2: Load dataset
data = pd.read_csv("E-Commerce Customer Behavior Dataset.csv")

# Step 3: Explore data
print(data.info())
print(data.head())

# Step 4: Drop columns not needed for clustering
data_clean = data.drop([
    'Order_ID', 'Customer_ID', 'Date', 'Gender', 'City',
    'Product_Category', 'Payment_Method', 'Device_Type'
], axis=1)

# Step 5: Convert boolean column to numeric
data_clean['Is_Returning_Customer'] = data_clean['Is_Returning_Customer'].astype(int)

# Step 6: Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_clean)

# Step 7: Find best k (Elbow Method)
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal k')
plt.show()

# Step 8: Train model (pick k=4 as example)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)
data['Cluster'] = kmeans.labels_

# Step 9: Show numeric averages by cluster
numeric_cols = data.select_dtypes(include=[np.number])
cluster_summary = numeric_cols.groupby(data['Cluster']).mean()
print(cluster_summary)

# Step 10: Visualize Age vs Total_Amount
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Total_Amount', hue='Cluster', data=data, palette='viridis')
plt.title('Customer Segmentation by Age and Total Spending')
plt.show()

# Step 11: Save clustered dataset
data.to_csv("E-Commerce_Customer_Behavior_with_Clusters.csv", index=False)
print("Saved as: E-Commerce_Customer_Behavior_with_Clusters.csv")
