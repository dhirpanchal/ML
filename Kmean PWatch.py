import pandas as pd
dataset=pd.read_csv("Mall_Customers_BP.csv")
x=dataset.iloc[:,2:5].values
print(x)

from sklearn.cluster import KMeans
kmean=KMeans(n_clusters=5,init="k-means++",random_state=4)
kmean.fit(x)
print(kmean.labels_)

wcss=[]
for k in range(1,15):
    kmean=KMeans(n_clusters=k,init="k-means++",random_state=4)
    kmean.fit(x)
    wcss.append(kmean.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,15),wcss)
plt.title("Elbow method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()