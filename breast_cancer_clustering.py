import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['OMP_NUM_THREADS'] = '3'

df = pd.read_csv(r'C:\Users\Wiktoria\Desktop\Python Basics\Projekt3_Klasteryzacja\data.csv')

df.shape
df.info()
df.isna()

pd.set_option('display.max_columns', None)

desc = df.describe()
desc

#target
df['diagnosis'].value_counts()
value_counts = df['diagnosis'].value_counts()

colors = ['green', 'salmon']

plt.figure()
value_counts.plot(kind='bar', color=colors)
plt.title('Value Counts By Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Counts')

df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

#high correlations between _mean and _se and _worst values
#no information on how the standard error values _se for individual
#characteristics were calculated
df = df.drop(columns=['id','radius_se', 'radius_worst', 'texture_se', 'texture_worst',
                      'perimeter_se', 'perimeter_worst', 'area_se', 'area_worst',
                      'smoothness_se', 'smoothness_worst', 'compactness_se',
                      'compactness_worst', 'concavity_se', 'concavity_worst',
                      'concave_points_se', 'concave_points_worst', 'symmetry_se',
                      'symmetry_worst', 'fractal_dimension_se',
                      'fractal_dimension_worst'])

#sns.pairplot(df, hue='diagnosis')

#correlation
corr = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True)

df = df.drop(columns=['area_mean', 'perimeter_mean', 'concavity_mean'])

#KMeans clustering
#reject of the target
X = df.drop(columns=['diagnosis'])

X

Y = df['diagnosis']

#scaling
scaler = MinMaxScaler()

X_clusters = scaler.fit_transform(X)

X_clusters

#elbow method
sum_sqr_distances = []

clusters = 20
for no in range(1, clusters+1):
    K_mean = KMeans(n_clusters = no, random_state=42, n_init=10)
    K_mean.fit(X_clusters)
    sum_sqr_distances.append(K_mean.inertia_)
    
sum_sqr_distances

plt.plot(list(range(1, clusters+1)), sum_sqr_distances, '-o')
plt.title('Elbow method')
plt.xlabel('Cluster')
plt.ylabel('Sum of squared distances')

#4 clusters
K_mean = KMeans(n_clusters=4, random_state=42, n_init=10)
K_mean.fit(X_clusters)


#clusters analysis
labels = pd.DataFrame(K_mean.labels_)
centroids = pd.DataFrame(K_mean.cluster_centers_)

labeledData = pd.concat((df, labels), axis=1)
labeledData = labeledData.rename({0: 'labels'}, axis=1)

labeledData


#content of clusters
labeledData['labels'].value_counts()

#target in clusters (na odwrót niż w jupyter)
for no in range(4):
    print(f'Cluster: {no}')
    print(labeledData[labeledData['labels'] == no]['diagnosis'].value_counts())

data_for_plot = []

#collecting data on the number of 'diagnosis' in each cluster
for no in range(4):
    cluster_data = labeledData[labeledData['labels'] == no]['diagnosis'].value_counts().reset_index()
    cluster_data.columns = ['diagnosis', 'count']
    cluster_data['cluster'] = no
    data_for_plot.append(cluster_data)

plot_data = pd.concat(data_for_plot)

plt.figure()
sns.barplot(x='cluster', y='count', hue='diagnosis', data=plot_data, palette='magma')
plt.title('Count of Diagnosis in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Diagnosis')
plt.show()


#add statistics
cluster_desc = []
for no in range(4):
    print(f'Cluster: {no}')  
    desc = labeledData[labeledData['labels'] == no].describe()
    cluster_desc.append(desc)
    print(desc)
    

#visualization
cols_ = labeledData.columns[1:-1]

cols_

param_ = 'mean' # std, ...

for col in cols_:
    plt.figure()
    values = [cluster_description[col][param_] for cluster_description in cluster_desc]
    plt.scatter(list(range(len(values))), values)
    plt.title(col + ' ' + param_)
    plt.xlabel('Cluster')
    plt.ylabel('Feature ' + param_)


df['diagnosis'].value_counts()