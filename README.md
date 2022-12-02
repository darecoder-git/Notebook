# Notebooks
# Import necessary modules
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import numpy as np

# Define the number of clusters for K-means
n_clusters = 5

# Load the images
glass_images = load_images('glass')
leather_images = load_images('leather')
paper_images = load_images('paper')
plastic_images = load_images('plastic')
metal_images = load_images('metal')

# Concatenate the images into a single array
images = np.concatenate([glass_images, leather_images, paper_images, plastic_images, metal_images])

# Extract SIFT features from the images
sift = cv2.xfeatures2d.SIFT_create()
features = []
for image in images:
    _, des = sift.detectAndCompute(image, None)
    features.append(des)

# Use K-means to group the features into clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)

# Create labels for each image based on the cluster it belongs to
glass_labels = [0] * len(glass_images)
leather_labels = [1] * len(leather_images)
paper_labels = [2] * len(paper_images)
plastic_labels = [3] * len(plastic_images)
metal_labels = [4] * len(metal_images)
labels = np.concatenate([glass_labels, leather_labels, paper_labels, plastic_labels, metal_labels])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

# Train a SVM classifier on the training data
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

# Evaluate the classifier on the test data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
