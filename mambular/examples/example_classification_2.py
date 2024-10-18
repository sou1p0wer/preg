import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/vepfs-sha/xiezixun/high_risk_pregnant/mambular')

from mambular.models import MambularClassifier

# Set random seed for reproducibility
np.random.seed(0)

# Number of samples
n_samples = 100
n_features = 5

# Generate random features
X = np.random.randn(n_samples, n_features)
coefficients1 = np.random.randn(n_features)
coefficients2 = np.random.randn(n_features)

# Generate target variable
y1 = np.dot(X, coefficients1) + np.random.randn(n_samples)
# Convert y to multiclass by categorizing into quartiles
y1 = pd.qcut(y1, 3, labels=False)

# Generate target variable
y2 = np.dot(X, coefficients2) + np.random.randn(n_samples)
# Convert y to multiclass by categorizing into quartiles
y2 = pd.qcut(y2, 3, labels=False)


# Create a DataFrame to store the data
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
data["target1"] = y1
data["target2"] = y2

# Split data into features and target variable
X = data.drop(columns=["target1","target2"])
y1 = data["target1"].values
y2 = data["target2"].values
y1 = y1.reshape((-1,1))
y2 = y2.reshape((-1,1))
y = np.concatenate((y1,y2),axis=1)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Instantiate the classifier
classifier = MambularClassifier()

# Fit the model on training data
classifier.fit(X_train, y_train, max_epochs=10)

# print(classifier.evaluate(X_test, y_test))
