
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("diabetes.csv")
print(data.head())

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, prediction)

print("Logistic Regression Accuracy:", logistic_accuracy * 100, "%")

# 2. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn_prediction = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_prediction)

print("KNN Accuracy:", knn_accuracy * 100, "%")

# 3. Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_prediction = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_prediction)

print("Decision Tree Accuracy:", tree_accuracy * 100, "%")

# 4. Random Forest
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)

forest_prediction = forest.predict(X_test)
forest_accuracy = accuracy_score(y_test, forest_prediction)

print("Random Forest Accuracy:", forest_accuracy * 100, "%")

# Predict for a new person
new_person = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 70,
    "BloodPressure": 110,
    "SkinThickness": 20,
    "Insulin": 1,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}])

new_prediction = model.predict(new_person)


if new_prediction[0] == 1:
    print("New Person: Diabetes Predicted")
else:
    print("New Person: Diabetes Not Predicted")

logistic_prediction = model.predict(new_person)[0]
knn_new_prediction = knn.predict(new_person)[0]
tree_new_prediction = tree.predict(new_person)[0]
forest_new_prediction = forest.predict(new_person)[0]

# Display predictions
print("\n--- Predictions for New Person ---")

print("Logistic Regression:", logistic_prediction)
print("KNN:", knn_new_prediction)
print("Decision Tree:", tree_new_prediction)
print("Random Forest:", forest_new_prediction)

# Store model names and accuracies
models = [
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Random Forest"
]

accuracies = [
    logistic_accuracy,
    knn_accuracy,
    tree_accuracy,
    forest_accuracy
]

# Display accuracy comparison as a bar graph
plt.figure(figsize=(9, 5))
plt.bar(models, [a * 100 for a in accuracies])

plt.title("Diabetes Prediction Accuracy")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.show()

