import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Expanded dataset
data = {
    "attendance": [60, 70, 80, 90, 50, 85, 95, 40, 75, 88],
    "marks":     [50, 65, 75, 90, 40, 80, 92, 35, 70, 85],
    "result":    [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]  # 1 = Pass, 0 = Fail
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["attendance", "marks"]]
y = df["result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model creation and training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# User input
att = int(input("\nEnter student attendance (%): "))
marks = int(input("Enter student marks: "))

# Prediction
prediction = model.predict([[att, marks]])
probability = model.predict_proba([[att, marks]])

# Output
if prediction[0] == 1:
    print("Result: PASS ✅")
else:
    print("Result: FAIL ❌")

print("Pass Probability:", round(probability[0][1] * 100, 2), "%")
