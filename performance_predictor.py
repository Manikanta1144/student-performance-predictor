import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("student_data.csv")  # Make sure this CSV is in the same folder

X = df[["attendance", "marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

att = int(input("\nEnter student attendance (%): "))
marks = int(input("Enter student marks: "))

prediction = model.predict([[att, marks]])
probability = model.predict_proba([[att, marks]])

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")

print("Pass Probability:", round(probability[0][1] * 100, 2), "%")
