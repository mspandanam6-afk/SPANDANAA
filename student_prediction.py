import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("student-mat.csv")


data = data[['studytime', 'absences', 'G1', 'G2', 'G3']]


X = data[['studytime', 'absences', 'G1', 'G2']]
y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

print("Model trained successfully with Kaggle dataset!")

score = model.score(X_test, y_test)
print("Model Accuracy (R²):", round(score * 100, 2), "%")

print("\nEnter student details:")

studytime = int(input("Study Time (1-4): "))
absences = int(input("Number of Absences: "))
G1 = int(input("Previous Grade 1: "))
G2 = int(input("Previous Grade 2: "))

prediction = model.predict([[studytime, absences, G1, G2]])

final_score = prediction[0]

print("\nPredicted Final Score (G3):", round(final_score, 2))

if final_score >= 10:
    print("Result: PASS ")
else:
    print("Result: FAIL ")