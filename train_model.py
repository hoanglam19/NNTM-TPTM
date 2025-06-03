
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/fall_data.csv")
X = df.drop("label", axis=1)
y = df["label"].map({"normal": 0, "fall": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

joblib.dump(clf, "fall_model.pkl")
print("Accuracy:", clf.score(X_test, y_test))
