import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


df = pd.read_csv("Leads.csv")


drop_cols = [
    "Lead Quality", "Asymmetrique Activity Index", "Asymmetrique Profile Index", 
    "Asymmetrique Activity Score", "Asymmetrique Profile Score"
]
df_cleaned = df.drop(columns=drop_cols)


categorical_cols = ["Tags", "Lead Profile", "What matters most to you in choosing a course",
                    "What is your current occupation", "Country", "How did you hear about X Education",
                    "Specialization", "City"]
df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna("Unknown")

numerical_cols = ["TotalVisits", "Page Views Per Visit"]
df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(df_cleaned[numerical_cols].median())


df_cleaned = df_cleaned.fillna(df_cleaned.mode().iloc[0])


df_cleaned = df_cleaned.drop(columns=["Prospect ID", "Lead Number"])


label_encoders = {}
for col in df_cleaned.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

X = df_cleaned.drop(columns=["Converted"])
y = df_cleaned["Converted"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)


y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC-AUC Score": roc_auc_score(y_test, y_prob)
}

print(metrics)
