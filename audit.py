import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Download the public dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
           "hours-per-week", "native-country", "income"]

print("📥 Downloading demographic data for auditing...")
df = pd.read_csv(url, names=columns, skipinitialspace=True)

# 2. THE AUDIT: Checking the balance of the data
print("\n--- Gender Distribution ---")
print(df['sex'].value_counts(normalize=True))

# 3. Create a Visualization
# This helps us see if certain groups are unfairly represented in high-income brackets
plt.figure(figsize=(10,6))
sns.countplot(x='income', hue='sex', data=df)
plt.title("Audit Report: Income Distribution by Gender")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Prepare the data (AI only understands numbers, not words)
print("\n🧠 Training the AI model...")
le = LabelEncoder()
df_numbers = df.apply(le.fit_transform)

X = df_numbers.drop('income', axis=1) # The "Questions" (Age, Gender, etc.)
y = df_numbers['income']              # The "Answer" (Income)

# 2. Split the data into "Study Material" and a "Final Exam"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create the AI and let it learn
ai_model = RandomForestClassifier(n_estimators=100)
ai_model.fit(X_train, y_train)

# 4. THE AUDIT: Test accuracy for Men vs Women
test_data = X_test.copy()
test_data['actual_income'] = y_test
test_data['ai_prediction'] = ai_model.predict(X_test)

# Check Accuracy for Women (sex 0)
female_data = test_results = test_data[test_data['sex'] == 0]
f_acc = accuracy_score(female_data['actual_income'], female_data['ai_prediction'])

# Check Accuracy for Men (sex 1)
male_data = test_data[test_data['sex'] == 1]
m_acc = accuracy_score(male_data['actual_income'], male_data['ai_prediction'])

print(f"\n✅ Audit Complete!")
print(f"Accuracy for Women: {f_acc:.2%}")
print(f"Accuracy for Men: {m_acc:.2%}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Prepare the data (Translation)
print("\n🧠 Training the AI model...")
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

X = df_encoded.drop('income', axis=1) # The Questions
y = df_encoded['income']              # The Answer

# 2. Split into Study (80%) and Exam (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. THE AUDIT: Compare accuracy between genders
test_results = X_test.copy()
test_results['actual'] = y_test
test_results['predicted'] = model.predict(X_test)

# Calculate accuracy for group 0 (Female) and group 1 (Male)
female_acc = accuracy_score(test_results[test_results['sex'] == 0]['actual'], 
                            test_results[test_results['sex'] == 0]['predicted'])

male_acc = accuracy_score(test_results[test_results['sex'] == 1]['actual'], 
                          test_results[test_results['sex'] == 1]['predicted'])

print(f"\n✅ AUDIT REPORT:")
print(f"Accuracy for Female candidates: {female_acc:.2%}")
print(f"Accuracy for Male candidates: {male_acc:.2%}")
import shap

# 1. Create the Explainer
# We use a small sample (100 rows) to make it run fast on your Mac
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:100])

# 2. Create the Summary Plot
print("\n📊 Generating SHAP Explanation Chart...")
plt.figure()
shap.summary_plot(shap_values[1], X_test.iloc[:100], plot_type="bar")
# --- FINAL ANALYTICS REPORT ---
print("\n" + "="*30)
print("📊 FINAL AUDIT SUMMARY")
print("="*30)
print(f"Total Records Analyzed: {len(df)}")
print(f"Bias Detected: Yes")
print(f"Impact: AI is {abs(female_acc - male_acc):.2%} more accurate for one group.")
print("Recommendation: Re-balance dataset before deployment.")
print("="*30)