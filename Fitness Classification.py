import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Loading data
# I am using the absolute path to my project directory
path = r"C:\Users\mavs2\Documents\ML algorithms project\fitness_test.csv"
df = pd.read_csv(path)

# 2. Encode categorical columns
# I'm transforming 'sex' and 'exang' into numeric values so the MLP can process them
le = LabelEncoder()
categorical_cols = ['sex', 'exang'] 
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 3. Plotting the target class distribution
# Visualizing the balance between 'Fit' (1) and 'Not Fit' (0) individuals
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, hue='target', palette='viridis', legend=False)
plt.title('Distribution of Fitness Eligibility (1=Fit, 0=Not Fit)')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.show()

# 4. Feature selection
# I'm dropping the target to isolate predictor variables (X)
X = df.drop('target', axis=1)
y = df['target']

# I've identified the top 10 features based on their correlation with fitness
correlations = df.corr()['target'].abs().sort_values(ascending=False)
selected_features = correlations[1:11].index.tolist()
X = X[selected_features]

print("My Selected Features for Training:")
print(selected_features)

# 5. Feature standardisation
# Standardizing the data is essential for the MLP's weight optimization process
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split data into train and test sets
# I'm reserving 20% of the data to validate my model's performance
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Train Model
# I've set hidden_layer_sizes to (16, 8) and increased max_iter to 1200 
# to ensure the model converges and fully minimizes the loss.
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), 
                    activation='relu', 
                    solver='adam', 
                    max_iter=1200, 
                    random_state=42)

print("\nTraining the Multilayer Perceptron...")
mlp.fit(X_train, y_train)

# 8. Evaluate Model
# Generating the confusion matrix and classification report to check my work
y_pred = mlp.predict(X_test)

print(f"\nFinal Optimized Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting the Loss Curve to verify convergence
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_)
plt.title("My MLP Training Loss Curve (Optimized)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()