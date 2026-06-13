import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# 1. Load and Preprocess Data
print("Loading heart.csv dataset...")
data = pd.read_csv("heart.csv")

# Handle missing values by replacing with column mean
data.fillna(data.mean(), inplace=True)

# Define the base 5 features entered by the user
base_features = ["age", "sex", "cp", "trestbps", "chol"]
X_base = data[base_features]
y = data["target"]

print("--- Training Base Models (SVM, KNN, Gradient Boosting) ---")
# Scale features
scaler_base = StandardScaler()
X_scaled_base = scaler_base.fit_transform(X_base)

# Balance dataset using SMOTE
smote_base = SMOTE(sampling_strategy='auto', random_state=42)
X_res_base, y_res_base = smote_base.fit_resample(X_scaled_base, y)

# Train-Test Split (80% Train, 20% Test)
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_res_base, y_res_base, test_size=0.2, random_state=42
)

# 1a. SVM (Linear Kernel)
print("Training Linear SVM...")
svm_model = SVC(kernel='linear', probability=True, random_state=42, max_iter=2000)
svm_model.fit(X_train_base, y_train_base)
y_pred_svm = svm_model.predict(X_test_base)

metrics = {}
metrics['svm'] = {
    'accuracy': float(accuracy_score(y_test_base, y_pred_svm)),
    'precision': float(precision_score(y_test_base, y_pred_svm)),
    'recall': float(recall_score(y_test_base, y_pred_svm)),
    'f1': float(f1_score(y_test_base, y_pred_svm))
}

# 1b. KNN
print("Training KNN (n_neighbors=5)...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_base, y_train_base)
y_pred_knn = knn_model.predict(X_test_base)

metrics['knn'] = {
    'accuracy': float(accuracy_score(y_test_base, y_pred_knn)),
    'precision': float(precision_score(y_test_base, y_pred_knn)),
    'recall': float(recall_score(y_test_base, y_pred_knn)),
    'f1': float(f1_score(y_test_base, y_pred_knn))
}

# 1c. Gradient Boosting
print("Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_base, y_train_base)
y_pred_gb = gb_model.predict(X_test_base)

metrics['gb'] = {
    'accuracy': float(accuracy_score(y_test_base, y_pred_gb)),
    'precision': float(precision_score(y_test_base, y_pred_gb)),
    'recall': float(recall_score(y_test_base, y_pred_gb)),
    'f1': float(f1_score(y_test_base, y_pred_gb))
}

# Save base models and scaler
joblib.dump(svm_model, "models/svm_model.joblib")
joblib.dump(scaler_base, "models/svm_scaler.joblib")
joblib.dump(knn_model, "models/knn_model.joblib")
joblib.dump(scaler_base, "models/knn_scaler.joblib")
joblib.dump(gb_model, "models/gb_model.joblib")
joblib.dump(scaler_base, "models/gb_scaler.joblib")

print("Base models saved successfully.")


print("\n--- Training SVM with Jellyfish Optimization (JFO) ---")
# 2. SVM + JFO (Option A: Selected 4 features out of the 5 user inputs)
# Fit Chi-Square Selector
chi2_selector = SelectKBest(chi2, k=4)
X_selected = chi2_selector.fit_transform(X_base, y)
jfo_features = X_base.columns[chi2_selector.get_support()].tolist()
print(f"Chi-Square Selected Features for JFO: {jfo_features}")

# Scale selected features
scaler_jfo = StandardScaler()
X_scaled_jfo = scaler_jfo.fit_transform(X_selected)

# Balance dataset using SMOTE
smote_jfo = SMOTE(sampling_strategy='auto', random_state=42)
X_res_jfo, y_res_jfo = smote_jfo.fit_resample(X_scaled_jfo, y)

# Train-Test Split
X_train_jfo, X_test_jfo, y_train_jfo, y_test_jfo = train_test_split(
    X_res_jfo, y_res_jfo, test_size=0.2, random_state=42
)

# Objective Function for Jellyfish Optimization
def objective_function(params):
    C, gamma = params
    # C must be > 0 and gamma > 0
    C = max(0.1, C)
    gamma = max(0.0001, gamma)
    model = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=42, max_iter=2000)
    scores = cross_val_score(model, X_train_jfo, y_train_jfo, cv=5, scoring='accuracy')
    return np.mean(scores)

# Jellyfish Optimization Algorithm Implementation
class JellyfishOptimization:
    def __init__(self, objective_function, bounds, population_size, iterations, alpha=3, beta=0.1):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.population = None
        self.best_solution = None
        self.best_score = float("-inf")

    def initialize_population(self):
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, len(self.bounds))
        )

    def evaluate_population(self):
        fitness = np.array([self.objective_function(ind) for ind in self.population])
        best_index = np.argmax(fitness)
        if fitness[best_index] > self.best_score:
            self.best_score = fitness[best_index]
            self.best_solution = self.population[best_index]
        return fitness

    def move_jellyfish(self, fitness):
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            if np.random.rand() < 0.5:
                # Move inside ocean current (attracted/repelled by others)
                r1, r2 = np.random.choice(self.population_size, 2, replace=False)
                direction = self.population[r1] - self.population[r2]
                new_position = self.population[i] + self.alpha * np.random.rand() * direction
            else:
                # Move within jellyfish swarm (towards average position)
                jellyfish_mean = np.mean(self.population, axis=0)
                direction = jellyfish_mean - self.population[i]
                new_position = self.population[i] + self.beta * np.random.rand() * direction
            new_population[i] = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        return new_population

    def optimize(self):
        self.initialize_population()
        for it in range(self.iterations):
            fitness = self.evaluate_population()
            self.population = self.move_jellyfish(fitness)
            print(f"JFO Iteration {it+1}/{self.iterations} - Best Score: {self.best_score:.4f}")
        return self.best_solution

# Define bounds for SVC (C: [0.1, 1000], gamma: [0.001, 1])
bounds = [(0.1, 1000), (0.001, 1)]

print("Optimizing SVM parameters with JFO (pop=10, iter=20)...")
jellyfish = JellyfishOptimization(objective_function, bounds, population_size=10, iterations=20)
best_params = jellyfish.optimize()
print(f"Optimal Parameters Found - C: {best_params[0]:.4f}, gamma: {best_params[1]:.4f}")

# Train final SVM model using optimized parameters
jfo_model = SVC(C=best_params[0], gamma=best_params[1], kernel='rbf', probability=True, random_state=42, max_iter=2000)
jfo_model.fit(X_train_jfo, y_train_jfo)
y_pred_jfo = jfo_model.predict(X_test_jfo)

metrics['jfo'] = {
    'accuracy': float(accuracy_score(y_test_jfo, y_pred_jfo)),
    'precision': float(precision_score(y_test_jfo, y_pred_jfo)),
    'recall': float(recall_score(y_test_jfo, y_pred_jfo)),
    'f1': float(f1_score(y_test_jfo, y_pred_jfo))
}

# Save JFO model, scaler, and features
joblib.dump(jfo_model, "models/jfo_model.joblib")
joblib.dump(scaler_jfo, "models/jfo_scaler.joblib")
joblib.dump(jfo_features, "models/jfo_features.joblib")

print("JFO model saved successfully.")

# Save evaluation metrics
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n--- Model Performance Summary ---")
for model_name, values in metrics.items():
    print(f"{model_name.upper()}:")
    print(f"  Accuracy:  {values['accuracy']:.4f}")
    print(f"  Precision: {values['precision']:.4f}")
    print(f"  Recall:    {values['recall']:.4f}")
    print(f"  F1 Score:  {values['f1']:.4f}")

print("\nAll models trained and saved successfully.")
