
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import os

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv")

# Load the trained model
model = joblib.load(r"C:\Users\DELL\Desktop\predict\water_quality_model_brisbane_xgb.joblib")

# Create the static directory if it doesn't exist
static_dir = r"C:\Users\DELL\Desktop\predict\app\static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Set the style
sns.set_style("whitegrid")
palette = "plasma"

# 1. Distribution of Numerical Features
plt.figure(figsize=(12, 7))
sns.histplot(df['Turbidity'], kde=True, color=sns.color_palette(palette)[0])
plt.title('Distribution of Turbidity (Brisbane)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_distribution_turbidity.png'))
plt.close()

plt.figure(figsize=(12, 7))
sns.histplot(df['Dissolved Oxygen'], kde=True, color=sns.color_palette(palette)[1])
plt.title('Distribution of Dissolved Oxygen (Brisbane)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_distribution_do.png'))
plt.close()

plt.figure(figsize=(12, 7))
sns.histplot(df['pH'], kde=True, color=sns.color_palette(palette)[2])
plt.title('Distribution of pH (Brisbane)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_distribution_ph.png'))
plt.close()

# 2. Target Variable Distribution
plt.figure(figsize=(10, 7))
sns.countplot(x='Deterioration', data=df, palette=palette)
plt.title('Distribution of Target Variable (Deterioration - Brisbane)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_target_distribution.png'))
plt.close()

# 3. Feature Importance
feature_names = df.drop(columns=['Deterioration']).columns.tolist()
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(14, 9))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette=palette)
plt.title('Feature Importance (Brisbane)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_feature_importance.png'))
plt.close()

# 4. Correlation Matrix
plt.figure(figsize=(12, 9))
sns.heatmap(df.drop(columns=['Deterioration']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features (Brisbane)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_correlation_heatmap.png'))
plt.close()

# 5. Model Evaluation Plots
X = df.drop(columns=['Deterioration'])
y = df['Deterioration']

y_pred_proba = cross_val_predict(model, X, y, cv=3, method="predict_proba")[:, 1]
y_pred = cross_val_predict(model, X, y, cv=3)

# ROC-AUC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color=sns.color_palette(palette)[2], lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Brisbane)', fontsize=16)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_roc_auc_curve.png'))
plt.close()

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap=palette)
plt.title('Confusion Matrix (Brisbane)', fontsize=16)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_confusion_matrix.png'))
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(recall, precision, color=sns.color_palette(palette)[4], lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Brisbane)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'brisbane_precision_recall_curve.png'))
plt.close()

print("Brisbane visualizations created and saved to app/static directory.")
