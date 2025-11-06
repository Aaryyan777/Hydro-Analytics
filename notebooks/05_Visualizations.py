
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

# Load the preprocessed data
df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\drinking-water-quality-preprocessed.csv")

# Load the trained model
model = joblib.load(r"C:\Users\DELL\Desktop\predict\app\water_quality_model_xgb.joblib")

# Create the static directory if it doesn't exist
import os
static_dir = r"C:\Users\DELL\Desktop\predict\app\static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Set the style
sns.set_style("whitegrid")
palette = "viridis"

# 1. Distribution of Numerical Features
plt.figure(figsize=(12, 7))
sns.histplot(df['Residual Free Chlorine (mg/L)'], kde=True, color=sns.color_palette(palette)[0])
plt.title('Distribution of Residual Free Chlorine', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'distribution_chlorine.png'))
plt.close()

plt.figure(figsize=(12, 7))
sns.histplot(df['Turbidity (NTU)'], kde=True, color=sns.color_palette(palette)[1])
plt.title('Distribution of Turbidity', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'distribution_turbidity.png'))
plt.close()

# 2. Target Variable Distribution
plt.figure(figsize=(10, 7))
sns.countplot(x='Deterioration', data=df, palette=palette)
plt.title('Distribution of Target Variable (Deterioration)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'target_distribution.png'))
plt.close()

# 3. Feature Importance
encoder = joblib.load(r"C:\Users\DELL\Desktop\predict\app\encoder.joblib")
original_cols = df.drop(columns=['Deterioration']).columns.tolist()
cat_features = encoder.get_feature_names_out(['Sample class'])
num_features = [col for col in original_cols if col != 'Sample class']
feature_names = num_features + list(cat_features)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(14, 9))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette=palette)
plt.title('Feature Importance', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'feature_importance.png'))
plt.close()

# 4. Correlation Matrix
plt.figure(figsize=(12, 9))
sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'correlation_heatmap.png'))
plt.close()

# 5. Deterioration over Time
original_df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\drinking-water-quality-distribution-monitoring-data.csv", low_memory=False)

def is_deteriorated(row):
    coliform = str(row['Coliform (Quanti-Tray) (MPN /100mL)']).strip()
    ecoli = str(row['E.coli(Quanti-Tray) (MPN/100mL)']).strip()
    if coliform != '<1' or ecoli != '<1':
        return 1
    return 0

original_df['Deterioration'] = original_df.apply(is_deteriorated, axis=1)
original_df['Sample_DateTime'] = pd.to_datetime(original_df['Sample Date'], errors='coerce')
deterioration_by_month = original_df.groupby(original_df['Sample_DateTime'].dt.to_period('M'))['Deterioration'].sum()

plt.figure(figsize=(14, 7))
deterioration_by_month.plot(kind='line', color=sns.color_palette(palette)[3])
plt.title('Deterioration Events Over Time', fontsize=16)
plt.ylabel('Number of Deterioration Events')
plt.xlabel('Month')
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'deterioration_over_time.png'))
plt.close()

# 6. Model Evaluation Plots
X = df.drop(columns=['Deterioration'])
y = df['Deterioration']
X_encoded = encoder.transform(X[['Sample class']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Sample class']))
X = pd.concat([X.drop(columns=['Sample class']), X_encoded_df], axis=1)

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
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'roc_auc_curve.png'))
plt.close()

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap=palette)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'))
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, y_pred_proba)

plt.figure(figsize=(10, 7))
plt.plot(recall, precision, color=sns.color_palette(palette)[4], lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'precision_recall_curve.png'))
plt.close()

print("Visualizations created and saved to app/static directory.")
