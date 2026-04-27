"""
House Price Prediction — End-to-End ML Project
===============================================
Author  : Your Name
GitHub  : github.com/yourusername
Skills  : Python, scikit-learn, pandas, numpy, matplotlib

Run:
    pip install -r requirements.txt
    python house_price_prediction.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("   HOUSE PRICE PREDICTION")
print("=" * 60)

print("\n[1] Loading dataset...")
df = pd.read_csv('data/housing_data.csv')
print(f"    Shape: {df.shape}")
print(f"    Price range: ${df['Price'].min():,} — ${df['Price'].max():,}")
print(f"    Missing values: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[2] Basic statistics:")
print(df.describe().round(2).to_string())


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3] Feature engineering...")
df['RoomsPerOccupant']       = df['AveRooms'] / df['AveOccup']
df['BedsPerRoom']            = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']
print(f"    Created 3 new features. Total features: {df.shape[1] - 1}")


# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n[4] Splitting data...")
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler      = StandardScaler()
X_train_s   = scaler.fit_transform(X_train)
X_test_s    = scaler.transform(X_test)
print(f"    Train: {len(X_train)} | Test: {len(X_test)}")


# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[5] Training models...")
models = {
    'Linear Regression': (LinearRegression(),                                        True),
    'Random Forest':     (RandomForestRegressor(n_estimators=100, random_state=42),  False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
}

results = {}
for name, (model, use_scaled) in models.items():
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s  if use_scaled else X_test
    model.fit(Xtr, y_train)
    preds  = model.predict(Xte)
    mae    = mean_absolute_error(y_test, preds)
    rmse   = np.sqrt(mean_squared_error(y_test, preds))
    r2     = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'preds': preds, 'model': model}
    print(f"    {name:22s} | R²={r2:.3f} | MAE=${mae:,.0f} | RMSE=${rmse:,.0f}")


# ─────────────────────────────────────────────
# 6. BEST MODEL & FEATURE IMPORTANCE
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['R2'])
best      = results[best_name]
print(f"\n[6] Best model: {best_name} (R²={best['R2']:.3f})")

rf_model = results['Random Forest']['model']
fi       = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n    Feature Importances (Random Forest):")
for feat, imp in fi.items():
    bar = '█' * int(imp * 50)
    print(f"    {feat:30s} {bar} {imp:.3f}")


# ─────────────────────────────────────────────
# 7. SAVE CHARTS
# ─────────────────────────────────────────────
print("\n[7] Saving charts to outputs/...")

# Chart 1 — Model comparison
names  = list(results.keys())
colors = ['#4e79a7', '#59a14f', '#e15759']
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = [
    ([results[n]['R2'] for n in names],   'R² Score',  'R²'),
    ([results[n]['MAE']/1000 for n in names], 'MAE ($k)', 'MAE ($k)'),
    ([results[n]['RMSE']/1000 for n in names],'RMSE ($k)','RMSE ($k)'),
]
for ax, (vals, title, ylabel) in zip(axes, metrics):
    bars = ax.bar(names, vals, color=colors, width=0.5)
    ax.set_title(title); ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v*1.01,
                f'{v:.3f}' if ylabel == 'R²' else f'${v:.1f}k',
                ha='center', fontsize=10, fontweight='bold')
fig.suptitle('Model Performance Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2 — Actual vs Predicted
rf_preds = results['Random Forest']['preds']
fig2, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test/1000, rf_preds/1000, alpha=0.5, s=20, color='#59a14f')
mn = min(y_test.min(), rf_preds.min()) / 1000
mx = max(y_test.max(), rf_preds.max()) / 1000
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Price ($k)'); ax.set_ylabel('Predicted Price ($k)')
ax.set_title('Random Forest — Actual vs Predicted')
ax.text(0.05, 0.95, f'R² = {results["Random Forest"]["R2"]:.3f}',
        transform=ax.transAxes, fontsize=13, fontweight='bold', color='green', va='top')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3 — Feature Importance
fi_sorted = fi.sort_values()
fig3, ax = plt.subplots(figsize=(9, 6))
bar_colors = ['#4e79a7' if v > 0.05 else '#adb5bd' for v in fi_sorted.values]
ax.barh(fi_sorted.index, fi_sorted.values, color=bar_colors)
for i, (val, name) in enumerate(zip(fi_sorted.values, fi_sorted.index)):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=10)
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("    Saved: model_comparison.png")
print("    Saved: actual_vs_predicted.png")
print("    Saved: feature_importance.png")


# ─────────────────────────────────────────────
# 8. PREDICT ON NEW DATA
# ─────────────────────────────────────────────
def predict_price(med_inc, house_age, ave_rooms, ave_bedrms,
                  population, ave_occup, latitude, longitude):
    """Predict house price using the best trained model."""
    input_data = pd.DataFrame([{
        'MedInc':   med_inc,   'HouseAge': house_age,
        'AveRooms': ave_rooms, 'AveBedrms': ave_bedrms,
        'Population': population, 'AveOccup': ave_occup,
        'Latitude': latitude,  'Longitude': longitude,
        'RoomsPerOccupant':       ave_rooms / ave_occup,
        'BedsPerRoom':            ave_bedrms / ave_rooms,
        'PopulationPerHousehold': population / ave_occup
    }])
    price = results['Random Forest']['model'].predict(input_data)[0]
    price = max(50000, min(price, 750000))
    return price


print("\n[8] Sample prediction on custom data:")
sample_price = predict_price(
    med_inc=6.0, house_age=15, ave_rooms=7, ave_bedrms=2.5,
    population=800, ave_occup=3.0, latitude=37.5, longitude=-122.0
)
mae = results['Random Forest']['MAE']
print(f"    Estimated Price : ${sample_price:,.0f}")
print(f"    Likely Range    : ${sample_price*0.9:,.0f} — ${sample_price*1.1:,.0f}")
print(f"    Typical Error   : ±${mae:,.0f}")

print("\n" + "=" * 60)
print(f"   DONE! Best model: {best_name} | R²={best['R2']:.3f} | MAE=${best['MAE']:,.0f}")
print("=" * 60)
