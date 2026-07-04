import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv("data.csv", encoding="latin-1")

# ── 2. Feature engineering ───────────────────────────────────

df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  format="%m/%d/%Y")
df["Order Month"] = df["Order Date"].dt.month
df["Order Year"]  = df["Order Date"].dt.year
df["Ship Days"]   = (df["Ship Date"] - df["Order Date"]).dt.days

# Convert text columns to numbers using LabelEncoder
cat_cols = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Choose features (X) and target (y)
features = ["Quantity", "Discount", "Profit", "Order Month", "Order Year",
            "Ship Days", "Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
X = df[features]
y = df["Sales"]

# ── 3. Split into train and test sets ────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

# ── 4. Train the Random Forest model ─────────────────────────
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nResults on test set:")
print(f"  MAE  (avg error)     : ${mae:.2f}")
print(f"  RMSE (penalises big errors): ${rmse:.2f}")
print(f"  R²   (1 = perfect)  : {r2:.4f}")

# ── 6. Plot results ───────────────────────────────────────────
BG, CARD, TEXT = "#0f1117", "#1a1d27", "#e8eaf0"
GREEN, BLUE    = "#2ecc71", "#3498db"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": "#2a2d3a", "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": "#2a2d3a", "grid.linewidth": 0.6, "font.size": 10,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Random Forest — Sales Prediction Results", fontsize=15, fontweight="bold", y=1.02)

# Plot 1: Actual vs Predicted
ax = axes[0]
lim = max(y_test.max(), y_pred.max()) * 1.05
ax.scatter(y_test, y_pred, alpha=0.3, s=12, color=GREEN)
ax.plot([0, lim], [0, lim], "--", color="#e74c3c", lw=1.5, label="Perfect fit")
ax.set(xlim=(0, lim), ylim=(0, lim),
       xlabel="Actual Sales ($)", ylabel="Predicted Sales ($)",
       title="Actual vs Predicted")
ax.legend(); ax.grid(True)

# Plot 2: Residuals (errors)
ax = axes[1]
residuals = y_test.values - y_pred
ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=BLUE)
ax.axhline(0, color="#e74c3c", lw=1.5, linestyle="--")
ax.set(xlabel="Predicted Sales ($)", ylabel="Error ($)",
       title="Residuals  (errors should scatter around 0)")
ax.grid(True)

# Plot 3: Top feature importances
ax = axes[2]
feat_imp = pd.Series(model.feature_importances_, index=features).sort_values()
feat_imp.plot(kind="barh", ax=ax, color=GREEN, edgecolor="none")
ax.set(xlabel="Importance", title="Feature Importances\n(what drives Sales the most)")
ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig("sales_prediction_results.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
