"""
ReturnIQ Dataset Builder and Model Trainer (v2)
-----------------------------------------------
✓ Generates realistic FedEx rates (in INR)
✓ Simulates 10,000 international + domestic returns
✓ Adjusts domestic distance and pricing
✓ Adds new feature: is_domestic
✓ Balances classes for better model predictions
✓ Saves dataset and trained RandomForest model
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# ---------------- Setup directories ----------------
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODELS = BASE / "models"
DATA.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)

# ---------------- FedEx INR Rate Table ----------------
zones = list("ABCDEFGH")
rows = []
for o in zones:
    for d in zones:
        base = 800 + 20 * zones.index(o) + 150 * zones.index(d)
        perkg = 400 + 30 * abs(zones.index(o) - zones.index(d))
        fragile = 150 + 10 * zones.index(d)
        valuable_pct = 0.02 + 0.002 * zones.index(d)
        rows.append(
            dict(
                origin_zone=o,
                destination_zone=d,
                service="EXPRESS",
                base_cost_inr=base,
                per_kg_cost_inr=perkg,
                fragile_surcharge_inr=fragile,
                valuable_surcharge_percent=valuable_pct,
            )
        )

fedex_rates = pd.DataFrame(rows)
fedex_rates.to_csv(DATA / "fedex_rates.csv", index=False)
print("✅ FedEx rate table created → data/fedex_rates.csv")

# ---------------- Simulate Returns Dataset ----------------
N = 10000
rng = np.random.default_rng(42)
df = pd.DataFrame(
    {
        "product_price": rng.uniform(10, 500000, N),
        "weight_kg": rng.uniform(0.1, 20, N),
        "distance_km": rng.uniform(200, 15000, N),
        "origin_zone": rng.choice(zones, N),
        "destination_zone": rng.choice(zones, N),
        "customs_delay_days": rng.integers(0, 10, N),
        "tariff_rate": rng.uniform(0, 0.2, N),
        "condition_score": rng.uniform(0.3, 1.0, N),
        "fragile": rng.choice([0, 1], N, p=[0.85, 0.15]),
        "valuable": rng.choice([0, 1], N, p=[0.9, 0.1]),
    }
)
df["is_domestic"] = (df["origin_zone"] == df["destination_zone"]).astype(int)

# Adjust domestic distances
df["distance_km"] = np.where(df["is_domestic"] == 1,
                             rng.uniform(50, 500, N),
                             df["distance_km"])

# ---------------- Shipping Cost ----------------
def shipping_cost(row):
    rate = fedex_rates[
        (fedex_rates.origin_zone == row.origin_zone)
        & (fedex_rates.destination_zone == row.destination_zone)
    ].iloc[0]
    cost = rate.base_cost_inr + rate.per_kg_cost_inr * row.weight_kg
    if row.fragile:
        cost += rate.fragile_surcharge_inr
    if row.valuable:
        cost += rate.valuable_surcharge_percent * row.product_price
    # Domestic discount
    if row.origin_zone == row.destination_zone:
        cost *= 0.5
    return cost


df["shipping_cost_inr"] = df.apply(shipping_cost, axis=1)

# ---------------- CO2 Estimation ----------------
def estimate_co2(billable_weight, distance_km, mode="air"):
    factors = {"air": 0.5, "road": 0.062, "sea": 0.02}
    f = factors.get(mode, 0.5)
    return f * (billable_weight / 1000.0) * distance_km


df["co2_kg"] = df.apply(lambda r: estimate_co2(r.weight_kg, r.distance_km), axis=1)

# ---------------- Base Profit ----------------
df["resale_value_est"] = df["product_price"] * rng.uniform(0.2, 0.9, N)
df["profit_after_return"] = df["resale_value_est"] - (
    df["shipping_cost_inr"] + df["tariff_rate"] * df["product_price"] + 100
)

# ---------------- Labels ----------------
def label(row):
    if row.e_profit > -100 and row.cvar_alpha_loss < 0.25 * row.product_price:
        return 1
    elif row.e_profit > -600:
        return 0
    else:
        return -1

# Simulate simple risk to enable labeling
df["e_profit"] = df["profit_after_return"]
df["cvar_alpha_loss"] = np.maximum(0, -0.1 * df["e_profit"])
df["label"] = df.apply(label, axis=1)

# ---------------- Rebalance Dataset ----------------
print("⚙️ Balancing dataset...")
majority = df[df.label == -1]
minor_accept = df[df.label == 1]
minor_refund = df[df.label == 0]

minor_accept_up = resample(minor_accept, replace=True, n_samples=len(majority)//2, random_state=42)
minor_refund_up = resample(minor_refund, replace=True, n_samples=len(majority)//2, random_state=42)

df = pd.concat([majority, minor_accept_up, minor_refund_up]).sample(frac=1, random_state=42)
print(df.label.value_counts())

# ---------------- Feature Engineering ----------------
df["return_cost_ratio"] = df["shipping_cost_inr"] / df["product_price"]
df["resale_ratio"] = df["resale_value_est"] / df["product_price"]
df["e_profit_rel"] = df["e_profit"] / df["product_price"]
df["cvar_rel"] = df["cvar_alpha_loss"] / df["product_price"]

# ---------------- Save Dataset ----------------
df.to_csv(DATA / "returniq_dataset_balanced.csv", index=False)
print("✅ Dataset saved → data/returniq_dataset_balanced.csv")

# ---------------- Train Model ----------------
features = [
    "return_cost_ratio", "resale_ratio", "condition_score",
    "customs_delay_days", "co2_kg", "distance_km",
    "fragile", "valuable", "is_domestic", "cvar_rel", "e_profit_rel"
]
X = df[features].fillna(0)
y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample")
model.fit(Xtr, ytr)
acc = model.score(Xte, yte)

joblib.dump(model, MODELS / "returniq_model.pkl")
print(f"✅ Model trained → models/returniq_model.pkl (Accuracy: {acc:.3f})")

# ---------------- Diagnostic Plot ----------------
# plt.figure(figsize=(6,4))
# df["label"].value_counts().sort_index().plot(kind='bar', color=['green','orange','red'])
# plt.title("Label Distribution After Balancing")
# plt.xlabel("Label (1=Accept, 0=Refund, -1=Reject)")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.savefig(DATA / "label_distribution.png")
print("📊 Diagnostic chart saved → data/label_distribution.png")

print("🎉 All files successfully generated. You can now run: streamlit run app.py")
