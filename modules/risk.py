import numpy as np

def simulate_profits(row, S=200, alpha=0.95):
    rng = np.random.default_rng()
    a = max(0.5, row["condition_score"] * 5)
    b = max(0.5, (1 - row["condition_score"]) * 5)
    cond = rng.beta(a, b, S)
    resale = np.clip(row["resale_value_est"] * cond, 0, None)
    extra = rng.gamma(1 + row["customs_delay_days"] * 0.15, 50, S)
    profit = resale - (row["shipping_cost_inr"] + row["tariff_rate"]*row["product_price"] + 100 + extra)
    losses = -profit
    losses.sort()
    k = int(alpha * S) - 1
    cvar = losses[k:].mean()
    return profit.mean(), cvar
