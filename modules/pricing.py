import pandas as pd

_rates = None

def load_rates(path="data/fedex_rates.csv"):
    global _rates
    _rates = pd.read_csv(path)
    return _rates

def lookup_rate(origin_zone, destination_zone):
    global _rates
    if _rates is None:
        _rates = load_rates()
    row = _rates[(_rates.origin_zone==origin_zone)&(_rates.destination_zone==destination_zone)]
    if row.empty:
        r = _rates.iloc[0]
        return r.base_cost_inr, r.per_kg_cost_inr, r.fragile_surcharge_inr, r.valuable_surcharge_percent
    r = row.iloc[0]
    return r.base_cost_inr, r.per_kg_cost_inr, r.fragile_surcharge_inr, r.valuable_surcharge_percent

def compute_shipping(origin_zone, destination_zone, billable_weight, price, fragile, valuable):
    base, perkg, fragile_s, val_pct = lookup_rate(origin_zone, destination_zone)
    cost = base + perkg * billable_weight
    if fragile:
        cost += fragile_s
    if valuable:
        cost += val_pct * price
    return round(float(cost),2)
