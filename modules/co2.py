def estimate_co2_kg(billable_weight_kg, distance_km, mode="air"):
    factors = {"air":0.5, "road":0.062, "sea":0.02}
    f = factors.get(mode, 0.5)
    co2 = f * (billable_weight_kg/1000.0) * distance_km
    return round(co2,3)
