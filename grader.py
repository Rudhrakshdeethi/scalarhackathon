def calculate_grade(final_obs, target_ratio=0.5) -> float:
    total_val = final_obs.portfolio_value
    if total_val <= 0: return 0.0
    actual_ratio = (total_val - final_obs.cash) / total_val
    error = abs(actual_ratio - target_ratio)
    score = max(0.0, 1.0 - (error / target_ratio))
    return round(score, 2)