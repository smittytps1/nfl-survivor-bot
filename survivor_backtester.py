def calculate_model_prob(market_prob: float, is_home: bool, spread: float, week: int) -> float:
    if market_prob is None:
        return None
    early_discount = -0.035 if week <= 4 else 0.0
    home_edge = 0.010 if is_home else -0.005
    heavy_fav_boost = 0.020 if abs(spread) >= 9.5 else (-0.030 if abs(spread) < 6.5 else 0.0)
    
    adj_prob = market_prob + early_discount + home_edge + heavy_fav_boost
    return min(0.96, max(0.50, round(adj_prob, 3)))
