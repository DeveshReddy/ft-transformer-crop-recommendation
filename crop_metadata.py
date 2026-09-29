"""
Crop metadata for recommendations: type, water, duration, sowing/harvest, ideal NPK, season suitability.
Keys match dataset labels (lowercase). Used by app for filters, calendar, NPK gap, water advisor, etc.
"""

from typing import Dict, List, Any, Optional

# Season names (India-style). Month 1-12.
MONTH_TO_SEASON = {
    1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Winter",
}

# Crop type for filter
CROP_TYPES = ["Cereal", "Pulses", "Vegetable", "Fruit", "Beverage", "Fiber", "Commercial"]

# Metadata: type, water_requirement_mm (approx), duration_days, sowing_months (1-12), harvest_months,
# ideal_N/P/K (min,max), water_source (Rainfed/Irrigated/Both), suitable_seasons
CROP_METADATA: Dict[str, Dict[str, Any]] = {
    "rice": {
        "crop_type": "Cereal",
        "water_requirement_mm": 1200,
        "water_category": "High",
        "duration_days": 120,
        "sowing_months": [6, 7, 1],  # Kharif / Rabi
        "harvest_months": [10, 11, 4, 5],
        "ideal_N": (50, 120),
        "ideal_P": (20, 60),
        "ideal_K": (20, 50),
        "water_source": "Irrigated",
        "suitable_seasons": ["Monsoon", "Winter", "Post-Monsoon"],
    },
    "maize": {
        "crop_type": "Cereal",
        "water_requirement_mm": 500,
        "water_category": "Medium",
        "duration_days": 100,
        "sowing_months": [6, 7],
        "harvest_months": [9, 10],
        "ideal_N": (60, 120),
        "ideal_P": (30, 70),
        "ideal_K": (25, 60),
        "water_source": "Both",
        "suitable_seasons": ["Monsoon", "Summer"],
    },
    "chickpea": {
        "crop_type": "Pulses",
        "water_requirement_mm": 350,
        "water_category": "Low",
        "duration_days": 110,
        "sowing_months": [10, 11],
        "harvest_months": [2, 3],
        "ideal_N": (20, 40),
        "ideal_P": (40, 80),
        "ideal_K": (20, 50),
        "water_source": "Rainfed",
        "suitable_seasons": ["Winter", "Post-Monsoon"],
    },
    "kidneybeans": {
        "crop_type": "Pulses",
        "water_requirement_mm": 400,
        "water_category": "Medium",
        "duration_days": 90,
        "sowing_months": [6, 7, 2],
        "harvest_months": [9, 10, 5],
        "ideal_N": (25, 50),
        "ideal_P": (30, 70),
        "ideal_K": (25, 55),
        "water_source": "Both",
        "suitable_seasons": ["Monsoon", "Winter", "Summer"],
    },
    "pigeonpeas": {
        "crop_type": "Pulses",
        "water_requirement_mm": 600,
        "water_category": "Medium",
        "duration_days": 160,
        "sowing_months": [6, 7],
        "harvest_months": [1, 2, 12],
        "ideal_N": (20, 50),
        "ideal_P": (30, 60),
        "ideal_K": (20, 40),
        "water_source": "Rainfed",
        "suitable_seasons": ["Monsoon", "Winter"],
    },
    "mothbeans": {
        "crop_type": "Pulses",
        "water_requirement_mm": 300,
        "water_category": "Low",
        "duration_days": 75,
        "sowing_months": [6, 7],
        "harvest_months": [9, 10],
        "ideal_N": (20, 45),
        "ideal_P": (25, 55),
        "ideal_K": (20, 45),
        "water_source": "Rainfed",
        "suitable_seasons": ["Monsoon", "Summer"],
    },
    "mungbean": {
        "crop_type": "Pulses",
        "water_requirement_mm": 350,
        "water_category": "Low",
        "duration_days": 65,
        "sowing_months": [3, 6, 7],
        "harvest_months": [5, 9, 10],
        "ideal_N": (20, 50),
        "ideal_P": (25, 60),
        "ideal_K": (20, 50),
        "water_source": "Both",
        "suitable_seasons": ["Summer", "Monsoon"],
    },
    "blackgram": {
        "crop_type": "Pulses",
        "water_requirement_mm": 400,
        "water_category": "Medium",
        "duration_days": 90,
        "sowing_months": [6, 7],
        "harvest_months": [9, 10],
        "ideal_N": (25, 50),
        "ideal_P": (30, 65),
        "ideal_K": (20, 50),
        "water_source": "Both",
        "suitable_seasons": ["Monsoon", "Summer"],
    },
    "lentil": {
        "crop_type": "Pulses",
        "water_requirement_mm": 250,
        "water_category": "Low",
        "duration_days": 100,
        "sowing_months": [10, 11],
        "harvest_months": [2, 3],
        "ideal_N": (20, 40),
        "ideal_P": (40, 80),
        "ideal_K": (20, 45),
        "water_source": "Rainfed",
        "suitable_seasons": ["Winter", "Post-Monsoon"],
    },
    "apple": {
        "crop_type": "Fruit",
        "water_requirement_mm": 900,
        "water_category": "High",
        "duration_days": 180,
        "sowing_months": [1, 2, 12],
        "harvest_months": [7, 8, 9],
        "ideal_N": (40, 80),
        "ideal_P": (30, 60),
        "ideal_K": (50, 120),
        "water_source": "Irrigated",
        "suitable_seasons": ["Winter", "Summer"],
    },
    "banana": {
        "crop_type": "Fruit",
        "water_requirement_mm": 1200,
        "water_category": "High",
        "duration_days": 365,
        "sowing_months": [1, 2, 3, 6, 7, 8],
        "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "ideal_N": (100, 200),
        "ideal_P": (40, 80),
        "ideal_K": (200, 400),
        "water_source": "Irrigated",
        "suitable_seasons": ["Summer", "Monsoon", "Winter"],
    },
    "mango": {
        "crop_type": "Fruit",
        "water_requirement_mm": 800,
        "water_category": "High",
        "duration_days": 365,
        "sowing_months": [6, 7, 8],
        "harvest_months": [3, 4, 5, 6],
        "ideal_N": (60, 120),
        "ideal_P": (30, 70),
        "ideal_K": (80, 180),
        "water_source": "Irrigated",
        "suitable_seasons": ["Monsoon", "Summer"],
    },
    "grapes": {
        "crop_type": "Fruit",
        "water_requirement_mm": 600,
        "water_category": "Medium",
        "duration_days": 150,
        "sowing_months": [1, 2, 12],
        "harvest_months": [3, 4, 5, 6],
        "ideal_N": (50, 100),
        "ideal_P": (30, 70),
        "ideal_K": (80, 150),
        "water_source": "Irrigated",
        "suitable_seasons": ["Winter", "Summer"],
    },
    "watermelon": {
        "crop_type": "Fruit",
        "water_requirement_mm": 500,
        "water_category": "Medium",
        "duration_days": 85,
        "sowing_months": [2, 3, 11],
        "harvest_months": [5, 6, 1],
        "ideal_N": (50, 100),
        "ideal_P": (30, 60),
        "ideal_K": (40, 100),
        "water_source": "Irrigated",
        "suitable_seasons": ["Summer", "Winter", "Post-Monsoon"],
    },
    "muskmelon": {
        "crop_type": "Fruit",
        "water_requirement_mm": 450,
        "water_category": "Medium",
        "duration_days": 90,
        "sowing_months": [2, 3],
        "harvest_months": [5, 6],
        "ideal_N": (50, 90),
        "ideal_P": (30, 60),
        "ideal_K": (40, 90),
        "water_source": "Irrigated",
        "suitable_seasons": ["Summer"],
    },
    "orange": {
        "crop_type": "Fruit",
        "water_requirement_mm": 900,
        "water_category": "High",
        "duration_days": 365,
        "sowing_months": [6, 7, 8],
        "harvest_months": [11, 12, 1, 2],
        "ideal_N": (80, 150),
        "ideal_P": (30, 70),
        "ideal_K": (100, 200),
        "water_source": "Irrigated",
        "suitable_seasons": ["Monsoon", "Winter"],
    },
    "papaya": {
        "crop_type": "Fruit",
        "water_requirement_mm": 1000,
        "water_category": "High",
        "duration_days": 330,
        "sowing_months": [6, 7, 8, 9],
        "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "ideal_N": (80, 150),
        "ideal_P": (40, 80),
        "ideal_K": (100, 200),
        "water_source": "Irrigated",
        "suitable_seasons": ["Monsoon", "Summer", "Post-Monsoon"],
    },
    "pomegranate": {
        "crop_type": "Fruit",
        "water_requirement_mm": 600,
        "water_category": "Medium",
        "duration_days": 180,
        "sowing_months": [6, 7, 2],
        "harvest_months": [9, 10, 11, 5, 6],
        "ideal_N": (50, 100),
        "ideal_P": (30, 70),
        "ideal_K": (80, 150),
        "water_source": "Irrigated",
        "suitable_seasons": ["Monsoon", "Summer", "Winter"],
    },
    "coconut": {
        "crop_type": "Fruit",
        "water_requirement_mm": 1200,
        "water_category": "High",
        "duration_days": 3650,
        "sowing_months": [5, 6, 7, 8],
        "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "ideal_N": (50, 100),
        "ideal_P": (30, 60),
        "ideal_K": (80, 150),
        "water_source": "Both",
        "suitable_seasons": ["Summer", "Monsoon"],
    },
    "cotton": {
        "crop_type": "Fiber",
        "water_requirement_mm": 700,
        "water_category": "High",
        "duration_days": 180,
        "sowing_months": [4, 5, 6],
        "harvest_months": [10, 11, 12],
        "ideal_N": (60, 120),
        "ideal_P": (30, 70),
        "ideal_K": (40, 80),
        "water_source": "Irrigated",
        "suitable_seasons": ["Summer", "Monsoon"],
    },
    "jute": {
        "crop_type": "Fiber",
        "water_requirement_mm": 500,
        "water_category": "Medium",
        "duration_days": 120,
        "sowing_months": [3, 4],
        "harvest_months": [6, 7, 8],
        "ideal_N": (40, 80),
        "ideal_P": (20, 50),
        "ideal_K": (40, 80),
        "water_source": "Both",
        "suitable_seasons": ["Summer", "Monsoon"],
    },
    "coffee": {
        "crop_type": "Beverage",
        "water_requirement_mm": 1100,
        "water_category": "High",
        "duration_days": 1095,
        "sowing_months": [6, 7, 8],
        "harvest_months": [11, 12, 1, 2],
        "ideal_N": (80, 150),
        "ideal_P": (30, 70),
        "ideal_K": (80, 150),
        "water_source": "Both",
        "suitable_seasons": ["Monsoon", "Winter"],
    },
}


def get_metadata(crop_name: str) -> Dict[str, Any]:
    """Return metadata for a crop. Keys normalized to lowercase. Unknown crops get defaults."""
    key = str(crop_name).strip().lower()
    if key in CROP_METADATA:
        return CROP_METADATA[key].copy()
    return {
        "crop_type": "Other",
        "water_requirement_mm": 500,
        "water_category": "Medium",
        "duration_days": 120,
        "sowing_months": [1, 6, 7],
        "harvest_months": [4, 5, 10, 11],
        "ideal_N": (30, 90),
        "ideal_P": (20, 60),
        "ideal_K": (25, 60),
        "water_source": "Both",
        "suitable_seasons": ["Summer", "Monsoon", "Winter", "Post-Monsoon"],
    }


def get_current_season(month: Optional[int] = None) -> str:
    """Return current season name. If month is None, use today."""
    if month is None:
        from datetime import datetime
        month = datetime.now().month
    return MONTH_TO_SEASON.get(month, "Summer")


def npk_gaps(user_N: float, user_P: float, user_K: float, meta: Dict[str, Any]) -> List[str]:
    """Return list of fertilizer gap messages (what to add or reduce)."""
    out = []
    n_lo, n_hi = meta["ideal_N"]
    p_lo, p_hi = meta["ideal_P"]
    k_lo, k_hi = meta["ideal_K"]
    if user_N < n_lo:
        out.append(f"N: add ~{n_lo - user_N:.0f} kg/ha")
    elif user_N > n_hi:
        out.append("N: reduce (above ideal range)")
    else:
        out.append("N: adequate")
    if user_P < p_lo:
        out.append(f"P: add ~{p_lo - user_P:.0f} kg/ha")
    elif user_P > p_hi:
        out.append("P: reduce (above ideal range)")
    else:
        out.append("P: adequate")
    if user_K < k_lo:
        out.append(f"K: add ~{k_lo - user_K:.0f} kg/ha")
    elif user_K > k_hi:
        out.append("K: reduce (above ideal range)")
    else:
        out.append("K: adequate")
    return out


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def month_name(month_num: int) -> str:
    """Single month number (1-12) to name."""
    return MONTH_NAMES[month_num - 1] if 1 <= month_num <= 12 else "—"


def month_names(months: List[int]) -> str:
    """Convert list of month numbers to readable string."""
    return ", ".join(MONTH_NAMES[m - 1] for m in sorted(set(months))) if months else "—"
