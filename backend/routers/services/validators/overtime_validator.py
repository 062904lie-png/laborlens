"""Overtime-specific validation configuration."""

OVERTIME_ESSENTIAL_CONCEPTS = ["ordinary_overtime_rule"]

OVERTIME_SUPPORTING_CONCEPTS = [
    "coverage_and_exclusions",
    "hours_worked",
    "rest_day_special_day_rates",
    "regular_holiday_rate",
    "omnibus_overtime_rules",
]

OVERTIME_REQUIRED_ARTICLES = {"82", "84", "87", "93", "94"}
