"""Definitions of the generated FR3 joint-lock and mass variants."""

FIXED_VARIANTS = {
    "FR3_fixed_J3_J7": ("fr3_fixed_j3_j7.xml", (3, 7)),
    "FR3_fixed_J3_J5_J7": ("fr3_fixed_j3_j5_j7.xml", (3, 5, 7)),
    "FR3_fixed_J3_J5_J6_J7": ("fr3_fixed_j3_j5_j6_j7.xml", (3, 5, 6, 7)),
}
MASS_VARIANTS = {
    "FR3_link3_mass_150pct": ("fr3_link3_mass_150pct.xml", "fr3_link3", 1.5),
    "FR3_link3_mass_200pct": ("fr3_link3_mass_200pct.xml", "fr3_link3", 2.0),
    "FR3_link7_mass_150pct": ("fr3_link7_mass_150pct.xml", "fr3_link7", 1.5),
    "FR3_link7_mass_200pct": ("fr3_link7_mass_200pct.xml", "fr3_link7", 2.0),
}
