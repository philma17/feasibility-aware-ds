"""Definitions of the generated FR3 joint-lock and mass variants."""

FIXED_VARIANTS = {
    "FR3_fixed_J3_J7": ("fr3_fixed_j3_j7.xml", (3, 7)),
    "FR3_fixed_J3_J5_J7": ("fr3_fixed_j3_j5_j7.xml", (3, 5, 7)),
    "FR3_fixed_J3_J5_J6_J7": ("fr3_fixed_j3_j5_j6_j7.xml", (3, 5, 6, 7)),
    "FR3_fixed_J1": ("fr3_fixed_j1.xml", (1,)),
    "FR3_fixed_J2": ("fr3_fixed_j2.xml", (2,)),
    "FR3_fixed_J4": ("fr3_fixed_j4.xml", (4,)),
    "FR3_fixed_J6": ("fr3_fixed_j6.xml", (6,)),
    "FR3_fixed_J7": ("fr3_fixed_j7.xml", (7,)),
    "FR3_fixed_J1_J3": ("fr3_fixed_j1_j3.xml", (1, 3)),
    "FR3_fixed_J2_J3": ("fr3_fixed_j2_j3.xml", (2, 3)),
    "FR3_fixed_J1_J5": ("fr3_fixed_j1_j5.xml", (1, 5)),
    "FR3_fixed_J5_J6_J7": ("fr3_fixed_j5_j6_j7.xml", (5, 6, 7)),
}
MASS_VARIANTS = {
    "FR3_link3_mass_150pct": ("fr3_link3_mass_150pct.xml", "fr3_link3", 1.5),
    "FR3_link3_mass_200pct": ("fr3_link3_mass_200pct.xml", "fr3_link3", 2.0),
    "FR3_link7_mass_150pct": ("fr3_link7_mass_150pct.xml", "fr3_link7", 1.5),
    "FR3_link7_mass_200pct": ("fr3_link7_mass_200pct.xml", "fr3_link7", 2.0),
}
