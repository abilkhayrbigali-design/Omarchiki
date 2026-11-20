"""Word pools for fictional names and line descriptions.

Names are random combinations of generic words. They are synthetic and do not refer to real
companies or people.
"""

from __future__ import annotations

from auditgym.util.rng import Stream

_COMPANY_A = (
    "Altair",
    "Birchfield",
    "Cobaltine",
    "Dunemark",
    "Emberline",
    "Fjordway",
    "Granitex",
    "Harborlight",
    "Irisvale",
    "Juniperon",
    "Kestrelis",
    "Lumenford",
    "Meridane",
    "Nimbuston",
    "Orchidia",
    "Pinecrestra",
    "Quartzel",
    "Riverbendt",
    "Saffronet",
    "Tundrava",
    "Umberlyn",
    "Vesperon",
    "Willowmere",
    "Zephyrant",
)
_COMPANY_B = (
    "Logistics",
    "Textiles",
    "Analytics",
    "Foods",
    "Instruments",
    "Supply",
    "Engineering",
    "Media",
    "Agro",
    "Systems",
    "Robotics",
    "Packaging",
)
_COMPANY_SUFFIX = ("LLP", "LLC", "Ltd", "GmbH", "JSC", "Trading Co.")

_CUSTOMER_A = (
    "Arvento",
    "Belmaris",
    "Corvane",
    "Delvista",
    "Estrelio",
    "Falconer",
    "Galvatek",
    "Hollisden",
    "Ingravia",
    "Jasperine",
    "Korvella",
    "Lindqvast",
    "Monteral",
    "Novacrest",
    "Ostrava Line",
    "Pellucid",
    "Quenmore",
    "Rostavik",
    "Sundaria",
    "Tavernor",
    "Ulvenhoff",
    "Valtessa",
    "Westerlo",
    "Yarrowby",
)
_CUSTOMER_B = ("Retail", "Holdings", "Works", "Partners", "Group", "Studio", "Markets", "Labs")

DESCRIPTIONS = (
    "Consulting hours",
    "Software licence",
    "Equipment rental",
    "Maintenance visit",
    "Training session",
    "Freight services",
    "Spare parts kit",
    "Installation works",
    "Support plan",
    "Data processing",
    "Warehouse storage",
    "Calibration service",
    "Design package",
    "Field survey",
    "Packaging materials",
    "Printing services",
    "Cloud hosting",
    "Translation services",
)

SUBSCRIPTIONS = (
    "Monthly support plan",
    "Monthly hosting plan",
    "Monthly maintenance contract",
    "Monthly analytics subscription",
    "Monthly equipment lease",
)


def company_name(s: Stream) -> str:
    return f"{s.choice(_COMPANY_A)} {s.choice(_COMPANY_B)} {s.choice(_COMPANY_SUFFIX)}"


def customer_name(s: Stream) -> str:
    return f"{s.choice(_CUSTOMER_A)} {s.choice(_CUSTOMER_B)}"
