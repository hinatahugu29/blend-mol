"""Single source of per-element data for Blend Mol.

Every element the add-on knows about has exactly one record here. Display
colours, display radii, covalent radii, the editor's element list, typical
valences and SMILES support are all derived from this table, so adding an
element means adding one record instead of updating several modules.

This module has no Blender dependency and can be imported by the
dependency-free tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementInfo:
    symbol: str
    # Default CPK/Jmol-style display colour (RGBA, 0-1).
    color: tuple[float, float, float, float]
    # Sphere radius in Blender units before the Atom Scale setting.
    display_radius: float
    # Covalent radius in angstroms (Cordero et al. 2008), used for
    # display-oriented bond lengths. Not a UFF parameter.
    covalent_radius: float
    # Neutral-state valence used for implicit hydrogen counts.
    typical_valence: int
    # Highest bond-order sum accepted before flagging an over-valence error.
    # Differs from typical_valence only for hypervalent S/P and their heavier
    # group partners (Se, Te, As, Sb).
    max_valence: int
    # Whether the SMILES reader accepts this element.
    smiles: bool
    # Whether it is offered in the editor's element list.
    in_editor: bool = True


# The order matters: it is the order of the editor's element EnumProperty,
# whose stored value is an index. Append new elements at the end so existing
# .blend files keep their selected element.
_ELEMENT_RECORDS = (
    ElementInfo("H", (0.92, 0.92, 0.92, 1.0), 0.22, 0.31, 1, 1, True),  # bracket only: [2H], [H]
    ElementInfo("C", (0.18, 0.18, 0.18, 1.0), 0.34, 0.76, 4, 4, True),
    ElementInfo("N", (0.12, 0.26, 0.80, 1.0), 0.33, 0.71, 3, 3, True),
    ElementInfo("O", (0.82, 0.08, 0.08, 1.0), 0.31, 0.66, 2, 2, True),
    ElementInfo("P", (1.00, 0.50, 0.00, 1.0), 0.38, 1.07, 3, 5, True),
    ElementInfo("S", (0.92, 0.78, 0.05, 1.0), 0.38, 1.05, 2, 6, True),
    ElementInfo("F", (0.15, 0.75, 0.20, 1.0), 0.29, 0.57, 1, 1, True),
    ElementInfo("Cl", (0.10, 0.60, 0.15, 1.0), 0.36, 1.02, 1, 1, True),
    ElementInfo("Br", (0.45, 0.12, 0.05, 1.0), 0.39, 1.20, 1, 1, True),
    ElementInfo("I", (0.36, 0.08, 0.62, 1.0), 0.43, 1.39, 1, 1, True),
    ElementInfo("B", (1.00, 0.71, 0.71, 1.0), 0.33, 0.84, 3, 3, True),
    ElementInfo("Si", (0.94, 0.78, 0.63, 1.0), 0.40, 1.11, 4, 4, True),
    # Heavier main-group elements with a UFF type (0.1.87). Appended after
    # Si so the editor list indices of existing elements do not change.
    ElementInfo("Ge", (0.40, 0.56, 0.56, 1.0), 0.41, 1.20, 4, 4, True),
    ElementInfo("Sn", (0.40, 0.50, 0.50, 1.0), 0.45, 1.39, 4, 4, True),
    ElementInfo("As", (0.74, 0.50, 0.89, 1.0), 0.41, 1.19, 3, 5, True),
    ElementInfo("Sb", (0.62, 0.39, 0.71, 1.0), 0.45, 1.39, 3, 5, True),
    ElementInfo("Se", (1.00, 0.63, 0.00, 1.0), 0.41, 1.20, 2, 6, True),
    ElementInfo("Te", (0.83, 0.48, 0.00, 1.0), 0.45, 1.38, 2, 6, True),
    # SMILES/MOL wildcard: a placeholder for "any atom". It takes no implicit
    # hydrogens and has no UFF type, so UFF refuses a molecule containing it.
    ElementInfo("*", (0.55, 0.55, 0.55, 1.0), 0.25, 0.80, 0, 8, True, in_editor=False),
)

DUMMY_SYMBOL = "*"

ELEMENTS: dict[str, ElementInfo] = {record.symbol: record for record in _ELEMENT_RECORDS}

# Fallbacks for an element symbol with no record (e.g. read from a file).
UNKNOWN_COLOR = (0.65, 0.25, 0.65, 1.0)
UNKNOWN_DISPLAY_RADIUS = 0.34
UNKNOWN_COVALENT_RADIUS = 0.8

# The editor's element list (excludes the wildcard).
ELEMENT_SYMBOLS = tuple(record.symbol for record in _ELEMENT_RECORDS if record.in_editor)
ELEMENT_COLORS = {symbol: info.color for symbol, info in ELEMENTS.items()}
ELEMENT_RADII = {symbol: info.display_radius for symbol, info in ELEMENTS.items()}
COVALENT_RADII = {symbol: info.covalent_radius for symbol, info in ELEMENTS.items()}
TYPICAL_VALENCE = {symbol: info.typical_valence for symbol, info in ELEMENTS.items()}
MAX_VALENCE = {symbol: info.max_valence for symbol, info in ELEMENTS.items()}
SMILES_ELEMENTS = frozenset(symbol for symbol, info in ELEMENTS.items() if info.smiles)


def charged_typical_valence(element: str, formal_charge: int) -> int | None:
    """Return the implicit-hydrogen valence after common formal-charge rules.

    Covers the charged states the add-on already handles: ammonium-like N+
    (4), alkoxide-like O- (1) and borate-like B- (4, e.g. BH4- or BF4-).
    Returns None for an element with no record.
    """
    if element == "N" and formal_charge > 0:
        return 4
    if element == "O" and formal_charge < 0:
        return 1
    if element == "B" and formal_charge < 0:
        return 4
    return TYPICAL_VALENCE.get(element)
