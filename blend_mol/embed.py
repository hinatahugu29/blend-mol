"""Dependency-free 3D starting geometry from a molecular graph (distance geometry).

The approach follows the distance-geometry family (as in RDKit's ETKDG, much
simplified):

1. Derive lower/upper distance bounds from the graph: bonds (covalent radii
   and bond order), bond angles (hybridisation and ring size), 1-4 pairs
   (cis/trans ranges, fixed for ring double bonds and SMILES ``/`` ``\\``),
   and a soft-contact lower bound for every other pair.
2. Place atoms in four dimensions and minimise a bounds-violation error plus
   chiral-volume and planarity terms. The fourth dimension lets an inverted
   stereocentre pass through instead of getting stuck.
3. Squeeze the fourth coordinate to zero and minimise again in 3D.

Several seeded attempts are made and the best one (all stereo constraints
met, smallest error) is returned, so results are reproducible.

Nothing here imports Blender or numpy, and atoms are plain indices so this
module does not depend on molfile.py.
"""

from __future__ import annotations

import math
import operator
import random
from dataclasses import dataclass

try:
    from .elements import COVALENT_RADII, UNKNOWN_COVALENT_RADIUS
except ImportError:  # Enables the module to be tested outside Blender.
    from elements import COVALENT_RADII, UNKNOWN_COVALENT_RADIUS

# Implicit hydrogen / lone pair slot in a stereocentre's neighbour order.
IMPLICIT = -1

# Contact radii (angstroms) for the soft lower bound between atoms more than
# three bonds apart; 0.75 x van der Waals radii keeps them from touching
# without forcing a specific conformation.
_CONTACT_RADIUS = {"H": 0.82, "C": 1.27, "N": 1.16, "O": 1.14, "F": 1.10, "P": 1.35, "S": 1.35,
                   "Cl": 1.31, "Br": 1.39, "I": 1.48, "B": 1.20, "Si": 1.58}
_DEFAULT_CONTACT_RADIUS = 1.27

# Signed volume (angstrom^3) a correctly handed tetrahedral centre must reach;
# an ideal sp3 centre with 1.5 A bonds has about 2.6.
_CHIRAL_VOLUME_MIN = 1.2
_CHIRAL_VOLUME_MAX = 4.5
# Sign of the neighbour-1/2/3 triple product for SMILES "@" when the first
# neighbour (or the implicit H) is the viewer. Verified against Open Babel's
# stereo perception from 3D coordinates (benchmarks/evaluate_smiles_embedding.py).
_AT_SIGN = -1.0


@dataclass(frozen=True)
class Tetrahedral:
    """SMILES-style tetrahedral stereo: neighbours in written order, IMPLICIT for H."""
    center: int
    neighbors: tuple[int, ...]
    clockwise: bool  # "@@"


@dataclass(frozen=True)
class DoubleBondStereo:
    """``first`` (on atom_a) and ``second`` (on atom_b) are cis (True) or trans."""
    atom_a: int
    atom_b: int
    first: int
    second: int
    cis: bool


def _bond_length(element_a: str, element_b: str, order: int) -> float:
    radius_a = COVALENT_RADII.get(element_a, UNKNOWN_COVALENT_RADIUS)
    radius_b = COVALENT_RADII.get(element_b, UNKNOWN_COVALENT_RADIUS)
    return (radius_a + radius_b) * (1.0 - 0.12 * (order - 1))


def _rings(count: int, neighbors: list[list[int]], max_size: int = 8) -> list[frozenset[int]]:
    """Smallest ring through each ring bond, up to ``max_size`` atoms, deduplicated."""
    rings: set[frozenset[int]] = set()
    for start in range(count):
        for end in neighbors[start]:
            if end < start:
                continue
            # Shortest path from start to end that does not use the start-end bond.
            previous = {start: None}
            frontier = [start]
            found = False
            for _depth in range(max_size - 1):
                next_frontier = []
                for atom in frontier:
                    for other in neighbors[atom]:
                        if atom == start and other == end:
                            continue
                        if other in previous:
                            continue
                        previous[other] = atom
                        if other == end:
                            found = True
                            break
                        next_frontier.append(other)
                    if found:
                        break
                if found or not next_frontier:
                    break
                frontier = next_frontier
            if found:
                path, atom = [], end
                while atom is not None:
                    path.append(atom)
                    atom = previous[atom]
                rings.add(frozenset(path))
    return sorted(rings, key=len)


def _hybridization(elements, neighbors, orders, ring_of_atom, index) -> int:
    """1 (sp, linear), 2 (sp2, planar) or 3 (sp3)."""
    bond_orders = [orders[(min(index, other), max(index, other))] for other in neighbors[index]]
    doubles = sum(1 for order in bond_orders if order == 2)
    if any(order == 3 for order in bond_orders) or doubles >= 2:
        return 1
    if doubles == 1:
        return 2
    element = elements[index]
    if element in ("N", "O", "S") and len(neighbors[index]) <= 3:
        # A lone-pair atom conjugated with a pi system: planar amide/aniline N,
        # and ring O/S/N in an aromatic five-membered ring (furan, pyrrole).
        conjugated = any(
            any(orders[(min(other, n), max(other, n))] == 2 for n in neighbors[other])
            for other in neighbors[index]
        )
        if conjugated and (element == "N" or ring_of_atom[index] in (5, 6)):
            return 2
    return 3


def _interior_angle(size: int, hybridization: int) -> tuple[float, float]:
    """(angle, tolerance) in degrees for an angle inside a ring of ``size``."""
    if size == 3:
        return 60.0, 3.0
    if size == 4:
        return 89.0, 5.0
    if size == 5:
        return (108.0, 5.0) if hybridization == 2 else (104.5, 6.0)
    if size == 6:
        return (120.0, 4.0) if hybridization == 2 else (111.0, 5.0)
    return (124.0, 12.0) if hybridization == 2 else (114.0, 9.0)


def _ideal_angle(hybridization: int) -> tuple[float, float]:
    return {1: (180.0, 4.0), 2: (120.0, 5.0), 3: (109.5, 5.0)}[hybridization]


def _law_of_cosines(a: float, b: float, angle_degrees: float) -> float:
    return math.sqrt(max(a * a + b * b - 2.0 * a * b * math.cos(math.radians(angle_degrees)), 1e-6))


def _dihedral_distance(a: float, b: float, c: float, angle_ab: float, angle_bc: float, dihedral: float) -> float:
    """1-4 distance for bonds a, b, c with angles (degrees) and a dihedral."""
    t1, t2 = math.radians(angle_ab), math.radians(angle_bc)
    # Place atom 2 at origin, atom 3 on +x; atom 1 and 4 in the xy / rotated plane.
    p1 = (a * math.cos(t1), a * math.sin(t1), 0.0)
    p4 = (b + c * math.cos(math.pi - t2), c * math.sin(math.pi - t2) * math.cos(math.radians(dihedral)),
          c * math.sin(math.pi - t2) * math.sin(math.radians(dihedral)))
    return math.dist(p1, p4)


class _Problem:
    """Distance bounds plus chiral and planarity constraints for one molecule."""

    def __init__(self, elements, bonds, tetrahedral, double_stereo):
        count = len(elements)
        self.count = count
        neighbors: list[list[int]] = [[] for _ in range(count)]
        orders: dict[tuple[int, int], int] = {}
        lengths: dict[tuple[int, int], float] = {}
        for first, second, order in bonds:
            key = (min(first, second), max(first, second))
            neighbors[first].append(second)
            neighbors[second].append(first)
            orders[key] = order
            lengths[key] = _bond_length(elements[first], elements[second], order)
        self.neighbors = neighbors
        rings = _rings(count, neighbors)
        ring_of_atom = [0] * count
        for ring in rings:
            for atom in ring:
                if not ring_of_atom[atom]:
                    ring_of_atom[atom] = len(ring)
        hybrid = [_hybridization(elements, neighbors, orders, ring_of_atom, i) for i in range(count)]

        def smallest_ring(*atoms) -> frozenset[int] | None:
            return next((ring for ring in rings if all(atom in ring for atom in atoms)), None)

        bounds: dict[tuple[int, int], tuple[float, float, float]] = {}  # (lower, upper, weight)

        def set_bounds(i, j, lower, upper, weight=1.0):
            key = (min(i, j), max(i, j))
            if key in bounds:
                old = bounds[key]
                lower, upper = max(lower, old[0]), min(upper, old[1])
                if lower > upper:
                    lower = upper = (lower + upper) / 2.0
                weight = max(weight, old[2])
            bounds[key] = (lower, upper, weight)

        for key, length in lengths.items():
            set_bounds(key[0], key[1], length - 0.01, length + 0.01, 4.0)

        angles: dict[tuple[int, int, int], float] = {}
        for center in range(count):
            for position, first in enumerate(neighbors[center]):
                for second in neighbors[center][position + 1:]:
                    ring = smallest_ring(center, first, second)
                    if ring is not None:
                        angle, tolerance = _interior_angle(len(ring), hybrid[center])
                    else:
                        center_ring = smallest_ring(center, first) or smallest_ring(center, second)
                        if center_ring is not None and hybrid[center] == 2 and len(center_ring) <= 6:
                            interior, _tol = _interior_angle(len(center_ring), 2)
                            angle, tolerance = (360.0 - interior) / 2.0, 6.0
                        else:
                            angle, tolerance = _ideal_angle(hybrid[center])
                            if ring_of_atom[center]:
                                tolerance += 4.0
                    angles[(first, center, second)] = angles[(second, center, first)] = angle
                    a = lengths[(min(center, first), max(center, first))]
                    b = lengths[(min(center, second), max(center, second))]
                    low = _law_of_cosines(a, b, max(angle - tolerance, 0.0))
                    high = _law_of_cosines(a, b, min(angle + tolerance, 180.0))
                    set_bounds(first, second, low, high, 2.0)

        # 1-4 pairs: dihedral range, or fixed for double bonds of known geometry.
        fixed_dihedral: dict[tuple[int, int, int, int], float] = {}

        for stereo in double_stereo:
            a_subs = [n for n in neighbors[stereo.atom_a] if n != stereo.atom_b]
            b_subs = [n for n in neighbors[stereo.atom_b] if n != stereo.atom_a]
            for x in a_subs:
                for y in b_subs:
                    cis = stereo.cis == ((x == stereo.first) == (y == stereo.second))
                    fixed_dihedral[(x, stereo.atom_a, stereo.atom_b, y)] = 0.0 if cis else 180.0
        for (middle_a, middle_b), order in orders.items():
            if hybrid[middle_a] == 1 or hybrid[middle_b] == 1:
                continue
            ring = smallest_ring(middle_a, middle_b)
            for x in neighbors[middle_a]:
                if x == middle_b:
                    continue
                for y in neighbors[middle_b]:
                    if y == middle_a or y == x:
                        continue
                    a = lengths[(min(x, middle_a), max(x, middle_a))]
                    b = lengths[(min(middle_a, middle_b), max(middle_a, middle_b))]
                    c = lengths[(min(middle_b, y), max(middle_b, y))]
                    angle_1 = angles.get((x, middle_a, middle_b), 109.5)
                    angle_2 = angles.get((middle_a, middle_b, y), 109.5)
                    dihedral = fixed_dihedral.get((x, middle_a, middle_b, y))
                    if dihedral is None:
                        dihedral = fixed_dihedral.get((y, middle_b, middle_a, x))
                    if dihedral is None and order == 2 and ring is not None and len(ring) <= 8:
                        # Ring double bond: ring-to-ring and exo-to-exo pairs are cis.
                        dihedral = 0.0 if ((x in ring) == (y in ring)) else 180.0
                    if dihedral is None and ring is not None and len(ring) <= 4 and x in ring and y in ring:
                        dihedral = 0.0
                    chair = ring is not None and len(ring) == 6 and all(hybrid[atom] == 3 for atom in ring)
                    if dihedral is None and chair and x in ring and y in ring:
                        # Saturated six-membered ring: chair dihedrals are about 55
                        # degrees; boat (0) and twist-boat (about 30) are excluded.
                        low = _dihedral_distance(a, b, c, angle_1, angle_2, 45.0)
                        high = _dihedral_distance(a, b, c, angle_1, angle_2, 70.0)
                        set_bounds(x, y, low, high, 1.0)
                        continue
                    if dihedral is None and chair and ((x in ring) != (y in ring)):
                        # Exocyclic substituent: prefer equatorial (anti to the ring
                        # bond), only softly, because stereo may force it axial.
                        low = _dihedral_distance(a, b, c, angle_1, angle_2, 150.0)
                        high = _dihedral_distance(a, b, c, angle_1, angle_2, 180.0)
                        set_bounds(x, y, low, high + 0.05, 0.3)
                        continue
                    if dihedral is not None:
                        distance = _dihedral_distance(a, b, c, angle_1, angle_2, dihedral)
                        set_bounds(x, y, distance - 0.08, distance + 0.08, 1.0)
                    else:
                        cis = _dihedral_distance(a, b, c, angle_1, angle_2, 0.0)
                        trans = _dihedral_distance(a, b, c, angle_1, angle_2, 180.0)
                        # Eclipsed (0 degrees) geometries are rare; start the range near gauche.
                        lower = _dihedral_distance(a, b, c, angle_1, angle_2, 45.0)
                        set_bounds(x, y, min(lower, trans - 0.01) if order == 1 else cis, trans + 0.05, 0.5)

        # A 1-4 distance cannot tell a 0 degree dihedral from 45 once the angles
        # flex, so saturated six-membered rings also get explicit torsions, in
        # ring order: a chair has all six near 55 degrees with alternating signs.
        self.chair_rings: list[list[tuple[int, int, int, int]]] = []
        for ring in rings:
            if len(ring) != 6 or not all(hybrid[atom] == 3 for atom in ring):
                continue
            cycle = [min(ring)]
            while len(cycle) < 6:
                options = [n for n in neighbors[cycle[-1]] if n in ring and n not in cycle]
                if not options:
                    break
                cycle.append(min(options))
            if len(cycle) == 6:
                self.chair_rings.append([
                    tuple(cycle[(start + offset) % 6] for offset in range(4)) for start in range(6)
                ])

        # Everything else only needs to stay out of contact.
        self.near_pairs = [(i, j, lower, upper, weight) for (i, j), (lower, upper, weight) in bounds.items()]
        contact = [_CONTACT_RADIUS.get(element, _DEFAULT_CONTACT_RADIUS) for element in elements]
        self.contact = contact
        self.bounded = set(bounds)

        # Chiral volume constraints.
        self.chiral: list[tuple[int, int, int, int, float]] = []
        for spec in tetrahedral:
            # The caller supplies four entries, with IMPLICIT standing in for an
            # implicit H or lone pair at its SMILES position.
            order = list(spec.neighbors)
            if len(order) != 4:
                continue
            clockwise = spec.clockwise
            if IMPLICIT in order:
                k = order.index(IMPLICIT)
                if k != 0:
                    order[0], order[k] = order[k], order[0]
                    clockwise = not clockwise
            sign = _AT_SIGN * (-1.0 if clockwise else 1.0)
            self.chiral.append((spec.center, order[1], order[2], order[3], sign))
            if order[0] != IMPLICIT:
                # With four explicit neighbours, any three of them fix the handedness;
                # also constrain the triple seen from neighbour 1 for robustness.
                self.chiral.append((spec.center, order[0], order[3], order[2], sign))

        # Linear (sp) centres: distance bounds barely constrain an angle near
        # 180 degrees, so pull the centre onto the line between its neighbours
        # at the point the ideal bond lengths put it.
        self.linear: list[tuple[int, int, int, float]] = []
        for center in range(count):
            if hybrid[center] == 1 and len(neighbors[center]) == 2:
                first, second = neighbors[center]
                la = lengths[(min(center, first), max(center, first))]
                lb = lengths[(min(center, second), max(center, second))]
                self.linear.append((center, first, second, lb / (la + lb)))

        # Planarity: sp2 centres with three neighbours, and double-bond substituents.
        self.planar: list[tuple[int, int, int, int]] = []
        for center in range(count):
            if hybrid[center] == 2 and len(neighbors[center]) == 3:
                first, second, third = neighbors[center]
                self.planar.append((center, first, second, third))
        for (middle_a, middle_b), order in orders.items():
            if order != 2:
                continue
            for x in neighbors[middle_a]:
                for y in neighbors[middle_b]:
                    if x != middle_b and y != middle_a and x != y:
                        self.planar.append((middle_a, x, middle_b, y))
        self.lengths = lengths


def _volume(positions, center, a, b, c) -> float:
    pc, pa, pb, pd = positions[center], positions[a], positions[b], positions[c]
    ax, ay, az = pa[0] - pc[0], pa[1] - pc[1], pa[2] - pc[2]
    bx, by, bz = pb[0] - pc[0], pb[1] - pc[1], pb[2] - pc[2]
    cx, cy, cz = pd[0] - pc[0], pd[1] - pc[1], pd[2] - pc[2]
    return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)


def _volume_gradient(positions, center, a, b, c):
    """Gradients of _volume with respect to a, b, c (center gets minus their sum)."""
    pc, pa, pb, pd = positions[center], positions[a], positions[b], positions[c]
    u = (pa[0] - pc[0], pa[1] - pc[1], pa[2] - pc[2])
    v = (pb[0] - pc[0], pb[1] - pc[1], pb[2] - pc[2])
    w = (pd[0] - pc[0], pd[1] - pc[1], pd[2] - pc[2])

    def cross(p, q):
        return (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])

    return cross(v, w), cross(w, u), cross(u, v)


class _Objective:
    def __init__(self, problem: _Problem, dimensions: int):
        self.problem = problem
        self.dimensions = dimensions
        self.fourth_weight = 0.0
        # Planarity is left out while the fourth dimension is free, so a double
        # bond or ring can still flip through it toward its target geometry.
        self.use_planarity = dimensions == 3
        # Target sign per chair ring (set before the 3D stage).
        self.chair_signs: list[float] = []
        self.contact_pairs: list[tuple[int, int, float]] = []
        # Pair terms as (flat index i, flat index j, lower^2, upper^2, weight);
        # contacts have no upper bound.
        self._bounded_terms = [
            (i * dimensions, j * dimensions, lower * lower, upper * upper, weight)
            for i, j, lower, upper, weight in problem.near_pairs
        ]
        self._terms = list(self._bounded_terms)

    def refresh_contacts(self, positions) -> None:
        """Track unbounded pairs closer than their contact distance plus a margin."""
        problem = self.problem
        cutoff_margin = 1.5
        cell = 2.0 * max(problem.contact) + cutoff_margin
        grid: dict[tuple[int, int, int], list[int]] = {}
        for index, position in enumerate(positions):
            key = (int(math.floor(position[0] / cell)), int(math.floor(position[1] / cell)),
                   int(math.floor(position[2] / cell)))
            grid.setdefault(key, []).append(index)
        pairs = []
        for (gx, gy, gz), members in grid.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        others = grid.get((gx + dx, gy + dy, gz + dz))
                        if not others:
                            continue
                        for i in members:
                            for j in others:
                                if j <= i or (i, j) in problem.bounded:
                                    continue
                                lower = problem.contact[i] + problem.contact[j]
                                if math.dist(positions[i][:3], positions[j][:3]) < lower + cutoff_margin:
                                    pairs.append((i, j, lower))
        self.contact_pairs = pairs
        dims = self.dimensions
        self._terms = self._bounded_terms + [
            (i * dims, j * dims, lower * lower, math.inf, 0.5) for i, j, lower in pairs
        ]

    def __call__(self, flat: list[float]) -> tuple[float, list[float]]:
        dims = self.dimensions
        count = self.problem.count
        gradient = [0.0] * len(flat)
        energy = 0.0
        # Flat-bottom bound error: (d^2/u^2 - 1)^2 above the upper bound,
        # (2 l^2 / (l^2 + d^2) - 1)^2 below the lower bound. Written out per
        # dimension count because this loop dominates the run time.
        if dims == 4:
            for bi, bj, l2, u2, weight in self._terms:
                dx = flat[bi] - flat[bj]
                dy = flat[bi + 1] - flat[bj + 1]
                dz = flat[bi + 2] - flat[bj + 2]
                dw = flat[bi + 3] - flat[bj + 3]
                d2 = dx * dx + dy * dy + dz * dz + dw * dw
                if d2 > u2:
                    term = d2 / u2 - 1.0
                    scale = weight * 4.0 * term / u2
                elif d2 < l2:
                    denominator = l2 + d2
                    term = 2.0 * l2 / denominator - 1.0
                    scale = -weight * 8.0 * term * l2 / (denominator * denominator)
                else:
                    continue
                energy += weight * term * term
                gradient[bi] += scale * dx
                gradient[bj] -= scale * dx
                gradient[bi + 1] += scale * dy
                gradient[bj + 1] -= scale * dy
                gradient[bi + 2] += scale * dz
                gradient[bj + 2] -= scale * dz
                gradient[bi + 3] += scale * dw
                gradient[bj + 3] -= scale * dw
        else:
            for bi, bj, l2, u2, weight in self._terms:
                dx = flat[bi] - flat[bj]
                dy = flat[bi + 1] - flat[bj + 1]
                dz = flat[bi + 2] - flat[bj + 2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > u2:
                    term = d2 / u2 - 1.0
                    scale = weight * 4.0 * term / u2
                elif d2 < l2:
                    denominator = l2 + d2
                    term = 2.0 * l2 / denominator - 1.0
                    scale = -weight * 8.0 * term * l2 / (denominator * denominator)
                else:
                    continue
                energy += weight * term * term
                gradient[bi] += scale * dx
                gradient[bj] -= scale * dx
                gradient[bi + 1] += scale * dy
                gradient[bj + 1] -= scale * dy
                gradient[bi + 2] += scale * dz
                gradient[bj + 2] -= scale * dz

        positions = [flat[index * dims:index * dims + 3] for index in range(count)]
        for center, a, b, c, sign in self.problem.chiral:
            volume = sign * _volume(positions, center, a, b, c)
            if volume < _CHIRAL_VOLUME_MIN:
                delta = volume - _CHIRAL_VOLUME_MIN
            elif volume > _CHIRAL_VOLUME_MAX:
                delta = volume - _CHIRAL_VOLUME_MAX
            else:
                continue
            energy += delta * delta
            ga, gb, gc = _volume_gradient(positions, center, a, b, c)
            factor = 2.0 * delta * sign
            for atom, g in ((a, ga), (b, gb), (c, gc)):
                for k in range(3):
                    gradient[atom * dims + k] += factor * g[k]
                    gradient[center * dims + k] -= factor * g[k]

        for center, a, b, c in (self.problem.planar if self.use_planarity else ()):
            volume = _volume(positions, center, a, b, c)
            energy += 0.5 * volume * volume
            ga, gb, gc = _volume_gradient(positions, center, a, b, c)
            factor = volume
            for atom, g in ((a, ga), (b, gb), (c, gc)):
                for k in range(3):
                    gradient[atom * dims + k] += factor * g[k]
                    gradient[center * dims + k] -= factor * g[k]

        if self.use_planarity and self.chair_signs:
            for torsions, ring_sign in zip(self.problem.chair_rings, self.chair_signs):
                for offset, torsion in enumerate(torsions):
                    angle, gradients = _dihedral_and_gradient(positions, *torsion)
                    # Signed so the target window is always positive.
                    sign = ring_sign * (-1.0 if offset % 2 else 1.0)
                    signed = sign * angle
                    if signed < _CHAIR_MIN:
                        delta = signed - _CHAIR_MIN
                    elif signed > _CHAIR_MAX:
                        delta = signed - _CHAIR_MAX
                    else:
                        continue
                    energy += 2.0 * delta * delta
                    factor = 4.0 * delta * sign
                    for atom, g in zip(torsion, gradients):
                        for n in range(3):
                            gradient[atom * dims + n] += factor * g[n]

        for center, first, second, fraction in self.problem.linear:
            # Target: centre = fraction * first + (1 - fraction) * second.
            bc, bf, bs = center * dims, first * dims, second * dims
            for k in range(dims):
                delta = flat[bc + k] - (fraction * flat[bf + k] + (1.0 - fraction) * flat[bs + k])
                energy += 4.0 * delta * delta
                g = 8.0 * delta
                gradient[bc + k] += g
                gradient[bf + k] -= g * fraction
                gradient[bs + k] -= g * (1.0 - fraction)

        if dims == 4 and self.fourth_weight:
            for index in range(count):
                w = flat[index * 4 + 3]
                energy += self.fourth_weight * w * w
                gradient[index * 4 + 3] += 2.0 * self.fourth_weight * w
        return energy, gradient


def _dot(a: list[float], b: list[float]) -> float:
    return sum(map(operator.mul, a, b))


def _lbfgs(objective, x: list[float], iterations: int, history: int = 8, tolerance: float = 1e-6):
    """Plain L-BFGS with Armijo backtracking; returns (x, energy)."""
    energy, gradient = objective(x)
    s_list: list[list[float]] = []
    y_list: list[list[float]] = []
    stalled = 0
    for _iteration in range(iterations):
        if energy < 1e-9 or max(abs(value) for value in gradient) < tolerance:
            break
        # Stop once the error has effectively stopped decreasing.
        if stalled >= 4:
            break
        q = list(gradient)
        alphas = []
        for s, y in reversed(list(zip(s_list, y_list))):
            rho = 1.0 / max(_dot(y, s), 1e-12)
            alpha = rho * _dot(s, q)
            alphas.append((rho, alpha))
            q = [qi - alpha * yi for qi, yi in zip(q, y)]
        if s_list:
            s, y = s_list[-1], y_list[-1]
            gamma = _dot(s, y) / max(_dot(y, y), 1e-12)
        else:
            gamma = 0.1 / max(max(abs(value) for value in gradient), 1e-12)
        direction = [gamma * value for value in q]
        for (rho, alpha), s, y in zip(reversed(alphas), s_list, y_list):
            beta = rho * _dot(y, direction)
            direction = [di + si * (alpha - beta) for di, si in zip(direction, s)]
        direction = [-value for value in direction]
        slope = _dot(gradient, direction)
        if slope >= 0.0:
            s_list.clear()
            y_list.clear()
            direction = [-value for value in gradient]
            slope = -_dot(gradient, gradient)
        # Cap the largest atom move per step.
        largest = max(abs(value) for value in direction)
        step = min(1.0, 0.3 / largest) if largest > 0 else 1.0
        while True:
            candidate = [xi + step * di for xi, di in zip(x, direction)]
            new_energy, new_gradient = objective(candidate)
            if new_energy <= energy + 1e-4 * step * slope or step < 1e-8:
                break
            step *= 0.5
        if step < 1e-8:
            break
        s = [a - b for a, b in zip(candidate, x)]
        y = [a - b for a, b in zip(new_gradient, gradient)]
        if _dot(s, y) > 1e-10:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > history:
                s_list.pop(0)
                y_list.pop(0)
        stalled = stalled + 1 if energy - new_energy < 1e-6 * max(energy, 1e-3) else 0
        x, energy, gradient = candidate, new_energy, new_gradient
    return x, energy


# Saturated six-membered ring dihedrals for a chair (radians); the
# acceptance window is a little wider than the target window.
_CHAIR_MIN, _CHAIR_MAX = math.radians(45.0), math.radians(70.0)
_CHAIR_ACCEPT_MIN, _CHAIR_ACCEPT_MAX = math.radians(38.0), math.radians(78.0)


def _dihedral_and_gradient(positions, i, j, k, l):
    """Dihedral angle i-j-k-l (radians) and its gradient for each atom."""
    pi, pj, pk, pl = positions[i], positions[j], positions[k], positions[l]
    f = (pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2])
    g = (pj[0] - pk[0], pj[1] - pk[1], pj[2] - pk[2])
    h = (pl[0] - pk[0], pl[1] - pk[1], pl[2] - pk[2])

    def cross(p, q):
        return (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])

    def dot(p, q):
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]

    a = cross(f, g)
    b = cross(h, g)
    a2, b2 = max(dot(a, a), 1e-12), max(dot(b, b), 1e-12)
    g_length = math.sqrt(max(dot(g, g), 1e-12))
    angle = math.atan2(dot(cross(b, a), g) / g_length, dot(a, b))
    fg, hg = dot(f, g), dot(h, g)
    grad_i = tuple(-g_length / a2 * value for value in a)
    grad_l = tuple(g_length / b2 * value for value in b)
    grad_j = tuple(g_length / a2 * a[n] + fg / (a2 * g_length) * a[n] - hg / (b2 * g_length) * b[n]
                   for n in range(3))
    grad_k = tuple(-g_length / b2 * b[n] - fg / (a2 * g_length) * a[n] + hg / (b2 * g_length) * b[n]
                   for n in range(3))
    return angle, (grad_i, grad_j, grad_k, grad_l)


def _chair_signs(problem: _Problem, positions) -> list[float]:
    """Alternating target signs per chair ring, anchored on its first torsion."""
    signs = []
    for torsions in problem.chair_rings:
        angle, _gradient = _dihedral_and_gradient(positions, *torsions[0])
        signs.append(1.0 if angle >= 0.0 else -1.0)
    return signs


def _chairs_ok(problem: _Problem, positions) -> bool:
    for torsions in problem.chair_rings:
        first_sign = None
        for offset, torsion in enumerate(torsions):
            angle, _gradient = _dihedral_and_gradient(positions, *torsion)
            if not _CHAIR_ACCEPT_MIN <= abs(angle) <= _CHAIR_ACCEPT_MAX:
                return False
            sign = (1.0 if angle >= 0.0 else -1.0) * (-1.0 if offset % 2 else 1.0)
            if first_sign is None:
                first_sign = sign
            elif sign != first_sign:
                return False
    return True


def _chirality_ok(problem: _Problem, positions) -> bool:
    return all(sign * _volume(positions, center, a, b, c) > 0.3
               for center, a, b, c, sign in problem.chiral)


def _worst_violation(problem: _Problem, positions) -> float:
    """Largest distance-bound violation (angstroms) among weighted pairs.

    """
    worst = 0.0
    for i, j, lower, upper, weight in problem.near_pairs:
        if weight < 1.0:
            continue
        distance = math.dist(positions[i], positions[j])
        worst = max(worst, lower - distance, distance - upper)
    return worst


# An attempt is accepted when every stereo constraint holds and no bond,
# angle or fixed 1-4 distance is off by more than this.
_ACCEPTED_VIOLATION = 0.2


def embed(
    elements: list[str],
    bonds: list[tuple[int, int, int]],
    tetrahedral: list[Tetrahedral] = (),
    double_stereo: list[DoubleBondStereo] = (),
    initial: list[tuple[float, float, float]] | None = None,
    attempts: int = 6,
    seed: int = 20261103,
) -> list[tuple[float, float, float]]:
    """Return 3D coordinates (angstroms, centred at the origin) for the graph."""
    count = len(elements)
    if count == 0:
        return []
    if count == 1:
        return [(0.0, 0.0, 0.0)]
    problem = _Problem(elements, bonds, list(tetrahedral), list(double_stereo))
    generator = random.Random(seed)
    best = None
    for attempt in range(attempts):
        if attempt == 0 and initial is not None:
            start = [[p[0], p[1], p[2] + generator.uniform(-0.6, 0.6), generator.uniform(-0.8, 0.8)]
                     for p in initial]
        else:
            box = 1.6 * count ** (1.0 / 3.0) + 1.0
            start = [[generator.uniform(-box, box) for _ in range(4)] for _ in range(count)]
        flat = [value for position in start for value in position]
        objective = _Objective(problem, 4)
        for round_index in range(4):
            objective.refresh_contacts([flat[i * 4:i * 4 + 4] for i in range(count)])
            flat, _energy = _lbfgs(objective, flat, 120)
        # Squeeze the fourth dimension out.
        for weight in (0.2, 1.0, 5.0):
            objective.fourth_weight = weight
            objective.refresh_contacts([flat[i * 4:i * 4 + 4] for i in range(count)])
            flat, _energy = _lbfgs(objective, flat, 80)
        positions3 = [flat[i * 4:i * 4 + 3] for i in range(count)]
        flat3 = [value for position in positions3 for value in position]
        objective3 = _Objective(problem, 3)
        objective3.chair_signs = _chair_signs(problem, positions3)
        for _round in range(3):
            objective3.refresh_contacts([flat3[i * 3:i * 3 + 3] for i in range(count)])
            flat3, energy = _lbfgs(objective3, flat3, 150)
        positions = [flat3[i * 3:i * 3 + 3] for i in range(count)]
        violation = _worst_violation(problem, positions)
        ok = (_chirality_ok(problem, positions) and violation < _ACCEPTED_VIOLATION
              and _chairs_ok(problem, positions))
        score = (0 if ok else 1, violation, energy)
        if best is None or score < best[0]:
            best = (score, positions)
        if ok:
            break
    positions = best[1]
    centre = [sum(p[k] for p in positions) / count for k in range(3)]
    return [tuple(p[k] - centre[k] for k in range(3)) for p in positions]
