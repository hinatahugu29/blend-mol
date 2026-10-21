use std::slice;
use std::thread;

#[unsafe(no_mangle)]
pub extern "C" fn blend_mol_uff_abi_version() -> u32 {
    1
}

struct PartialResult {
    energy: f64,
    gradient: Vec<f64>,
}

fn evaluate_range(
    atom_count: usize,
    coordinates: &[f64],
    pair_atoms: &[u32],
    pair_parameters: &[f64],
    start: usize,
    end: usize,
) -> PartialResult {
    let mut result = PartialResult {
        energy: 0.0,
        gradient: vec![0.0; atom_count * 3],
    };
    for pair_index in start..end {
        let first = pair_atoms[pair_index * 2] as usize;
        let second = pair_atoms[pair_index * 2 + 1] as usize;
        let depth = pair_parameters[pair_index * 2];
        let equilibrium = pair_parameters[pair_index * 2 + 1];
        let first_offset = first * 3;
        let second_offset = second * 3;
        let delta = [
            coordinates[first_offset] - coordinates[second_offset],
            coordinates[first_offset + 1] - coordinates[second_offset + 1],
            coordinates[first_offset + 2] - coordinates[second_offset + 2],
        ];
        let distance_squared = delta.iter().map(|value| value * value).sum::<f64>();
        if distance_squared > 100.0 {
            continue;
        }
        let equilibrium_sixth = equilibrium.powi(6);
        let observed_sixth = (distance_squared.powi(3)).max(1.0e-6);
        result.energy += depth
            * (equilibrium_sixth.powi(2) / observed_sixth.powi(2)
                - 2.0 * equilibrium_sixth / observed_sixth);
        if distance_squared.powi(3) <= 1.0e-6 {
            continue;
        }
        let distance = distance_squared.sqrt();
        let derivative = 12.0
            * depth
            * (equilibrium_sixth / distance.powi(7)
                - equilibrium_sixth.powi(2) / distance.powi(13))
            / distance;
        for axis in 0..3 {
            let component = derivative * delta[axis];
            result.gradient[first_offset + axis] += component;
            result.gradient[second_offset + axis] -= component;
        }
    }
    result
}

/// Evaluate UFF van der Waals energy and Cartesian gradient.
///
/// Return codes: 0 success, 1 invalid pointer, 2 invalid atom index or value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blend_mol_vdw_energy_gradient(
    atom_count: usize,
    coordinates: *const f64,
    pair_count: usize,
    pair_atoms: *const u32,
    pair_parameters: *const f64,
    out_energy: *mut f64,
    out_gradient: *mut f64,
) -> i32 {
    if coordinates.is_null()
        || out_energy.is_null()
        || out_gradient.is_null()
        || (pair_count > 0 && (pair_atoms.is_null() || pair_parameters.is_null()))
    {
        return 1;
    }
    let coordinates = unsafe { slice::from_raw_parts(coordinates, atom_count * 3) };
    let pair_atoms = unsafe { slice::from_raw_parts(pair_atoms, pair_count * 2) };
    let pair_parameters = unsafe { slice::from_raw_parts(pair_parameters, pair_count * 2) };
    if coordinates.iter().any(|value| !value.is_finite())
        || pair_parameters
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || pair_atoms.iter().any(|index| *index as usize >= atom_count)
    {
        return 2;
    }

    let available = thread::available_parallelism().map_or(1, |count| count.get());
    let worker_count = if pair_count < 512 {
        1
    } else {
        available.min(pair_count.div_ceil(256))
    };
    let mut parts = Vec::with_capacity(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let start = pair_count * worker / worker_count;
            let end = pair_count * (worker + 1) / worker_count;
            handles.push(scope.spawn(move || {
                evaluate_range(
                    atom_count,
                    coordinates,
                    pair_atoms,
                    pair_parameters,
                    start,
                    end,
                )
            }));
        }
        for handle in handles {
            parts.push(handle.join().expect("scoped UFF worker panicked"));
        }
    });

    let output_gradient = unsafe { slice::from_raw_parts_mut(out_gradient, atom_count * 3) };
    output_gradient.fill(0.0);
    let mut energy = 0.0;
    for part in parts {
        energy += part.energy;
        for (target, value) in output_gradient.iter_mut().zip(part.gradient) {
            *target += value;
        }
    }
    unsafe { *out_energy = energy };
    0
}

fn cross(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn point(coordinates: &[f64], index: usize) -> [f64; 3] {
    let offset = index * 3;
    [
        coordinates[offset],
        coordinates[offset + 1],
        coordinates[offset + 2],
    ]
}

fn subtract(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ]
}

fn residual_energy(
    coordinates: &[f64],
    torsion_atoms: &[u32],
    torsion_parameters: &[f64],
    inversion_atoms: &[u32],
    inversion_barriers: &[f64],
) -> Option<f64> {
    let mut energy = 0.0;
    for term in 0..torsion_parameters.len() / 3 {
        let first = point(coordinates, torsion_atoms[term * 4] as usize);
        let second = point(coordinates, torsion_atoms[term * 4 + 1] as usize);
        let third = point(coordinates, torsion_atoms[term * 4 + 2] as usize);
        let fourth = point(coordinates, torsion_atoms[term * 4 + 3] as usize);
        let first_bond = subtract(second, first);
        let center_bond = subtract(third, second);
        let last_bond = subtract(fourth, third);
        let first_normal = cross(first_bond, center_bond);
        let last_normal = cross(center_bond, last_bond);
        let first_norm = norm(first_normal);
        let last_norm = norm(last_normal);
        let center_norm = norm(center_bond);
        if first_norm.min(last_norm).min(center_norm) < 1.0e-10 {
            return None;
        }
        let cosine = first_normal
            .iter()
            .zip(last_normal)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (first_norm * last_norm);
        let normal_cross = cross(first_normal, last_normal);
        let sine = normal_cross
            .iter()
            .zip(center_bond)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (first_norm * last_norm * center_norm);
        let angle = sine.atan2(cosine.clamp(-1.0, 1.0));
        let periodicity = torsion_parameters[term * 3];
        let cosine_ideal = torsion_parameters[term * 3 + 1];
        let barrier = torsion_parameters[term * 3 + 2];
        energy += barrier * (1.0 - cosine_ideal * (periodicity * angle).cos());
    }
    for term in 0..inversion_barriers.len() {
        let center = point(coordinates, inversion_atoms[term * 4] as usize);
        let first = point(coordinates, inversion_atoms[term * 4 + 1] as usize);
        let second = point(coordinates, inversion_atoms[term * 4 + 2] as usize);
        let outer = point(coordinates, inversion_atoms[term * 4 + 3] as usize);
        let center_to_first = subtract(first, center);
        let center_to_second = subtract(second, center);
        let center_to_outer = subtract(outer, center);
        let normal = cross(center_to_first, center_to_second);
        let normal_length = norm(normal);
        let outer_length = norm(center_to_outer);
        if normal_length.min(outer_length) < 1.0e-10 {
            return None;
        }
        let sine = center_to_outer
            .iter()
            .zip(normal)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (normal_length * outer_length);
        let angle = sine.clamp(-1.0, 1.0).asin();
        energy += inversion_barriers[term] * (1.0 - angle.cos());
    }
    Some(energy)
}

/// Evaluate torsion/inversion energy and their forward-difference gradient.
/// Independent coordinate perturbations are distributed across CPU threads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blend_mol_residual_energy_gradient(
    atom_count: usize,
    coordinates: *const f64,
    torsion_count: usize,
    torsion_atoms: *const u32,
    torsion_parameters: *const f64,
    inversion_count: usize,
    inversion_atoms: *const u32,
    inversion_barriers: *const f64,
    active_count: usize,
    active_atoms: *const u32,
    finite_difference: f64,
    out_energy: *mut f64,
    out_gradient: *mut f64,
) -> i32 {
    if coordinates.is_null()
        || out_energy.is_null()
        || out_gradient.is_null()
        || (torsion_count > 0 && (torsion_atoms.is_null() || torsion_parameters.is_null()))
        || (inversion_count > 0 && (inversion_atoms.is_null() || inversion_barriers.is_null()))
        || (active_count > 0 && active_atoms.is_null())
        || !finite_difference.is_finite()
        || finite_difference <= 0.0
    {
        return 1;
    }
    let coordinates = unsafe { slice::from_raw_parts(coordinates, atom_count * 3) };
    // C callers commonly represent an empty array with a null pointer. Rust's
    // from_raw_parts still requires a non-null pointer when the length is zero.
    let torsion_atoms = if torsion_count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(torsion_atoms, torsion_count * 4) }
    };
    let torsion_parameters = if torsion_count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(torsion_parameters, torsion_count * 3) }
    };
    let inversion_atoms = if inversion_count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(inversion_atoms, inversion_count * 4) }
    };
    let inversion_barriers = if inversion_count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(inversion_barriers, inversion_count) }
    };
    let active_atoms = if active_count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(active_atoms, active_count) }
    };
    if coordinates.iter().any(|value| !value.is_finite())
        || torsion_atoms
            .iter()
            .chain(inversion_atoms)
            .chain(active_atoms)
            .any(|index| *index as usize >= atom_count)
    {
        return 2;
    }
    let Some(base_energy) = residual_energy(
        coordinates,
        torsion_atoms,
        torsion_parameters,
        inversion_atoms,
        inversion_barriers,
    ) else {
        return 3;
    };
    let tasks: Vec<usize> = active_atoms
        .iter()
        .flat_map(|atom| {
            let offset = *atom as usize * 3;
            [offset, offset + 1, offset + 2]
        })
        .collect();
    let available = thread::available_parallelism().map_or(1, |count| count.get());
    let worker_count = if tasks.len() < 12 {
        1
    } else {
        available.min(tasks.len())
    };
    let mut results: Vec<Vec<(usize, f64)>> = Vec::with_capacity(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let start = tasks.len() * worker / worker_count;
            let end = tasks.len() * (worker + 1) / worker_count;
            let task_slice = &tasks[start..end];
            handles.push(scope.spawn(move || {
                let mut perturbed = coordinates.to_vec();
                let mut values = Vec::with_capacity(task_slice.len());
                for coordinate_index in task_slice {
                    perturbed[*coordinate_index] += finite_difference;
                    let higher = residual_energy(
                        &perturbed,
                        torsion_atoms,
                        torsion_parameters,
                        inversion_atoms,
                        inversion_barriers,
                    );
                    perturbed[*coordinate_index] -= finite_difference;
                    let Some(higher) = higher else { return None };
                    values.push((
                        *coordinate_index,
                        (higher - base_energy) / finite_difference,
                    ));
                }
                Some(values)
            }));
        }
        for handle in handles {
            let Some(values) = handle.join().expect("scoped residual worker panicked") else {
                return;
            };
            results.push(values);
        }
    });
    if results.len() != worker_count {
        return 3;
    }
    let output_gradient = unsafe { slice::from_raw_parts_mut(out_gradient, atom_count * 3) };
    output_gradient.fill(0.0);
    for worker_results in results {
        for (coordinate_index, value) in worker_results {
            output_gradient[coordinate_index] = value;
        }
    }
    unsafe { *out_energy = base_energy };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_atom_energy_has_equal_and_opposite_gradient() {
        let coordinates = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        let pairs = [0_u32, 1_u32];
        let parameters = [0.105_f64, 3.851_f64];
        let mut energy = 0.0;
        let mut gradient = [0.0; 6];
        let status = unsafe {
            blend_mol_vdw_energy_gradient(
                2,
                coordinates.as_ptr(),
                1,
                pairs.as_ptr(),
                parameters.as_ptr(),
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        assert!(energy < 0.0);
        assert!((gradient[0] + gradient[3]).abs() < 1.0e-12);
    }

    #[test]
    fn residual_batch_returns_finite_torsion_gradient() {
        let coordinates = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5, 1.0, 1.0];
        let torsion_atoms = [0_u32, 1, 2, 3];
        let torsion_parameters = [3.0_f64, -1.0, 1.0];
        let active = [0_u32, 1, 2, 3];
        let mut energy = 0.0;
        let mut gradient = [0.0; 12];
        let status = unsafe {
            blend_mol_residual_energy_gradient(
                4,
                coordinates.as_ptr(),
                1,
                torsion_atoms.as_ptr(),
                torsion_parameters.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                active.len(),
                active.as_ptr(),
                1.0e-4,
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        assert!(energy.is_finite());
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient.iter().any(|value| value.abs() > 1.0e-8));
    }

    #[test]
    fn rejects_out_of_range_atom_indices() {
        let coordinates = [0.0, 0.0, 0.0];
        let pairs = [0_u32, 7_u32];
        let parameters = [0.1_f64, 3.5_f64];
        let mut energy = 0.0;
        let mut gradient = [0.0; 3];
        let status = unsafe {
            blend_mol_vdw_energy_gradient(
                1,
                coordinates.as_ptr(),
                1,
                pairs.as_ptr(),
                parameters.as_ptr(),
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(status, 2);
    }

    #[test]
    fn rejects_degenerate_torsion_geometry() {
        let coordinates = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let torsion_atoms = [0_u32, 1, 2, 3];
        let torsion_parameters = [3.0_f64, -1.0, 1.0];
        let active = [0_u32];
        let mut energy = 0.0;
        let mut gradient = [0.0; 12];
        let status = unsafe {
            blend_mol_residual_energy_gradient(
                4,
                coordinates.as_ptr(),
                1,
                torsion_atoms.as_ptr(),
                torsion_parameters.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                active.len(),
                active.as_ptr(),
                1.0e-4,
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(status, 3);
    }
}
