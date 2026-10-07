use numpy::{
    ndarray::{self, Array2},
    IntoPyArray, PyArray2, PyReadonlyArray2,
};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn extract_rdch<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    ring_step: f64,
    center_x: Option<f64>,
    center_y: Option<f64>,
    min_val: u8,
    max_val: u8,
    bins: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = image.as_array();
    let (h, w) = (array.shape()[0], array.shape()[1]);

    validate_rdch_args(ring_step, bins)?;

    let (cx, cy) = calculate_center(&array, center_x, center_y, w, h, min_val, max_val);
    let max_radius = calculate_max_radius(cx, cy, w as f64, h as f64);
    let num_rings = ((max_radius / ring_step).ceil() as usize).max(1);

    let (ring_counts, ring_totals) =
        binning_pass(&array, cx, cy, ring_step, num_rings, min_val, max_val, bins);
    let hists = normalize_rings(&ring_counts, &ring_totals, num_rings, bins);
    let out_array = build_rdch_matrix(hists, num_rings, bins);

    Ok(out_array.into_pyarray(py))
}

fn validate_rdch_args(ring_step: f64, bins: usize) -> PyResult<()> {
    assert!(ring_step > 0.0, "ring_step must be > 0");
    assert!(bins > 0, "bins must be > 0");
    Ok(())
}

fn calculate_center(
    array: &ndarray::ArrayView2<u8>,
    center_x: Option<f64>,
    center_y: Option<f64>,
    w: usize,
    h: usize,
    min_val: u8,
    max_val: u8,
) -> (f64, f64) {
    if let (Some(x), Some(y)) = (center_x, center_y) {
        (x * w as f64, y * h as f64)
    } else {
        let (mut sum_x, mut sum_y, mut count) = (0usize, 0usize, 0usize);
        for ((y, x), &p) in array.indexed_iter() {
            if p >= min_val && p <= max_val {
                sum_x += x;
                sum_y += y;
                count += 1;
            }
        }
        if count > 0 {
            (sum_x as f64 / count as f64, sum_y as f64 / count as f64)
        } else {
            (w as f64 / 2.0, h as f64 / 2.0)
        }
    }
}

fn calculate_max_radius(cx: f64, cy: f64, w: f64, h: f64) -> f64 {
    let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)];
    corners
        .iter()
        .map(|&(kx, ky)| ((kx - cx).powi(2) + (ky - cy).powi(2)).sqrt())
        .fold(0.0, f64::max)
}

fn binning_pass(
    array: &ndarray::ArrayView2<u8>,
    cx: f64,
    cy: f64,
    ring_step: f64,
    num_rings: usize,
    min_val: u8,
    max_val: u8,
    bins: usize,
) -> (Vec<usize>, Vec<usize>) {
    let range = (max_val as usize).saturating_sub(min_val as usize) + 1;
    let mut ring_counts = vec![0usize; num_rings * bins];
    let mut ring_totals = vec![0usize; num_rings];

    for ((y, x), &p) in array.indexed_iter() {
        if p >= min_val && p <= max_val {
            let dist = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            let ring_idx = ((dist / ring_step) as usize).min(num_rings - 1);
            let bin_idx = (((p - min_val) as usize * bins) / range).min(bins - 1);

            ring_counts[ring_idx * bins + bin_idx] += 1;
            ring_totals[ring_idx] += 1;
        }
    }

    (ring_counts, ring_totals)
}

fn normalize_rings(
    ring_counts: &[usize],
    ring_totals: &[usize],
    num_rings: usize,
    bins: usize,
) -> Vec<Vec<f64>> {
    (0..num_rings)
        .into_par_iter()
        .map(|r_idx| {
            let mut normalized = vec![0.0; bins];
            let total = ring_totals[r_idx];

            if total > 0 {
                let mut running_sum = 0usize;
                let offset = r_idx * bins;
                for b in 0..bins {
                    running_sum += ring_counts[offset + b];
                    normalized[b] = running_sum as f64 / total as f64;
                }
            }
            normalized
        })
        .collect()
}

fn build_rdch_matrix(hists: Vec<Vec<f64>>, num_rings: usize, bins: usize) -> Array2<f64> {
    let mut out_array = Array2::<f64>::zeros((num_rings, bins));
    for (i, ring_data) in hists.into_iter().enumerate() {
        for (j, val) in ring_data.into_iter().enumerate() {
            out_array[[i, j]] = val;
        }
    }
    out_array
}
