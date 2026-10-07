use numpy::{
    ndarray::{self, Array2},
    IntoPyArray, PyArray2, PyReadonlyArray2,
};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn extract_cdch<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    angle_step: usize,
    center_x: Option<f64>,
    center_y: Option<f64>,
    min_val: u8,
    max_val: u8,
    bins: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = image.as_array();
    let (h, w) = (array.shape()[0], array.shape()[1]);

    validate_args(angle_step, center_x, center_y)?;
    let (cx, cy) = calculate_center(&array, center_x, center_y, w, h, min_val, max_val);
    let angles_deg = generate_angles(angle_step);
    let ray_hists =
        compute_ray_histograms(&array, cx, cy, &angles_deg, min_val, max_val, bins, w, h);
    let out_array = build_output_matrix(&ray_hists, bins);

    Ok(out_array.into_pyarray(py))
}

fn validate_args(angle_step: usize, center_x: Option<f64>, center_y: Option<f64>) -> PyResult<()> {
    assert!(
        0 < angle_step && angle_step < 360,
        "Angle step must be between 0 and 360."
    );
    if let Some(cx) = center_x {
        assert!(0.0 <= cx && cx <= 1.0, "center_x must be between 0 and 1");
    }
    if let Some(cy) = center_y {
        assert!(0.0 <= cy && cy <= 1.0, "center_y must be between 0 and 1");
    }
    Ok(())
}

/// Converts the relative coordinates of the center to the absolute
fn calculate_center(
    array: &ndarray::ArrayView2<u8>,
    center_x: Option<f64>,
    center_y: Option<f64>,
    w: usize,
    h: usize,
    min_val: u8,
    max_val: u8,
) -> (i32, i32) {
    if let (Some(x), Some(y)) = (center_x, center_y) {
        // if the center is given
        ((x * w as f64).round() as i32, (y * h as f64).round() as i32)
    } else {
        // uses center of mass if the center is None
        let (mut sum_x, mut sum_y, mut count) = (0usize, 0usize, 0usize);
        for ((y, x), &p) in array.indexed_iter() {
            if p >= min_val && p <= max_val {
                sum_x += x;
                sum_y += y;
                count += 1;
            }
        }
        if count > 0 {
            ((sum_x / count) as i32, (sum_y / count) as i32)
        } else {
            ((w / 2) as i32, (h / 2) as i32)
        }
    }
}

fn generate_angles(angle_step: usize) -> Vec<f64> {
    (0..360)
        .step_by(angle_step)
        .map(|a| a as f64 - 90.0)
        .collect()
}

/// Calculates the histograms along each ray
fn compute_ray_histograms(
    array: &ndarray::ArrayView2<u8>,
    cx: i32,
    cy: i32,
    angles_deg: &[f64],
    min_val: u8,
    max_val: u8,
    bins: usize,
    w: usize,
    h: usize,
) -> Vec<Vec<f64>> {
    let max_radius = (w as f64).hypot(h as f64) as i32;
    let range = (max_val as usize).saturating_sub(min_val as usize) + 1;

    angles_deg
        .into_par_iter()
        .map(|&angle_deg| {
            let rad = angle_deg.to_radians();
            let (cos_a, sin_a) = (rad.cos(), rad.sin());
            let mut counts = vec![0usize; bins];
            let mut total_valid = 0usize;

            for r in 0..max_radius {
                let px = (cx as f64 + r as f64 * cos_a).round() as i32;
                let py = (cy as f64 + r as f64 * sin_a).round() as i32;

                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    let p = array[[py as usize, px as usize]];
                    if p >= min_val && p <= max_val {
                        let bin_idx = (((p - min_val) as usize * bins) / range).min(bins - 1);
                        counts[bin_idx] += 1;
                        total_valid += 1;
                    }
                }
            }

            let mut normalized = vec![0.0; bins];
            if total_valid > 0 {
                let mut running_sum = 0usize;
                for b in 0..bins {
                    running_sum += counts[b];
                    normalized[b] = running_sum as f64 / total_valid as f64;
                }
            }
            normalized
        })
        .collect()
}

fn build_output_matrix(ray_hists: &[Vec<f64>], bins: usize) -> Array2<f64> {
    let num_rays = ray_hists.len();
    let mut out_array = Array2::<f64>::zeros((bins, num_rays));
    for i in 0..num_rays {
        for j in 0..bins {
            out_array[[bins - 1 - j, i]] = ray_hists[i][j];
        }
    }
    out_array
}
