use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array1, Array2};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

#[allow(dead_code)]
pub fn split_data(
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(42);
    split_data_rng(x, y, &mut rng)
}

#[allow(dead_code)]
pub fn split_data_rng<R: Rng + ?Sized>(
    x: &Array2<f64>,
    y: &Array1<f64>,
    rng: &mut R,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let n = y.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.as_mut_slice().shuffle(rng);
    let split = (n as f64 * 0.8) as usize;
    let train_idx = &indices[..split];
    let test_idx = &indices[split..];
    (
        x.select(ndarray::Axis(0), train_idx),
        y.select(ndarray::Axis(0), train_idx),
        x.select(ndarray::Axis(0), test_idx),
        y.select(ndarray::Axis(0), test_idx),
    )
}

#[allow(dead_code)]
pub fn setup_data_csv(path: &str) -> (Array2<f64>, Array1<f64>) {
    // Reads features + target from a CSV at `path` (last column = target y; no header).
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .expect("Failed to open file");

    let mut x_data_: Vec<Vec<f64>> = Vec::new();
    let mut y_data = Vec::new();

    // Skip the header row since we're using has_headers(true)
    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let mut x_row = Vec::new();

        // Parse all columns except the last one as features
        for i in 0..record.len() - 1 {
            let val = record[i].parse::<f64>().unwrap();
            x_row.push(val);
        }
        // Parse the last column as the target
        let y_val = record[record.len() - 1].parse::<f64>().unwrap();
        x_data_.push(x_row);
        y_data.push(y_val);
    }

    let num_features = if let Some(first_row) = x_data_.first() {
        first_row.len()
    } else {
        0
    };

    let x_data: Vec<f64> = x_data_.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((x_data.len() / num_features, num_features), x_data)
        .expect("Failed to create Array2 for x");
    let y = Array1::from(y_data);
    (x, y)
}

#[allow(dead_code)]
pub fn save_x_y(
    x: &Array2<f64>,
    y: &Array1<f64>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create CSV writer
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path(output_path)?;

    // Write each row: x features followed by y value
    for i in 0..x.nrows() {
        let mut record = Vec::new();

        // Add all features from x
        for j in 0..x.ncols() {
            record.push(x[[i, j]].to_string());
        }

        // Add y value
        record.push(y[i].to_string());

        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}
