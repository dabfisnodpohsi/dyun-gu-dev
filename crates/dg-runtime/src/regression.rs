use dg_core::{CpuDevice, DataFormat, DataType, DeviceKind, Shape, Tensor, TensorDesc};
use serde::Deserialize;
use thiserror::Error;

use crate::{Error as RuntimeError, Runtime};

/// A fixed floating-point tensor used by [`RegressionCase`].
#[derive(Clone, Debug, Deserialize)]
pub struct RegressionTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Per-case numerical acceptance criteria.
#[derive(Clone, Debug, Deserialize)]
pub struct RegressionTolerance {
    pub absolute: f32,
    pub relative: f32,
    pub cosine: Option<f32>,
}

/// A deterministic input/output numerical regression case.
#[derive(Clone, Debug, Deserialize)]
pub struct RegressionCase {
    pub name: String,
    pub inputs: Vec<RegressionTensor>,
    pub expected_outputs: Vec<RegressionTensor>,
    pub tolerance: RegressionTolerance,
}

impl RegressionCase {
    /// Loads a regression case from its small JSON fixture representation.
    pub fn from_json(json: &str) -> Result<Self, RegressionError> {
        serde_json::from_str(json)
            .map_err(|error| RegressionError::Invalid(format!("invalid fixture: {error}")))
    }
}

/// Measured numerical error for a successful regression run.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionReport {
    pub case: String,
    pub max_absolute_error: f32,
    pub max_relative_error: f32,
    pub minimum_cosine_similarity: f32,
}

/// Errors produced while loading or comparing a regression case.
#[derive(Debug, Error)]
pub enum RegressionError {
    #[error("invalid regression case: {0}")]
    Invalid(String),
    #[error("regression backend failed: {0}")]
    Backend(#[from] RuntimeError),
    #[error(
        "regression case {case} output {output} failed at index {index}: \
         expected {expected}, actual {actual}, absolute_error={absolute_error}, \
         relative_error={relative_error}, cosine_similarity={cosine_similarity}"
    )]
    Mismatch {
        case: String,
        output: usize,
        index: usize,
        expected: f32,
        actual: f32,
        absolute_error: f32,
        relative_error: f32,
        cosine_similarity: f32,
    },
}

/// Runs fixed cases through a runtime and checks numerical closeness.
pub struct RegressionHarness;

impl RegressionHarness {
    pub fn run(
        runtime: &mut Runtime,
        case: &RegressionCase,
    ) -> Result<RegressionReport, RegressionError> {
        validate_case(case)?;
        let device = CpuDevice::new();
        let mut inputs = Vec::with_capacity(case.inputs.len());
        for input in &case.inputs {
            inputs.push(make_tensor(input, &device)?);
        }
        let outputs = runtime.run(&inputs)?;
        if outputs.len() != case.expected_outputs.len() {
            return Err(RegressionError::Invalid(format!(
                "case {} expected {} outputs, backend returned {}",
                case.name,
                case.expected_outputs.len(),
                outputs.len()
            )));
        }

        let mut max_absolute_error: f32 = 0.0;
        let mut max_relative_error: f32 = 0.0;
        let mut minimum_cosine_similarity: f32 = 1.0;
        for (output_index, (actual, expected)) in
            outputs.iter().zip(case.expected_outputs.iter()).enumerate()
        {
            let actual_values = read_f32(actual, &case.name, output_index)?;
            if actual.desc().shape().dims() != expected.shape.as_slice() {
                return Err(RegressionError::Invalid(format!(
                    "case {} output {} shape mismatch: expected {:?}, actual {:?}",
                    case.name,
                    output_index,
                    expected.shape,
                    actual.desc().shape().dims()
                )));
            }
            let cosine_similarity = cosine(&actual_values, &expected.values).ok_or_else(|| {
                RegressionError::Invalid(format!(
                    "case {} output {} has no valid cosine similarity",
                    case.name, output_index
                ))
            })?;
            minimum_cosine_similarity = minimum_cosine_similarity.min(cosine_similarity);
            for (index, (&actual_value, &expected_value)) in
                actual_values.iter().zip(expected.values.iter()).enumerate()
            {
                if !actual_value.is_finite() || !expected_value.is_finite() {
                    return Err(RegressionError::Mismatch {
                        case: case.name.clone(),
                        output: output_index,
                        index,
                        expected: expected_value,
                        actual: actual_value,
                        absolute_error: f32::INFINITY,
                        relative_error: f32::INFINITY,
                        cosine_similarity,
                    });
                }
                let absolute_error = (actual_value - expected_value).abs();
                let relative_error = absolute_error / expected_value.abs().max(f32::MIN_POSITIVE);
                max_absolute_error = max_absolute_error.max(absolute_error);
                max_relative_error = max_relative_error.max(relative_error);
                let within_tolerance = absolute_error <= case.tolerance.absolute
                    || relative_error <= case.tolerance.relative;
                let cosine_ok = case
                    .tolerance
                    .cosine
                    .is_none_or(|threshold| cosine_similarity >= threshold);
                if !within_tolerance || !cosine_ok {
                    return Err(RegressionError::Mismatch {
                        case: case.name.clone(),
                        output: output_index,
                        index,
                        expected: expected_value,
                        actual: actual_value,
                        absolute_error,
                        relative_error,
                        cosine_similarity,
                    });
                }
            }
        }
        Ok(RegressionReport {
            case: case.name.clone(),
            max_absolute_error,
            max_relative_error,
            minimum_cosine_similarity,
        })
    }
}

fn validate_case(case: &RegressionCase) -> Result<(), RegressionError> {
    if case.inputs.is_empty() || case.expected_outputs.is_empty() {
        return Err(RegressionError::Invalid(
            "a regression case requires inputs and expected outputs".to_string(),
        ));
    }
    if !case.tolerance.absolute.is_finite()
        || !case.tolerance.relative.is_finite()
        || case.tolerance.absolute < 0.0
        || case.tolerance.relative < 0.0
    {
        return Err(RegressionError::Invalid(
            "absolute and relative tolerances must be finite and non-negative".to_string(),
        ));
    }
    if let Some(cosine) = case.tolerance.cosine {
        if !cosine.is_finite() || !(-1.0..=1.0).contains(&cosine) {
            return Err(RegressionError::Invalid(
                "cosine tolerance must be finite and between -1 and 1".to_string(),
            ));
        }
    }
    for tensor in case.inputs.iter().chain(case.expected_outputs.iter()) {
        let elements = tensor
            .shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
        if elements != Some(tensor.values.len()) {
            return Err(RegressionError::Invalid(format!(
                "shape {:?} does not contain {} values",
                tensor.shape,
                tensor.values.len()
            )));
        }
    }
    Ok(())
}

fn make_tensor(fixture: &RegressionTensor, device: &CpuDevice) -> Result<Tensor, RegressionError> {
    let desc = TensorDesc::new(
        Shape::new(fixture.shape.clone()),
        DataType::F32,
        DataFormat::NC,
        DeviceKind::Cpu,
    );
    let tensor = Tensor::allocate(device, desc).map_err(RuntimeError::from)?;
    let bytes = fixture
        .values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    tensor
        .buffer()
        .write_from_slice(&bytes)
        .map_err(RuntimeError::from)?;
    Ok(tensor)
}

fn read_f32(tensor: &Tensor, case: &str, output: usize) -> Result<Vec<f32>, RegressionError> {
    let bytes = tensor.buffer().read_bytes();
    let chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
    if !chunks.remainder().is_empty() {
        return Err(RegressionError::Invalid(format!(
            "case {case} output {output} has a non-f32 byte length: {}",
            bytes.len()
        )));
    }
    Ok(chunks
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn cosine(actual: &[f32], expected: &[f32]) -> Option<f32> {
    if actual.len() != expected.len() || actual.is_empty() {
        return None;
    }
    let mut dot = 0.0;
    let mut actual_norm = 0.0;
    let mut expected_norm = 0.0;
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        if !actual.is_finite() || !expected.is_finite() {
            return None;
        }
        dot += actual * expected;
        actual_norm += actual * actual;
        expected_norm += expected * expected;
    }
    let denominator = actual_norm.sqrt() * expected_norm.sqrt();
    if denominator == 0.0 {
        (actual == expected).then_some(1.0)
    } else {
        Some(dot / denominator)
    }
}
