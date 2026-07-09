/// Quantization scheme used by a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantizationScheme {
    None,
    AffineAsymmetric,
    Symmetric,
    DynamicFixedPoint,
}

/// Per-tensor or per-axis quantization metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantization {
    pub scheme: QuantizationScheme,
    pub scale: Vec<f32>,
    pub zero_point: Vec<i32>,
    pub axis: Option<usize>,
}

impl Quantization {
    pub fn none() -> Self {
        Self {
            scheme: QuantizationScheme::None,
            scale: Vec::new(),
            zero_point: Vec::new(),
            axis: None,
        }
    }
}

impl Default for Quantization {
    fn default() -> Self {
        Self::none()
    }
}
