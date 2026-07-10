use dg_core::{BBox, Detection};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub source_width: usize,
    pub source_height: usize,
    pub target_width: usize,
    pub target_height: usize,
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl Letterbox {
    pub fn new(
        source_width: usize,
        source_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Self, String> {
        if source_width == 0 || source_height == 0 || target_width == 0 || target_height == 0 {
            return Err("letterbox dimensions must be non-zero".to_string());
        }
        let source_width_f = usize_to_f32(source_width)?;
        let source_height_f = usize_to_f32(source_height)?;
        let target_width_f = usize_to_f32(target_width)?;
        let target_height_f = usize_to_f32(target_height)?;
        let scale = (target_width_f / source_width_f).min(target_height_f / source_height_f);
        let resized_width = source_width_f * scale;
        let resized_height = source_height_f * scale;
        Ok(Self {
            source_width,
            source_height,
            target_width,
            target_height,
            scale,
            pad_x: (target_width_f - resized_width) * 0.5,
            pad_y: (target_height_f - resized_height) * 0.5,
        })
    }

    pub fn map_to_source(&self, bbox: BBox) -> BBox {
        let source_width = dimension_as_f32(self.source_width);
        let source_height = dimension_as_f32(self.source_height);
        let x = ((bbox.x - self.pad_x) / self.scale).clamp(0.0, source_width);
        let y = ((bbox.y - self.pad_y) / self.scale).clamp(0.0, source_height);
        let right = ((bbox.x + bbox.w - self.pad_x) / self.scale).clamp(0.0, source_width);
        let bottom = ((bbox.y + bbox.h - self.pad_y) / self.scale).clamp(0.0, source_height);
        BBox::new(x, y, (right - x).max(0.0), (bottom - y).max(0.0))
    }
}

pub fn resize_letterbox(
    source: &[f32],
    channels: usize,
    source_width: usize,
    source_height: usize,
    target_width: usize,
    target_height: usize,
    padding: f32,
) -> Result<(Vec<f32>, Letterbox), String> {
    if channels == 0 {
        return Err("resize channels must be non-zero".to_string());
    }
    let expected = source_width
        .checked_mul(source_height)
        .and_then(|size| size.checked_mul(channels))
        .ok_or_else(|| "resize source size overflow".to_string())?;
    if source.len() != expected {
        return Err("resize source length does not match dimensions".to_string());
    }
    let letterbox = Letterbox::new(source_width, source_height, target_width, target_height)?;
    let resized_width = round_to_usize(
        usize_to_f32(source_width)? * letterbox.scale,
        "resized width",
    )?
    .max(1)
    .min(target_width);
    let resized_height = round_to_usize(
        usize_to_f32(source_height)? * letterbox.scale,
        "resized height",
    )?
    .max(1)
    .min(target_height);
    let mut output = vec![
        padding;
        target_width
            .checked_mul(target_height)
            .and_then(|size| size.checked_mul(channels))
            .ok_or_else(|| "resize target size overflow".to_string())?
    ];
    let pad_x = (target_width - resized_width) / 2;
    let pad_y = (target_height - resized_height) / 2;
    for y in 0..resized_height {
        let source_y = y
            .saturating_mul(source_height)
            .checked_div(resized_height)
            .ok_or_else(|| "resize source y overflow".to_string())?;
        for x in 0..resized_width {
            let source_x = x
                .saturating_mul(source_width)
                .checked_div(resized_width)
                .ok_or_else(|| "resize source x overflow".to_string())?;
            let source_index = (source_y * source_width + source_x) * channels;
            let target_index = ((y + pad_y) * target_width + x + pad_x) * channels;
            output[target_index..target_index + channels]
                .copy_from_slice(&source[source_index..source_index + channels]);
        }
    }
    Ok((output, letterbox))
}

pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents = values.iter().map(|value| (*value - max).exp());
    let sum: f32 = exponents.clone().sum();
    if sum == 0.0 || !sum.is_finite() {
        return vec![0.0; values.len()];
    }
    exponents.map(|value| value / sum).collect()
}

pub fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));
    indexed.truncate(k);
    indexed
}

pub fn iou(left: BBox, right: BBox) -> f32 {
    let left_right = left.x + left.w;
    let right_right = right.x + right.w;
    let left_bottom = left.y + left.h;
    let right_bottom = right.y + right.h;
    let intersection_width = (left_right.min(right_right) - left.x.max(right.x)).max(0.0);
    let intersection_height = (left_bottom.min(right_bottom) - left.y.max(right.y)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let union = left.area() + right.area() - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

pub fn nms(detections: &[Detection], threshold: f32) -> Vec<Detection> {
    let mut ordered = detections.to_vec();
    ordered.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut selected = Vec::new();
    for candidate in ordered {
        let suppressed = selected.iter().any(|existing: &Detection| {
            existing.class_id == candidate.class_id
                && iou(existing.bbox, candidate.bbox) > threshold
        });
        if !suppressed {
            selected.push(candidate);
        }
    }
    selected
}

fn usize_to_f32(value: usize) -> Result<f32, String> {
    if value > 16_777_216 {
        return Err("dimension cannot be represented exactly as f32".to_string());
    }
    let value = u32::try_from(value).map_err(|_| "dimension is out of range".to_string())?;
    value
        .to_string()
        .parse::<f32>()
        .map_err(|_| "dimension cannot be represented as f32".to_string())
}

fn round_to_usize(value: f32, field: &str) -> Result<usize, String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{field} must be finite and non-negative"));
    }
    let rounded = value.round();
    rounded
        .to_string()
        .parse::<usize>()
        .map_err(|_| format!("{field} is out of range"))
}

fn dimension_as_f32(value: usize) -> f32 {
    u32::try_from(value)
        .ok()
        .and_then(|value| value.to_string().parse::<f32>().ok())
        .unwrap_or(f32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn nms_suppresses_overlapping_same_class() {
        let detections = vec![
            Detection::new(BBox::new(0.0, 0.0, 10.0, 10.0), 0.9, 1),
            Detection::new(BBox::new(1.0, 1.0, 10.0, 10.0), 0.8, 1),
            Detection::new(BBox::new(1.0, 1.0, 10.0, 10.0), 0.7, 2),
        ];
        assert_eq!(nms(&detections, 0.5).len(), 2);
    }

    #[test]
    fn letterbox_maps_coordinates_back_to_source() {
        let letterbox = Letterbox::new(200, 100, 100, 100).expect("valid dimensions");
        let mapped = letterbox.map_to_source(BBox::new(25.0, 25.0, 50.0, 50.0));
        assert_eq!(mapped, BBox::new(50.0, 0.0, 100.0, 100.0));
    }

    proptest! {
        #[test]
        fn softmax_sums_to_one(values in proptest::collection::vec(-10.0_f32..10.0, 1..8)) {
            let output = softmax(&values);
            let sum: f32 = output.iter().sum();
            prop_assert!((sum - 1.0).abs() < 0.0001);
        }

        #[test]
        fn top_k_never_returns_more_than_k(
            values in proptest::collection::vec(-10.0_f32..10.0, 0..16),
            k in 0_usize..16,
        ) {
            prop_assert!(top_k(&values, k).len() <= k.min(values.len()));
        }
    }
}
