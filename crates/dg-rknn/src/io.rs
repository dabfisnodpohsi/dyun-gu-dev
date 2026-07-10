//! Hardware-independent RKNN I/O helpers.
//!
//! Pure logic shared by the feature-gated FFI backend: quantization metadata
//! mapping, `w_stride` layout handling, and the zero-copy vs host-staging path
//! decision. Kept free of `unsafe` and SDK types so it is unit-testable
//! without RKNN hardware.

use dg_core::{DataFormat, Quantization, QuantizationScheme, Shape, Strides};
use dg_runtime::{Error, Result};

/// Raw `rknn_tensor_qnt_type` values from `rknn_api.h`.
pub const RKNN_QNT_NONE: u32 = 0;
pub const RKNN_QNT_DFP: u32 = 1;
pub const RKNN_QNT_AFFINE_ASYMMETRIC: u32 = 2;

/// Data-movement path used between host tensors and the NPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoPath {
    /// NPU-allocated buffers bound once via `rknn_set_io_mem`.
    ZeroCopy,
    /// Per-run host copies via `rknn_inputs_set` / `rknn_outputs_get`.
    Staging,
}

impl IoPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ZeroCopy => "zero_copy",
            Self::Staging => "staging",
        }
    }
}

/// Selects the I/O path: zero-copy only when requested and the SDK memory API
/// is usable; otherwise the explicit host-staging fallback.
pub fn select_io_path(zero_copy_requested: bool, mem_api_available: bool) -> IoPath {
    if zero_copy_requested && mem_api_available {
        IoPath::ZeroCopy
    } else {
        IoPath::Staging
    }
}

/// Maps raw RKNN quantization attributes (`qnt_type` / `fl` / `zp` / `scale`)
/// onto the shared [`Quantization`] model. RKNN quantization is per-tensor,
/// so `axis` is always `None`.
pub fn quantization_from_rknn(qnt_type: u32, fl: i8, zp: i32, scale: f32) -> Result<Quantization> {
    match qnt_type {
        RKNN_QNT_NONE => Ok(Quantization::none()),
        RKNN_QNT_DFP => Ok(Quantization {
            scheme: QuantizationScheme::DynamicFixedPoint,
            scale: vec![(-f32::from(fl)).exp2()],
            zero_point: vec![i32::from(fl)],
            axis: None,
        }),
        RKNN_QNT_AFFINE_ASYMMETRIC => Ok(Quantization {
            scheme: QuantizationScheme::AffineAsymmetric,
            scale: vec![scale],
            zero_point: vec![zp],
            axis: None,
        }),
        other => Err(Error::Backend(format!(
            "unsupported RKNN quantization type: {other}"
        ))),
    }
}

/// Derives element strides for RKNN's padded-width layout. Returns `None`
/// when `w_stride` is zero (SDK convention for "equal to width") or matches
/// the logical width, i.e. the tensor is contiguous.
pub fn strides_from_w_stride(
    shape: &Shape,
    layout: DataFormat,
    w_stride: usize,
) -> Option<Strides> {
    if w_stride == 0 || shape.rank() != 4 {
        return None;
    }
    let dims = shape.dims();
    match layout {
        DataFormat::NCHW => {
            let (c, h, w) = (dims[1], dims[2], dims[3]);
            if w_stride == w {
                return None;
            }
            Some(Strides::new([c * h * w_stride, h * w_stride, w_stride, 1]))
        }
        DataFormat::NHWC => {
            let (h, w, c) = (dims[1], dims[2], dims[3]);
            if w_stride == w {
                return None;
            }
            Some(Strides::new([h * w_stride * c, w_stride * c, c, 1]))
        }
        _ => None,
    }
}

/// Byte size of the padded physical layout described by element `strides`.
pub fn padded_byte_len(shape: &Shape, strides: &Strides, elem_bytes: usize) -> Result<usize> {
    let (dims, stride_values) = validated_layout(shape, strides)?;
    let last_offset: usize = dims
        .iter()
        .zip(stride_values)
        .map(|(dim, stride)| (dim - 1) * stride)
        .sum();
    Ok((last_offset + 1) * elem_bytes)
}

/// Expands contiguous row-major bytes into the padded layout described by
/// `strides` (padding bytes are zeroed).
pub fn pad_bytes(
    contiguous: &[u8],
    shape: &Shape,
    strides: &Strides,
    elem_bytes: usize,
) -> Result<Vec<u8>> {
    let mut padded = vec![0u8; padded_byte_len(shape, strides, elem_bytes)?];
    for_each_run(
        shape,
        strides,
        elem_bytes,
        contiguous.len(),
        |pad, contig, run| {
            padded[pad..pad + run].copy_from_slice(&contiguous[contig..contig + run]);
        },
    )?;
    Ok(padded)
}

/// Collapses bytes in the padded layout described by `strides` into
/// contiguous row-major bytes.
pub fn depad_bytes(
    padded: &[u8],
    shape: &Shape,
    strides: &Strides,
    elem_bytes: usize,
) -> Result<Vec<u8>> {
    let needed = padded_byte_len(shape, strides, elem_bytes)?;
    if padded.len() < needed {
        return Err(Error::Backend(format!(
            "padded buffer too small: {} < {needed}",
            padded.len()
        )));
    }
    let logical = shape.element_count()? * elem_bytes;
    let mut contiguous = vec![0u8; logical];
    for_each_run(shape, strides, elem_bytes, logical, |pad, contig, run| {
        contiguous[contig..contig + run].copy_from_slice(&padded[pad..pad + run]);
    })?;
    Ok(contiguous)
}

fn validated_layout<'a>(
    shape: &'a Shape,
    strides: &'a Strides,
) -> Result<(&'a [usize], &'a [usize])> {
    let dims = shape.dims();
    let stride_values = strides.values();
    if dims.is_empty() || dims.len() != stride_values.len() {
        return Err(Error::Backend(format!(
            "stride rank {} does not match shape rank {}",
            stride_values.len(),
            dims.len()
        )));
    }
    if dims.contains(&0) {
        return Err(Error::Backend("padded layout with empty shape".to_string()));
    }
    if stride_values[dims.len() - 1] != 1 {
        return Err(Error::Backend(format!(
            "unsupported innermost stride: {}",
            stride_values[dims.len() - 1]
        )));
    }
    Ok((dims, stride_values))
}

/// Visits every contiguous innermost-dimension run, yielding byte offsets
/// into the padded and contiguous buffers plus the run length in bytes.
fn for_each_run(
    shape: &Shape,
    strides: &Strides,
    elem_bytes: usize,
    contiguous_len: usize,
    mut visit: impl FnMut(usize, usize, usize),
) -> Result<()> {
    let (dims, stride_values) = validated_layout(shape, strides)?;
    let logical = shape.element_count()? * elem_bytes;
    if contiguous_len != logical {
        return Err(Error::Backend(format!(
            "contiguous buffer size mismatch: {contiguous_len} != {logical}"
        )));
    }
    let rank = dims.len();
    let run = dims[rank - 1] * elem_bytes;
    let outer = &dims[..rank - 1];
    let mut index = vec![0usize; outer.len()];
    let mut contig = 0usize;
    loop {
        let pad: usize = index
            .iter()
            .zip(stride_values)
            .map(|(i, stride)| i * stride * elem_bytes)
            .sum();
        visit(pad, contig, run);
        contig += run;
        let mut axis = outer.len();
        loop {
            if axis == 0 {
                return Ok(());
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < outer[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_path_selection() {
        assert_eq!(select_io_path(true, true), IoPath::ZeroCopy);
        assert_eq!(select_io_path(true, false), IoPath::Staging);
        assert_eq!(select_io_path(false, true), IoPath::Staging);
        assert_eq!(select_io_path(false, false), IoPath::Staging);
    }

    #[test]
    fn quantization_none() {
        let quant = quantization_from_rknn(RKNN_QNT_NONE, 0, 0, 0.0).expect("none");
        assert_eq!(quant, Quantization::none());
    }

    #[test]
    fn quantization_affine_asymmetric_keeps_zp_and_scale() {
        let quant =
            quantization_from_rknn(RKNN_QNT_AFFINE_ASYMMETRIC, 0, -12, 0.0157).expect("affine");
        assert_eq!(quant.scheme, QuantizationScheme::AffineAsymmetric);
        assert_eq!(quant.scale, vec![0.0157]);
        assert_eq!(quant.zero_point, vec![-12]);
        assert_eq!(quant.axis, None);
    }

    #[test]
    fn quantization_dfp_derives_scale_from_fractional_length() {
        let quant = quantization_from_rknn(RKNN_QNT_DFP, 7, 0, 0.0).expect("dfp");
        assert_eq!(quant.scheme, QuantizationScheme::DynamicFixedPoint);
        assert_eq!(quant.scale, vec![2f32.powi(-7)]);
        assert_eq!(quant.zero_point, vec![7]);
    }

    #[test]
    fn quantization_unknown_type_is_rejected() {
        assert!(quantization_from_rknn(99, 0, 0, 0.0).is_err());
    }

    #[test]
    fn strides_contiguous_when_w_stride_matches_width() {
        let shape = Shape::new([1, 3, 224, 224]);
        assert_eq!(strides_from_w_stride(&shape, DataFormat::NCHW, 224), None);
        assert_eq!(strides_from_w_stride(&shape, DataFormat::NCHW, 0), None);
    }

    #[test]
    fn strides_nchw_padded_width() {
        let shape = Shape::new([1, 3, 224, 224]);
        let strides = strides_from_w_stride(&shape, DataFormat::NCHW, 256).expect("padded");
        assert_eq!(strides.values(), &[3 * 224 * 256, 224 * 256, 256, 1]);
    }

    #[test]
    fn strides_nhwc_padded_width() {
        let shape = Shape::new([1, 224, 224, 3]);
        let strides = strides_from_w_stride(&shape, DataFormat::NHWC, 256).expect("padded");
        assert_eq!(strides.values(), &[224 * 256 * 3, 256 * 3, 3, 1]);
    }

    #[test]
    fn strides_non_4d_shapes_are_ignored() {
        let shape = Shape::new([1, 128]);
        assert_eq!(strides_from_w_stride(&shape, DataFormat::NCHW, 256), None);
    }

    #[test]
    fn padded_byte_len_accounts_for_w_stride() {
        let shape = Shape::new([1, 2, 2, 3]);
        let strides = strides_from_w_stride(&shape, DataFormat::NCHW, 4).expect("padded");
        // Three padded rows of 4 plus the final row's 3 logical elements.
        assert_eq!(padded_byte_len(&shape, &strides, 1).expect("len"), 15);
    }

    #[test]
    fn pad_and_depad_round_trip_nchw() {
        let shape = Shape::new([1, 2, 2, 3]);
        let strides = strides_from_w_stride(&shape, DataFormat::NCHW, 4).expect("padded");
        let contiguous: Vec<u8> = (1..=12).collect();
        let padded = pad_bytes(&contiguous, &shape, &strides, 1).expect("pad");
        assert_eq!(padded, vec![1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12]);
        let restored = depad_bytes(&padded, &shape, &strides, 1).expect("depad");
        assert_eq!(restored, contiguous);
    }

    #[test]
    fn depad_reads_padded_nhwc_rows() {
        let shape = Shape::new([1, 2, 2, 2]);
        let strides = strides_from_w_stride(&shape, DataFormat::NHWC, 3).expect("padded");
        // Two rows of 2x2 elements padded to width 3 (x = padding).
        let padded = vec![1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0];
        let restored = depad_bytes(&padded, &shape, &strides, 1).expect("depad");
        assert_eq!(restored, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn pad_and_depad_respect_element_size() {
        let shape = Shape::new([1, 1, 2, 2]);
        let strides = strides_from_w_stride(&shape, DataFormat::NCHW, 3).expect("padded");
        let contiguous: Vec<u8> = (1..=8).collect();
        let padded = pad_bytes(&contiguous, &shape, &strides, 2).expect("pad");
        assert_eq!(padded, vec![1, 2, 3, 4, 0, 0, 5, 6, 7, 8]);
        let restored = depad_bytes(&padded, &shape, &strides, 2).expect("depad");
        assert_eq!(restored, contiguous);
    }

    #[test]
    fn pad_and_depad_reject_size_mismatch() {
        let shape = Shape::new([1, 1, 2, 2]);
        let strides = strides_from_w_stride(&shape, DataFormat::NCHW, 3).expect("padded");
        assert!(pad_bytes(&[0u8; 3], &shape, &strides, 1).is_err());
        assert!(depad_bytes(&[0u8; 3], &shape, &strides, 1).is_err());
    }
}
