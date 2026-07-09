use dg_core::{
    pack_float4, pack_int4, unpack_float4, unpack_int4, BufferDesc, CpuDevice, DataFormat,
    DataType, Device, DeviceKind, Shape, Tensor, TensorDesc,
};
use proptest::prelude::*;

#[test]
fn cpu_tensor_allocate_copy_and_reshape() {
    let device = CpuDevice::new();
    let shape = Shape::new([2, 2]);
    let desc = TensorDesc::new(
        shape.clone(),
        DataType::U8,
        DataFormat::NCHW,
        DeviceKind::Cpu,
    );
    let mut tensor = Tensor::allocate(&device, desc).expect("allocate tensor");
    tensor
        .buffer()
        .write_from_slice(&[1, 2, 3, 4])
        .expect("write tensor");

    let other_desc = TensorDesc::new(
        shape.clone(),
        DataType::U8,
        DataFormat::NCHW,
        DeviceKind::Cpu,
    );
    let mut other = Tensor::allocate(&device, other_desc).expect("allocate other tensor");
    tensor.copy_to(&mut other).expect("copy tensor");
    assert_eq!(other.buffer().read_bytes(), vec![1, 2, 3, 4]);

    tensor.reshape(Shape::new([4])).expect("reshape");
    assert_eq!(tensor.desc().shape().dims(), &[4]);
}

#[test]
fn tensor_reshape_rejects_non_contiguous_strides() {
    let device = CpuDevice::new();
    let desc = TensorDesc::new(
        Shape::new([2, 2]),
        DataType::U8,
        DataFormat::NCHW,
        DeviceKind::Cpu,
    )
    .with_strides(dg_core::Strides::new([3, 1]));
    let mut tensor = Tensor::allocate(&device, desc).expect("allocate tensor");

    let err = tensor
        .reshape(Shape::new([4]))
        .expect_err("reshape should fail");
    assert!(matches!(
        err,
        dg_core::Error::Shape(message) if message.contains("non-contiguous strides")
    ));
}

#[test]
fn buffer_refcount_and_copy_semantics() {
    let device = CpuDevice::new();
    let buffer = device
        .alloc(BufferDesc::new(4, 1))
        .expect("allocate buffer");
    assert_eq!(buffer.ref_count(), 1);

    let cloned = buffer.clone();
    assert_eq!(buffer.ref_count(), 2);

    cloned.write_from_slice(&[9, 8, 7, 6]).expect("write");
    let mut dst = [0; 4];
    buffer.copy_into(&mut dst).expect("copy into");
    assert_eq!(dst, [9, 8, 7, 6]);
}

#[test]
fn datatype_round_trip_and_sizes() {
    assert_eq!(DataType::of::<f32>(), DataType::F32);
    assert_eq!(DataType::of::<half::f16>(), DataType::F16);
    assert_eq!(DataType::of::<half::bf16>(), DataType::BF16);
    assert_eq!(
        DataType::I4.storage_bytes_for_elements(2).expect("bytes"),
        1
    );
    assert_eq!(
        DataType::F4.storage_bytes_for_elements(3).expect("bytes"),
        2
    );
}

#[test]
fn packed_int4_and_float4_round_trip() {
    let int4 = [-8, -1, 0, 3, 7];
    let packed_int4 = pack_int4(&int4).expect("pack int4");
    assert_eq!(
        unpack_int4(&packed_int4, int4.len()).expect("unpack int4"),
        int4
    );

    let float4 = [0, 1, 2, 15, 7];
    let packed_float4 = pack_float4(&float4).expect("pack float4");
    assert_eq!(
        unpack_float4(&packed_float4, float4.len()).expect("unpack float4"),
        float4
    );
}

proptest! {
    #[test]
    fn datatype_storage_bytes_match_packed_length(count in 0usize..256, bits in prop_oneof![Just(4u8), Just(8u8), Just(16u8), Just(32u8)]) {
        let dtype = DataType::new(dg_core::TypeCode::Uint, bits, 1);
        let bytes = dtype.storage_bytes_for_elements(count).expect("size");
        let expected = (count * usize::from(bits)).div_ceil(8);
        prop_assert_eq!(bytes, expected);
    }

    #[test]
    fn int4_pack_unpack_round_trip(values in prop::collection::vec(-8i8..=7, 0..128)) {
        let packed = pack_int4(&values).expect("pack");
        let unpacked = unpack_int4(&packed, values.len()).expect("unpack");
        prop_assert_eq!(unpacked, values);
    }

    #[test]
    fn float4_pack_unpack_round_trip(values in prop::collection::vec(0u8..=15, 0..128)) {
        let packed = pack_float4(&values).expect("pack");
        let unpacked = unpack_float4(&packed, values.len()).expect("unpack");
        prop_assert_eq!(unpacked, values);
    }

    #[test]
    fn contiguous_shape_stride_round_trip(dims in prop::collection::vec(1usize..8, 0..6)) {
        let shape = Shape::new(dims);
        let strides = shape.contiguous_strides();
        prop_assert!(strides.is_contiguous_for(&shape));
        prop_assert_eq!(strides.values().len(), shape.rank());
    }
}
