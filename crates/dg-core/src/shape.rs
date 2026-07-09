use crate::{Error, Result};

/// Logical tensor extents.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn element_count(&self) -> Result<usize> {
        self.dims.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::Shape("shape element count overflow".to_string()))
        })
    }

    pub fn contiguous_strides(&self) -> Strides {
        Strides::contiguous_for(self)
    }
}

/// Stride vector expressed in logical elements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Strides {
    values: Vec<usize>,
}

impl Strides {
    pub fn new(values: impl Into<Vec<usize>>) -> Self {
        Self {
            values: values.into(),
        }
    }

    pub fn values(&self) -> &[usize] {
        &self.values
    }

    pub fn rank(&self) -> usize {
        self.values.len()
    }

    pub fn contiguous_for(shape: &Shape) -> Self {
        let mut values = vec![0; shape.rank()];
        let mut stride = 1usize;
        for (index, dim) in shape.dims().iter().enumerate().rev() {
            values[index] = stride;
            stride = stride.saturating_mul(*dim);
        }
        Self { values }
    }

    pub fn is_contiguous_for(&self, shape: &Shape) -> bool {
        self.values == Self::contiguous_for(shape).values
    }
}
