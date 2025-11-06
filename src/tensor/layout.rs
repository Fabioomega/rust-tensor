use crate::tensor::internals::{calculate_adjacent_dim_stride, calculate_dim_stride};

#[derive(Clone)]
pub struct Layout {
    pub(crate) shape: Vec<i32>,
    pub(crate) stride: Vec<i32>,
    pub(crate) adj_stride: Vec<i32>,
}

impl Layout {
    pub fn new(shape: Vec<i32>, stride: Vec<i32>, adj_stride: Vec<i32>) -> Self {
        Self {
            shape: shape,
            stride: stride,
            adj_stride: adj_stride,
        }
    }

    pub fn from_shape(shape: &[i32]) -> Self {
        Self {
            shape: shape.to_vec(),
            stride: calculate_dim_stride(shape),
            adj_stride: vec![1; shape.len()],
        }
    }

    pub fn from_slice(shape: &[i32], stride: &[i32]) -> Self {
        Self {
            shape: shape.to_vec(),
            stride: stride.to_vec(),
            adj_stride: calculate_adjacent_dim_stride(stride, shape),
        }
    }

    #[inline]
    pub fn shape(&self) -> &'_ [i32] {
        &self.shape
    }

    #[inline]
    pub fn stride(&self) -> &'_ [i32] {
        &self.stride
    }

    #[inline]
    pub fn adj_stride(&self) -> &'_ [i32] {
        &self.adj_stride
    }
}
