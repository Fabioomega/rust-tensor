use crate::tensor::internals::{calculate_adjacent_dim_stride, calculate_dim_stride};

#[derive(Clone)]
pub struct Layout {
    pub(crate) shape: Box<[i32]>,
    pub(crate) stride: Box<[i32]>,
    pub(crate) adj_stride: Box<[i32]>,
}

impl Layout {
    pub fn new(shape: Box<[i32]>, stride: Box<[i32]>, adj_stride: Box<[i32]>) -> Self {
        Self {
            shape: shape,
            stride: stride,
            adj_stride: adj_stride,
        }
    }

    pub fn from_shape(shape: &[i32]) -> Self {
        Self {
            shape: shape.to_vec().into_boxed_slice(),
            stride: calculate_dim_stride(shape),
            adj_stride: vec![1; shape.len()].into_boxed_slice(),
        }
    }

    pub fn from_slice(shape: &[i32], stride: &[i32]) -> Self {
        Self {
            shape: shape.to_vec().into_boxed_slice(),
            stride: stride.to_vec().into_boxed_slice(),
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
