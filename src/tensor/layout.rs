use std::iter::zip;

use crate::tensor::{
    errors::OpError,
    internals::{calculate_adjacent_dim_stride, calculate_dim_stride},
};

#[derive(Clone, Debug)]
pub struct Layout {
    pub(crate) shape: Box<[i32]>,
    pub(crate) stride: Box<[i32]>,
    pub(crate) adj_stride: Box<[i32]>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl Layout {
    pub fn new(
        shape: Box<[i32]>,
        stride: Box<[i32]>,
        adj_stride: Box<[i32]>,
        offset: usize,
        len: usize,
    ) -> Self {
        Self {
            shape,
            stride,
            adj_stride,
            offset,
            len,
        }
    }

    pub fn from_shape(shape: &[i32], offset: usize) -> Self {
        let len: i32 = shape.iter().product();

        Self {
            shape: shape.to_vec().into_boxed_slice(),
            stride: calculate_dim_stride(shape),
            adj_stride: vec![1; shape.len()].into_boxed_slice(),
            offset,
            len: len as usize,
        }
    }

    pub fn from_slice(shape: &[i32], stride: &[i32], offset: usize) -> Self {
        let len: i32 = shape.iter().product();

        Self {
            shape: shape.to_vec().into_boxed_slice(),
            stride: stride.to_vec().into_boxed_slice(),
            adj_stride: calculate_adjacent_dim_stride(stride, shape),
            offset,
            len: len as usize,
        }
    }

    pub fn view(&self, shape: &[i32]) -> Result<Self, OpError<'_>> {
        #[cfg(feature = "debug_only_check")]
        {
            debug_only!({
                let size: i32 = shape.iter().product();
                if size.max(0) as usize != self.len() {
                    return Err(OpError::InvalidViewShape);
                }
            })
        }
        #[cfg(not(feature = "debug_only_check"))]
        {
            let size: i32 = shape.iter().product();
            if size.max(0) as usize != self.len() {
                return Err(OpError::InvalidViewShape);
            }
        }
        Ok(Layout::from_shape(shape, self.offset))
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() != self.stride.len() {
            return false;
        }

        let mut acc = 1;

        for (&dim_size, &stride) in zip(self.shape.iter(), self.stride.iter()).rev() {
            if stride != acc {
                return false;
            }

            acc *= dim_size;
        }

        true
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

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layout {{ shape: {:?}, stride: {:?}, offset: {} }}",
            &self.shape, &self.stride, self.offset
        )
    }
}
