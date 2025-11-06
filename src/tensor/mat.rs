use std::iter::zip;
use std::marker::PhantomData;
use std::ops::Index;
use std::ptr::NonNull;

use crate::tensor::device::{CPU, DeviceInfo};
use crate::tensor::internals::calculate_adjacent_dim_stride;
use crate::tensor::iter::{ContiguousIter, MutContiguousIter};
use crate::tensor::layout::Layout;
use crate::tensor::slice::{RawMutTensorSlice, RawTensorSlice, SliceInfo, SliceRange};
use crate::tensor::traits::{Dimension, LinearIndexMemory, MutLinearIndexMemory};
use crate::{debug_assert_positive, impl_display, impl_index};
pub struct RawTensor<T, D>
where
    D: DeviceInfo,
{
    buffer: Vec<T>,
    layout: Layout,
    device: PhantomData<D>,
}

impl<T: Copy, D: DeviceInfo> RawTensor<T, D> {
    pub fn assign_scalar(&mut self, range: &[SliceRange], value: T) {
        let mut slice = self.mut_slice(range);
    }

    #[inline]
    pub fn iter(&self) -> ContiguousIter<'_, T, D> {
        ContiguousIter::new(&self)
    }

    #[inline]
    pub fn mut_iter(&mut self) -> MutContiguousIter<'_, T> {
        MutContiguousIter::new(self.as_mut_ptr(), self.len())
    }

    #[inline]
    pub fn slice(&self, range: &[SliceRange]) -> RawTensorSlice<'_, T, D> {
        RawTensorSlice::from_info(&self, SliceInfo::from_range(self, range))
    }

    #[inline]
    pub fn as_slice(&self) -> RawTensorSlice<'_, T, D> {
        RawTensorSlice::new(&self, self.layout.clone(), 0)
    }

    #[inline]
    pub fn mut_slice(&mut self, range: &[SliceRange]) -> RawMutTensorSlice<'_, T, D> {
        RawMutTensorSlice::from_info(self, SliceInfo::from_range(self, range))
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> RawMutTensorSlice<'_, T, D> {
        RawMutTensorSlice::new(self, self.layout.clone(), 0)
    }

    #[inline]
    pub fn transposed(&self) -> RawTensorSlice<'_, T, D> {
        let new_shape = self.shape().to_vec();

        let mut new_stride: Vec<i32> = self.stride().iter().map(|x| *x).collect();

        let last = new_stride.len() - 1;
        let temp = new_stride[0];
        new_stride[0] = new_stride[last];
        new_stride[last] = temp;

        let new_adj = calculate_adjacent_dim_stride(&new_stride, &new_shape);

        RawTensorSlice::new(&self, Layout::new(new_shape, new_stride, new_adj), 0)
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buffer.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> NonNull<T> {
        NonNull::new(self.buffer.as_mut_ptr()).unwrap()
    }
}

impl<T> RawTensor<T, CPU>
where
    T: Copy,
{
    pub fn from_scalar(scalar: T, shape: &[i32]) -> Self {
        let size: i32 = shape.iter().product();

        debug_assert_positive!(size);

        Self {
            buffer: vec![scalar; size as usize],
            layout: Layout::from_shape(shape),
            device: PhantomData,
        }
    }

    pub fn from_vec(vector: Vec<T>, shape: &[i32]) -> Self {
        Self {
            buffer: vector,
            layout: Layout::from_shape(shape),
            device: PhantomData,
        }
    }
}

impl<'a, T, D: DeviceInfo> LinearIndexMemory for RawTensor<T, D> {
    type Output = T;

    fn index_memory(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}

impl<'a, T, D: DeviceInfo> MutLinearIndexMemory for RawTensor<T, D> {
    type Output = T;

    fn mut_index_memory(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buffer[index]
    }
}

impl<T, D: DeviceInfo> Dimension for RawTensor<T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    fn shape(&self) -> &[i32] {
        self.layout.shape()
    }

    #[inline]
    fn stride(&self) -> &'_ [i32] {
        self.layout.stride()
    }

    #[inline]
    fn adj_stride(&self) -> &'_ [i32] {
        self.layout.adj_stride()
    }

    #[inline]
    fn offset(&self) -> usize {
        0
    }
}

impl_display!(RawTensor<T, D>);
impl_index!(RawTensor<T, CPU>);
