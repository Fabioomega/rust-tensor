use parking_lot::RwLock;
use std::sync::Arc;

use crate::tensor::internals::calculate_adjacent_dim_stride;
use crate::tensor::iter::{ContiguousIter, InformedSliceIter, MutContiguousIter};
use crate::tensor::layout::Layout;
use crate::tensor::slice::{RawTensorSlice, SliceInfo, SliceRange};
use crate::tensor::traits::Dimension;
use crate::{debug_assert_positive, debug_only, impl_display};

#[derive(Clone)]
pub struct RawTensor<T> {
    buffer: Arc<RwLock<Box<[T]>>>,
    layout: Layout,
}

impl<T: Copy> RawTensor<T> {
    pub fn from_scalar(scalar: T, shape: &[i32]) -> Self {
        let size: i32 = shape.iter().product();

        debug_assert_positive!(size);

        Self {
            buffer: Arc::new(RwLock::new(vec![scalar; size as usize].into_boxed_slice())),
            layout: Layout::from_shape(shape),
        }
    }

    pub fn from_arc(buffer: Arc<RwLock<Box<[T]>>>, shape: &[i32]) -> Self {
        Self {
            buffer: buffer,
            layout: Layout::from_shape(shape),
        }
    }

    pub fn from_vec(vector: Box<[T]>, shape: &[i32]) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(vector)),
            layout: Layout::from_shape(shape),
        }
    }
}

impl<T: Copy> RawTensor<T> {
    #[inline]
    pub fn iter(&self) -> ContiguousIter<'_, T> {
        ContiguousIter::new(&self.buffer)
    }

    #[inline]
    pub fn mut_iter(&mut self) -> MutContiguousIter<'_, T> {
        MutContiguousIter::new(&self.buffer)
    }

    #[inline]
    pub fn informed_iter(&self) -> InformedSliceIter<'_, T> {
        InformedSliceIter::new(self.buffer.read(), self.len(), &self.layout)
    }

    #[inline]
    pub fn slice(&self, range: &[SliceRange]) -> RawTensorSlice<T> {
        RawTensorSlice::from_info(
            &self.buffer,
            self.stride().into(),
            SliceInfo::from_range(self, range),
        )
    }

    #[inline]
    pub fn mut_slice(&mut self, range: &[SliceRange]) -> RawTensorSlice<T> {
        RawTensorSlice::from_info(
            &self.buffer,
            self.stride().into(),
            SliceInfo::from_range(self, range),
        )
    }

    #[inline]
    pub fn as_slice(&self) -> RawTensorSlice<T> {
        RawTensorSlice::new(&self.buffer, self.layout.clone(), 0)
    }

    #[inline]
    pub fn transposed(&self) -> RawTensorSlice<T> {
        let new_shape = self.shape().to_vec().into_boxed_slice();

        let mut new_stride: Box<[i32]> = self.layout.stride.clone();

        let last = new_stride.len() - 1;
        let temp = new_stride[0];
        new_stride[0] = new_stride[last];
        new_stride[last] = temp;

        let new_adj = calculate_adjacent_dim_stride(&new_stride, &new_shape);

        RawTensorSlice::new(&self.buffer, Layout::new(new_shape, new_stride, new_adj), 0)
    }

    #[inline]
    pub fn transposed_n(&self, axis: usize) -> RawTensorSlice<T> {
        debug_assert!(axis < self.shape().len());

        let new_shape = self.shape().to_vec().into_boxed_slice();

        let mut new_stride: Box<[i32]> = self.layout.stride.clone();

        let last = new_stride.len() - axis - 1;
        let temp = new_stride[axis];
        new_stride[axis] = new_stride[last];
        new_stride[last] = temp;

        let new_adj = calculate_adjacent_dim_stride(&new_stride, &new_shape);

        RawTensorSlice::new(&self.buffer, Layout::new(new_shape, new_stride, new_adj), 0)
    }

    #[inline]
    pub fn view(&self, new_shape: &[i32]) -> RawTensorSlice<T> {
        debug_only!({
            let size: i32 = new_shape.iter().product();
            if size.max(0) as usize != self.len() {
                panic!(
                    "view can only be used to change the shape of a tensor not its element count. Maybe what you want is .slice?"
                );
            }
        });

        let layout = Layout::from_shape(new_shape);
        RawTensorSlice::new(&self.buffer, layout, 0)
    }

    #[inline]
    pub fn reshape(&self, new_shape: &[i32]) -> RawTensor<T> {
        debug_only!({
            let size: i32 = new_shape.iter().product();
            if size.max(0) as usize != self.len() {
                panic!(
                    "reshape can only be used to change the shape of a tensor not its element count. Maybe what you want is .slice?"
                );
            }
        });

        RawTensor::from_arc(Arc::clone(&self.buffer), new_shape)
    }

    #[inline]
    pub fn reshape_inplace(&mut self, new_shape: &[i32]) -> &Self {
        debug_only!({
            let size: i32 = new_shape.iter().product();
            if size.max(0) as usize != self.len() {
                panic!(
                    "reshape_inplace can only be used to change the shape of a tensor not its element count. Maybe what you want is .slice?"
                );
            }
        });

        self.layout = Layout::from_shape(new_shape);

        self
    }

    #[inline]
    pub fn assign_scalar(&mut self, range: &[SliceRange], value: T) {
        for el in self.mut_slice(range).mut_iter() {
            *el = value;
        }
    }
}

impl<T> Dimension for RawTensor<T> {
    #[inline]
    fn len(&self) -> usize {
        self.buffer.read().len()
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

impl_display!(RawTensor<T>);
// impl_index!(RawTensor<T, CPU>);
