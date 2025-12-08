use std::iter::{Zip, zip};
use std::ops::Index;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::tensor::iter::MutSliceIter;
use crate::{
    debug_only, impl_display,
    tensor::{
        internals::calculate_adjacent_dim_stride,
        iter::{InformedSliceIter, SliceIter},
        layout::Layout,
        mat::RawTensor,
        traits::Dimension,
    },
};
use parking_lot::RwLock;

enum SliceBounds {
    Beginning,
    Index(usize),
    ReverseIndex(usize),
    End,
}

pub struct SliceRange {
    start: SliceBounds,
    end: SliceBounds,
}

impl From<RangeFrom<i32>> for SliceRange {
    #[inline]
    fn from(value: RangeFrom<i32>) -> Self {
        if value.start >= 0 {
            Self {
                start: SliceBounds::Index(value.start as usize),
                end: SliceBounds::End,
            }
        } else {
            Self {
                start: SliceBounds::ReverseIndex((-value.start) as usize),
                end: SliceBounds::End,
            }
        }
    }
}

impl From<RangeTo<i32>> for SliceRange {
    #[inline]
    fn from(value: RangeTo<i32>) -> Self {
        if value.end >= 0 {
            Self {
                start: SliceBounds::Beginning,
                end: SliceBounds::Index(value.end as usize),
            }
        } else {
            Self {
                start: SliceBounds::Beginning,
                end: SliceBounds::ReverseIndex((-value.end) as usize),
            }
        }
    }
}

impl From<RangeFull> for SliceRange {
    #[inline]
    fn from(_: RangeFull) -> Self {
        Self {
            start: SliceBounds::Beginning,
            end: SliceBounds::End,
        }
    }
}

impl From<Range<i32>> for SliceRange {
    #[inline]
    fn from(value: Range<i32>) -> Self {
        let start = if value.start >= 0 {
            SliceBounds::Index(value.start as usize)
        } else {
            SliceBounds::ReverseIndex((-value.start) as usize)
        };

        let end = if value.end >= 0 {
            SliceBounds::Index(value.end as usize)
        } else {
            SliceBounds::ReverseIndex((-value.end) as usize)
        };

        Self { start, end }
    }
}

/////////////////////////////////////////////////////

#[derive(Debug)]
pub struct SliceInfo {
    offset: usize,
    shape: Box<[i32]>,
    adj_stride: Box<[i32]>,
}

impl SliceInfo {
    pub(crate) fn from_range<T: Dimension>(container: &T, range: &[SliceRange]) -> Self {
        debug_assert!(container.shape().len() >= range.len());

        let mut offset: i64 = container.offset() as i64;
        let mut new_shape: Box<[i32]> = container.shape().to_vec().into_boxed_slice();

        for (dim, r) in range.iter().enumerate() {
            let start = match r.start {
                SliceBounds::Beginning => 0,
                SliceBounds::Index(i) => {
                    offset += (i as i64) * container.stride()[dim] as i64;

                    i as i32
                }
                SliceBounds::ReverseIndex(i) => {
                    let true_index = container.shape()[dim] as i64 - i as i64;
                    offset += true_index * container.stride()[dim] as i64;

                    true_index as i32
                }
                _ => unreachable!(),
            };

            let end = match r.end {
                SliceBounds::End => container.shape()[dim],
                SliceBounds::Index(i) => i as i32,
                SliceBounds::ReverseIndex(i) => {
                    let true_index = container.shape()[dim] as i64 - i as i64;
                    true_index as i32
                }
                _ => unreachable!(),
            };

            debug_only!({
                if end <= start {
                    panic!("You cannot create a slice that references out of bounds memory!")
                }
            });

            new_shape[dim] = end - start;
        }

        debug_only!({
            let len: i32 = new_shape.iter().product();
            let len = len as usize;

            if len + (offset as usize) > container.len() {
                panic!(
                    "You cannot create a slice bigger than the matrix it originates from! Expected a maximum length of {} elements but found {}!",
                    container.len(),
                    len
                );
            }
        });

        let adj_stride = calculate_adjacent_dim_stride(container.stride(), &new_shape);

        Self {
            offset: offset as usize,
            shape: new_shape,
            adj_stride,
        }
    }
}

/////////////////////////////////////////////////////

pub struct RawTensorSlice<T: Copy> {
    base: Arc<RwLock<Box<[T]>>>,
    offset: usize,
    layout: Layout,
    len: usize,
}

impl<T: Copy> RawTensorSlice<T> {
    #[inline]
    pub(crate) fn new(base: &Arc<RwLock<Box<[T]>>>, layout: Layout, offset: usize) -> Self {
        let len: i32 = layout.shape.iter().product();

        Self {
            base: base.clone(),
            layout,
            len: len as usize,
            offset,
        }
    }

    pub(crate) fn from_info(
        base: &Arc<RwLock<Box<[T]>>>,
        stride: Box<[i32]>,
        info: SliceInfo,
    ) -> Self {
        let len: i32 = info.shape.iter().product();

        Self {
            base: base.clone(),
            offset: info.offset,
            len: len as usize,
            layout: Layout::new(info.shape, stride.to_vec().into(), info.adj_stride),
        }
    }

    #[inline]
    pub fn iter(&self) -> SliceIter<'_, T> {
        SliceIter::new(&self.base, self.len(), self.shape(), self.adj_stride())
    }

    #[inline]
    pub fn mut_iter(&self) -> MutSliceIter<'_, T> {
        MutSliceIter::new(&self.base, self.len(), self.shape(), self.adj_stride())
    }

    #[inline]
    pub fn informed_iter(&self) -> InformedSliceIter<'_, T> {
        InformedSliceIter::new(self.base.read(), self.len(), &self.layout)
    }

    #[inline]
    pub fn slice(&self, range: &[SliceRange]) -> RawTensorSlice<T> {
        let info = SliceInfo::from_range(self, range);

        RawTensorSlice::from_info(&self.base, self.stride().into(), info)
    }

    #[inline]
    pub fn as_slice(&self) -> &Self {
        self
    }
}

impl<T: Copy> Dimension for RawTensorSlice<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn shape(&self) -> &[i32] {
        self.layout.shape()
    }

    #[inline]
    fn stride(&self) -> &[i32] {
        self.layout.stride()
    }

    #[inline]
    fn adj_stride(&self) -> &[i32] {
        self.layout.adj_stride()
    }

    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }
}

impl_display!(RawTensorSlice<T>);
// impl_index!(RawTensorSlice<'_, T, CPU>);
