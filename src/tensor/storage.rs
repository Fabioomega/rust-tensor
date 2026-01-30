use parking_lot::RwLock;
use std::boxed::Box;
use std::sync::Arc;

use crate::tensor::iter::{ChunkedSliceIter, CopiedSliceIter, InformedSliceIter, SliceIter};
use crate::tensor::layout::Layout;
use crate::tensor::traits::Dimension;
use crate::{debug_assert_positive, impl_display};

#[derive(Debug)]
pub struct Storage<T: Copy> {
    buffer: Arc<RwLock<Box<[T]>>>,
}

impl<T: Copy> Storage<T> {
    #[inline]
    pub fn from_scalar(scalar: T, len: usize) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(vec![scalar; len].into_boxed_slice())),
        }
    }

    #[inline]
    pub fn from_arc(buffer: Arc<RwLock<Box<[T]>>>) -> Self {
        Self { buffer }
    }

    #[inline]
    pub fn from_vec(vector: Box<[T]>) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(vector)),
        }
    }

    #[inline]
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vector = std::vec::Vec::from_iter(iter).into_boxed_slice();
        Self::from_vec(vector)
    }

    #[inline]
    pub fn clone_reference(&self) -> Self {
        Storage::from_arc(self.buffer.clone())
    }
}

impl<T: Copy> Clone for Storage<T> {
    fn clone(&self) -> Self {
        let buffer = self.buffer.read().clone();
        Storage::from_vec(buffer)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct TensorData<T: Copy> {
    storage: Storage<T>,
    layout: Layout,
}

impl<T: Copy> TensorData<T> {
    #[inline]
    pub fn new(storage: Storage<T>, layout: Layout) -> Self {
        Self { storage, layout }
    }

    #[inline]
    pub fn from_scalar(scalar: T, shape: &[i32]) -> Self {
        let len: i32 = shape.iter().product();

        debug_assert_positive!(len);

        Self {
            storage: Storage::from_scalar(scalar, len as usize),
            layout: Layout::from_shape(shape, 0),
        }
    }

    #[inline]
    pub fn from_arc(buffer: Arc<RwLock<Box<[T]>>>, shape: &[i32]) -> Self {
        Self {
            storage: Storage::from_arc(buffer),
            layout: Layout::from_shape(shape, 0),
        }
    }

    #[inline]
    pub fn from_vec(vector: Box<[T]>, shape: &[i32]) -> Self {
        debug_assert!(vector.len() == (shape.iter().product::<i32>() as usize));

        Self {
            storage: Storage::from_vec(vector),
            layout: Layout::from_shape(shape, 0),
        }
    }

    #[inline]
    pub fn from_iter<I>(iter: I, shape: &[i32]) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vector = std::vec::Vec::from_iter(iter).into_boxed_slice();
        Self::from_vec(vector, shape)
    }

    #[inline]
    pub fn as_layout(&self, layout: Layout) -> Self {
        Self {
            storage: self.storage.clone_reference(),
            layout,
        }
    }

    #[inline]
    pub fn iter(&self) -> SliceIter<'_, T> {
        SliceIter::new(
            &self.storage.buffer,
            self.len(),
            self.shape(),
            self.adj_stride(),
        )
    }

    #[inline]
    pub fn copied_iter(&self) -> CopiedSliceIter<'_, T> {
        CopiedSliceIter::new(
            &self.storage.buffer,
            self.len(),
            self.shape(),
            self.adj_stride(),
        )
    }

    #[inline]
    pub fn informed_iter(&self) -> InformedSliceIter<'_, T> {
        InformedSliceIter::new(&self.storage.buffer, &self.layout)
    }

    #[inline]
    pub fn clone_reference(&self) -> Self {
        Self {
            storage: self.storage.clone_reference(),
            layout: self.layout.clone(),
        }
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Copy + Default> TensorData<T> {
    #[inline]
    pub fn packed_iter(&self) -> crate::tensor::definitions::ChunkedIter<'_, T> {
        ChunkedSliceIter::new(self.copied_iter())
    }
}

impl<T: Copy> Clone for TensorData<T> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<T: Copy> Dimension for TensorData<T> {
    #[inline]
    fn len(&self) -> usize {
        self.layout.len()
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
        self.layout.offset()
    }
}

impl_display!(TensorData<T>);
