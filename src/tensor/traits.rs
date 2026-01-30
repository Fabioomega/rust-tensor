use crate::tensor::layout::Layout;
use crate::tensor::storage::TensorData;

pub trait Dimension {
    fn shape(&self) -> &'_ [i32];
    fn stride(&self) -> &'_ [i32];
    fn adj_stride(&self) -> &'_ [i32];
    fn len(&self) -> usize;
    fn offset(&self) -> usize;
}

pub trait Promising {
    type Output: Copy;

    fn compute(&self) -> TensorData<Self::Output>;

    fn layout(&self) -> &Layout;
}

pub trait StreamingIterator {
    type Item<'a>
    where
        Self: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}
