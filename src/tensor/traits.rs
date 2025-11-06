pub trait LinearIndexMemory {
    type Output;

    fn index_memory(&self, index: usize) -> &Self::Output;
}

pub trait MutLinearIndexMemory {
    type Output;

    fn mut_index_memory(&mut self, index: usize) -> &mut Self::Output;
}

pub trait Dimension {
    fn shape(&self) -> &'_ [i32];
    fn stride(&self) -> &'_ [i32];
    fn adj_stride(&self) -> &'_ [i32];
    fn len(&self) -> usize;
    fn offset(&self) -> usize;
}

pub trait Sliceable: Dimension + LinearIndexMemory {}

pub trait MutSliceable: Dimension + MutLinearIndexMemory {}

// pub trait MutSliceable: Dimension + MutLinearIndexMemory {}
