use parking_lot::RwLockReadGuard;

pub trait Dimension {
    fn shape(&self) -> &'_ [i32];
    fn stride(&self) -> &'_ [i32];
    fn adj_stride(&self) -> &'_ [i32];
    fn len(&self) -> usize;
    fn offset(&self) -> usize;
}
