extern crate cblas;
extern crate intel_mkl_src;
extern crate intel_mkl_sys;
extern crate lapacke;

pub const PACKING_BUFFER_SIZE: usize = 128;

#[macro_use]
mod convenience;

mod definitions;
pub mod errors;
mod impl_generics;
mod internals;
mod iter;
mod layout;
mod macros;
mod mkl_extension;
mod storage;
mod traits;

pub mod graph;
pub mod ops;
pub mod promise;
// pub mod slice;
pub mod tensor;
pub use convenience::*;
// pub use iter::StepInfo;
// pub use traits::Dimension;

pub use promise::{CachedTensorPromise, TensorPromise};
pub use tensor::Tensor;
pub use traits::Dimension;
