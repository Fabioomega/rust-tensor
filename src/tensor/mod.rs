#[macro_use]
mod convenience;

mod impl_generics;
mod internals;
mod iter;
mod layout;
mod macros;
mod traits;

pub mod mat;
pub mod slice;
pub use convenience::*;
pub use iter::StepInfo;
pub use traits::Dimension;

pub type Mat<T> = mat::RawTensor<T>;
