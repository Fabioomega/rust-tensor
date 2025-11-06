#[macro_use]
mod convenience;

mod device;
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

pub type CPU = device::CPU;
pub type Mat<T> = mat::RawTensor<T, device::CPU>;
