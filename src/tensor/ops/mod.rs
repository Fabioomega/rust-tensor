pub mod fusion;
pub mod impl_compute_op;
pub mod impl_op;

pub use impl_compute_op::ComputeWrapperSpec;
pub use impl_compute_op::{compute_layout, cpu_compute};
