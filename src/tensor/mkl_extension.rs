#![allow(
    improper_ctypes,
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case
)]

use std::ffi::{c_double, c_float, c_int};

unsafe extern "C" {
    pub fn cblas_dscal(N: c_int, alpha: f64, X: *mut f64, incX: c_int);
}
