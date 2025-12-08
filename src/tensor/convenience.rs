#[macro_export]
macro_rules! s {
    ($($range: expr),*) => {
        &[$(crate::tensor::slice::SliceRange::from($range)),*]
    };
}

#[macro_export]
macro_rules! zeros {
    ($shape:expr) => {
        crate::tensor::Mat::from_scalar(0.0, $shape)
    };
}

#[macro_export]
macro_rules! ones {
    ($shape:expr) => {
        crate::tensor::Mat::from_scalar(1.0, $shape)
    };
}

pub mod arange {
    use crate::Mat;

    #[macro_export]
    macro_rules! arange {
        ($size: expr) => {
            crate::arange::_arange_default($size)
        };

        ($start: expr, $end: expr) => {
            crate::arange::_arange_start($start, $end)
        };

        ($start: expr, $end: expr, $step: expr) => {
            crate::arange::_arange_step($start, $end, $step)
        };
    }

    pub fn _arange_default(size: usize) -> Mat<f64> {
        let mut v: Vec<f64> = Vec::with_capacity(size);

        for i in 0..size {
            v.push(i as f64);
        }

        Mat::from_vec(v.into_boxed_slice(), &[size as i32])
    }

    pub fn _arange_start(start: usize, end: usize) -> Mat<f64> {
        let mut v: Vec<f64> = Vec::with_capacity(end - start);

        for i in start..end {
            v.push(i as f64);
        }

        let size = v.len();

        Mat::from_vec(v.into_boxed_slice(), &[1, size as i32])
    }

    pub fn _arange_step(start: usize, end: usize, step: usize) -> Mat<f64> {
        let mut v: Vec<f64> = Vec::with_capacity((end - start) / step);

        for i in (start..end).step_by(step) {
            v.push(i as f64);
        }

        let size = v.len();

        Mat::from_vec(v.into_boxed_slice(), &[size as i32])
    }

    #[macro_export]
    macro_rules! srange {
        ($size: expr, $shape: expr) => {
            crate::arange::_arange_default_shape($size, $shape)
        };

        ($start: expr, $end: expr, $shape: expr) => {
            crate::arange::_arange_start_shape($start, $end, $shape)
        };

        ($start: expr, $end: expr, $step: expr, $shape: expr) => {
            crate::arange::_arange_step_shape($start, $end, $step, $shape)
        };
    }

    pub fn _arange_default_shape(size: usize, shape: &[i32]) -> Mat<f64> {
        let mut v: Vec<f64> = Vec::with_capacity(size);

        for i in 0..size {
            v.push(i as f64);
        }

        Mat::from_vec(v.into_boxed_slice(), shape)
    }

    pub fn _arange_start_shape(start: usize, end: usize, shape: &[i32]) -> Mat<f64> {
        let mut v: Vec<f64> = Vec::with_capacity(end - start);

        for i in start..end {
            v.push(i as f64);
        }

        Mat::from_vec(v.into_boxed_slice(), shape)
    }

    pub fn _arange_step_shape(start: usize, end: usize, step: usize, shape: &[i32]) -> Mat<f64> {
        let mut v: Vec<f64> = Vec::with_capacity((end - start) / step);

        for i in (start..end).step_by(step) {
            v.push(i as f64);
        }

        Mat::from_vec(v.into_boxed_slice(), shape)
    }
}
