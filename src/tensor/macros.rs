#[macro_export]
macro_rules! debug_only {
     ($($stmt:stmt)*) => {
         #[cfg(debug_assertions)]
         { $($stmt)* }
    };
}

#[macro_export]
macro_rules! debug_assert_positive {
    ($e: expr) => {
        debug_assert!($e >= 0)
    };
}

#[macro_export]
macro_rules! debug_shape_check {
    ($size_a: expr, $size_b: expr) => {
        debug_only!(if $size_a != $size_b {
            panic!(
                "The shape of both matrixes does not match! Expected {:?} but got {:?}",
                $size_a, $size_b
            );
        });
    };
}
