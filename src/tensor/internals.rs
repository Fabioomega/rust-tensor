pub(super) fn calculate_dim_stride(shape: &[i32]) -> Vec<i32> {
    let mut v: Vec<i32> = Vec::new();
    v.resize(shape.len(), 1);

    for i in (0..shape.len().saturating_sub(1)).rev() {
        v[i] = shape[i + 1] * v[i + 1];
    }

    v
}

pub(super) fn calculate_adjacent_dim_stride(stride: &[i32], slice_shape: &[i32]) -> Vec<i32> {
    let mut v: Vec<i32> = stride.to_vec();

    let mut accum: i32 = 0;
    for i in (0..stride.len() - 1).rev() {
        accum += stride[i + 1] * (slice_shape[i + 1] - 1);
        v[i] -= accum;
    }

    v
}
