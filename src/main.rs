use crate::tensor::{Dimension, Tensor, arange};

mod tensor;

fn main() {
    let t1 = arange![12];
    let mut p = t1.as_promise();
    for i in 0..20 {
        p = p + i as f64;
    }

    println!("{}", (p * 2.0).materialize());
}
