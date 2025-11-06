use crate::tensor::{Mat, arange};

mod tensor;

fn main() {
    let mat = srange![27, &[3, 3, 3]];
    let test = mat.slice(s![.., 1..2, 1..2]);

    println!("This is a matrix:\n {}", mat);
    println!("This is a slice:\n {}", test);
}
