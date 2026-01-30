use crate::tensor::definitions::{ChunkedIter, NumberLike};
use crate::tensor::errors::OpError;
use crate::tensor::layout::Layout;
use crate::tensor::mkl_extension::cblas_dscal;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Dimension, StreamingIterator};
use intel_mkl_sys::{vdAdd, vdDiv, vdMul, vdSub};

#[derive(Clone, Debug)]
pub enum OpScalarKind<T: Copy> {
    Sum(T),
    Sub(T),
    Mul(T),
    Div(T),
}

#[derive(Clone, Debug)]
pub enum OpKind<T: Copy> {
    NoOp,
    Add,
    Sub,
    Mul,
    Div,
    ScalarOp(OpScalarKind<T>),
    FusedScalar(Box<[OpScalarKind<T>]>),
    View(Layout),
}

fn compute_scalar_op<T: NumberLike>(
    op: &OpScalarKind<T>,
    mut input: Box<[T]>,
    dscal: unsafe extern "C" fn(i32, T, *mut T, i32),
) -> Box<[T]> {
    match op {
        OpScalarKind::Sum(scalar) => {
            for el in input.iter_mut() {
                *el = *el + *scalar;
            }
            input
        }
        OpScalarKind::Sub(scalar) => {
            for el in input.iter_mut() {
                *el = *el + *scalar;
            }
            input
        }
        OpScalarKind::Mul(scalar) => {
            unsafe { dscal(input.len() as i32, *scalar, input.as_mut_ptr(), 1) };
            input
        }
        OpScalarKind::Div(scalar) => {
            unsafe { dscal(input.len() as i32, *scalar, input.as_mut_ptr(), 1) };
            input
        }
    }
}

fn cpu_compute_op_f64(op: &OpKind<f64>, inputs: &[TensorData<f64>]) -> TensorData<f64> {
    println!("Calling compute! op = {:?}", op);

    match op {
        OpKind::ScalarOp(op) => {
            let buffer = Vec::from_iter(inputs[0].copied_iter()).into_boxed_slice();
            TensorData::from_vec(
                compute_scalar_op(op, buffer, cblas_dscal),
                inputs[0].shape(),
            )
        }
        OpKind::FusedScalar(ops) => {
            let mut buffer = Vec::from_iter(inputs[0].copied_iter()).into_boxed_slice();

            for op in ops {
                buffer = compute_scalar_op(op, buffer, cblas_dscal);
            }

            TensorData::from_vec(buffer, inputs[0].shape())
        }
        OpKind::View(new_layout) => inputs[0].as_layout(new_layout.clone()),
        OpKind::Add => {
            let mut output_data = Vec::from_iter(inputs[0].copied_iter());
            let mut packed_iter: ChunkedIter<'_, f64> = inputs[1].packed_iter();

            while let Some(chunk) = packed_iter.next() {
                let buffer_size: usize = chunk.packing_buffer.len();

                unsafe {
                    vdAdd(
                        buffer_size as i32,
                        output_data.as_ptr().add(chunk.absolute_buffer_position),
                        chunk.packing_buffer.as_ptr(),
                        output_data.as_mut_ptr().add(chunk.absolute_buffer_position),
                    )
                }
            }

            TensorData::from_vec(output_data.into(), inputs[0].shape())
        }
        OpKind::Sub => {
            let mut output_data = Vec::from_iter(inputs[0].copied_iter());
            let mut packed_iter: ChunkedIter<'_, f64> = inputs[1].packed_iter();

            while let Some(chunk) = packed_iter.next() {
                let buffer_size: usize = chunk.packing_buffer.len();

                unsafe {
                    vdSub(
                        buffer_size as i32,
                        output_data.as_ptr().add(chunk.absolute_buffer_position),
                        chunk.packing_buffer.as_ptr(),
                        output_data.as_mut_ptr().add(chunk.absolute_buffer_position),
                    )
                }
            }

            TensorData::from_vec(output_data.into(), inputs[0].shape())
        }
        OpKind::Mul => {
            let mut output_data = Vec::from_iter(inputs[0].copied_iter());
            let mut packed_iter: ChunkedIter<'_, f64> = inputs[1].packed_iter();

            while let Some(chunk) = packed_iter.next() {
                let buffer_size: usize = chunk.packing_buffer.len();

                unsafe {
                    vdMul(
                        buffer_size as i32,
                        output_data.as_ptr().add(chunk.absolute_buffer_position),
                        chunk.packing_buffer.as_ptr(),
                        output_data.as_mut_ptr().add(chunk.absolute_buffer_position),
                    )
                }
            }

            TensorData::from_vec(output_data.into(), inputs[0].shape())
        }
        OpKind::Div => {
            let mut output_data = Vec::from_iter(inputs[0].copied_iter());
            let mut packed_iter: ChunkedIter<'_, f64> = inputs[1].packed_iter();

            while let Some(chunk) = packed_iter.next() {
                let buffer_size: usize = chunk.packing_buffer.len();

                unsafe {
                    vdDiv(
                        buffer_size as i32,
                        output_data.as_ptr().add(chunk.absolute_buffer_position),
                        chunk.packing_buffer.as_ptr(),
                        output_data.as_mut_ptr().add(chunk.absolute_buffer_position),
                    )
                }
            }

            TensorData::from_vec(output_data.into(), inputs[0].shape())
        }
        OpKind::NoOp => inputs[0].clone_reference(),
        _ => todo!("not implemented"),
    }
}

pub fn compute_layout<'a, T: Copy>(
    op: &OpKind<T>,
    inputs: &[&'a Layout],
) -> Result<Layout, OpError<'a>> {
    match op {
        OpKind::ScalarOp(_) | OpKind::FusedScalar(_) | OpKind::NoOp => Ok(inputs[0].clone()),
        OpKind::View(new_layout) => Ok(new_layout.clone()),
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            if inputs[0].shape() == inputs[1].shape() {
                Ok(inputs[0].clone())
            } else {
                Err(OpError::NotSameShape(inputs[0].shape(), inputs[0].shape()))
            }
        }
        _ => todo!("not implemented"),
    }
}

pub trait ComputeWrapperSpec
where
    Self: Copy,
{
    fn compute_for_type(op: &OpKind<Self>, inputs: &[TensorData<Self>]) -> TensorData<Self>;
}

impl ComputeWrapperSpec for f64 {
    #[inline]
    fn compute_for_type(op: &OpKind<f64>, inputs: &[TensorData<f64>]) -> TensorData<f64> {
        cpu_compute_op_f64(op, inputs)
    }
}

#[inline]
pub fn cpu_compute<T: ComputeWrapperSpec>(
    op: &OpKind<T>,
    inputs: &[TensorData<T>],
) -> TensorData<T> {
    T::compute_for_type(op, inputs)
}
