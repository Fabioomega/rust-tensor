use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::Matches;

use crate::tensor::definitions::NumberLike;
use crate::tensor::graph::NodeKind;
use crate::tensor::ops::impl_compute_op::{OpKind, OpScalarKind};

///////////////////////////////////////////

#[derive(Debug)]
pub(crate) struct Fusion<T: Copy> {
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T>]>,
}

pub fn try_fuse<T: NumberLike>(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Fusion<T> {
    let mut current_fusion: Fusion<T> = Fusion {
        op,
        inputs: inputs.clone(),
    };

    for (idx, inp) in inputs.iter().enumerate() {
        match inp {
            NodeKind::Edge(_) => continue,
            NodeKind::Node(node) => {
                let fused = compute_fusion(
                    &node.op,
                    &node.inputs,
                    &current_fusion.op,
                    &current_fusion.inputs,
                    idx,
                );

                if let Some(f) = fused {
                    current_fusion = f;
                }
            }
            NodeKind::Cache(cache) => {
                let node = cache.get_node();

                let fused = compute_fusion(
                    &node.op,
                    &node.inputs,
                    &current_fusion.op,
                    &current_fusion.inputs,
                    idx,
                );

                if let Some(f) = fused {
                    current_fusion = f;
                }
            }
        }
    }

    current_fusion
}

fn fuse_sum_scalar<T: NumberLike>(
    op1: &OpScalarKind<T>, // Parent
    inputs1: &[NodeKind<T>],
    op2: &OpScalarKind<T>, // Child
) -> Fusion<T> {
    let s1: T = match op1 {
        OpScalarKind::Sum(scalar) => *scalar,
        OpScalarKind::Sub(scalar) => -*scalar,
        _ => unreachable!("You fucked something up!"),
    };

    let s2: T = match op2 {
        OpScalarKind::Sum(scalar) => *scalar,
        OpScalarKind::Sub(scalar) => -*scalar,
        _ => unreachable!("You fucked something up!"),
    };

    Fusion {
        op: OpKind::ScalarOp(OpScalarKind::Sum(s1 + s2)),
        inputs: inputs1.into(),
    }
}

fn fuse_mul_scalar<T: NumberLike>(
    op1: &OpScalarKind<T>, // Parent
    inputs1: &[NodeKind<T>],
    op2: &OpScalarKind<T>, // Child
) -> Fusion<T> {
    match op1 {
        OpScalarKind::Mul(s1) => match op2 {
            OpScalarKind::Mul(s2) => Fusion {
                op: OpKind::ScalarOp(OpScalarKind::Mul(*s1 * *s2)),
                inputs: inputs1.into(),
            },
            OpScalarKind::Div(s2) => Fusion {
                op: OpKind::ScalarOp(OpScalarKind::Mul(*s1 / *s2)),
                inputs: inputs1.into(),
            },
            _ => unreachable!("no other op should appear here"),
        },
        OpScalarKind::Div(s1) => match op2 {
            OpScalarKind::Mul(s2) => Fusion {
                op: OpKind::ScalarOp(OpScalarKind::Mul(*s2 / *s1)),
                inputs: inputs1.into(),
            },
            OpScalarKind::Div(s2) => Fusion {
                op: OpKind::ScalarOp(OpScalarKind::Div(*s1 * *s2)),
                inputs: inputs1.into(),
            },
            _ => unreachable!("no other op should appear here"),
        },
        _ => unreachable!("no other op should appear here"),
    }
}

#[inline]
fn fuse_scalars_into_combination<T: NumberLike>(
    op1: &OpScalarKind<T>,
    inputs1: &[NodeKind<T>],
    op2: &OpScalarKind<T>,
) -> Fusion<T> {
    let ops = Box::new([op1.clone(), op2.clone()]);
    Fusion {
        op: OpKind::FusedScalar(ops),
        inputs: inputs1.into(),
    }
}

fn fuse_scalars<T: NumberLike>(
    op1: &OpScalarKind<T>,
    inputs1: &[NodeKind<T>],
    op2: &OpScalarKind<T>,
) -> Fusion<T> {
    match op1 {
        OpScalarKind::Sum(_) => match op2 {
            OpScalarKind::Sum(_) => fuse_sum_scalar(op1, inputs1, op2),
            OpScalarKind::Sub(_) => fuse_sum_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
        OpScalarKind::Sub(_) => match op2 {
            OpScalarKind::Sum(_) => fuse_sum_scalar(op1, inputs1, op2),
            OpScalarKind::Sub(_) => fuse_sum_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
        OpScalarKind::Mul(_) => match op2 {
            OpScalarKind::Mul(_) => fuse_mul_scalar(op1, inputs1, op2),
            OpScalarKind::Div(_) => fuse_mul_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
        OpScalarKind::Div(_) => match op2 {
            OpScalarKind::Mul(_) => fuse_mul_scalar(op1, inputs1, op2),
            OpScalarKind::Div(_) => fuse_mul_scalar(op1, inputs1, op2),
            _ => fuse_scalars_into_combination(op1, inputs1, op2),
        },
    }
}

fn fuse_scalar_combination<T: NumberLike>(
    ops: &[OpScalarKind<T>],
    inputs1: &[NodeKind<T>],
    op2: &OpScalarKind<T>,
) -> Fusion<T> {
    let tail = &ops[ops.len() - 1];
    let fused = fuse_scalars(tail, inputs1, op2);

    let op = fused.op;
    let inputs = fused.inputs;

    let new_ops = match op {
        OpKind::FusedScalar(_) => {
            let mut vec: Vec<OpScalarKind<T>> = Vec::with_capacity(ops.len() + 1);
            vec.extend(ops[..ops.len() - 1].iter().cloned());
            vec.push(op2.clone());

            vec.into_boxed_slice()
        }
        OpKind::ScalarOp(scalar_op) => {
            let mut vec: Vec<OpScalarKind<T>> = Vec::with_capacity(ops.len());
            vec.extend(ops[..ops.len() - 1].iter().cloned());
            vec.push(scalar_op.clone());

            vec.into_boxed_slice()
        }
        _ => unreachable!("not other options should be returned from fuse_scalars"),
    };

    Fusion {
        op: OpKind::FusedScalar(new_ops),
        inputs,
    }
}

pub fn compute_fusion<T>(
    op1: &OpKind<T>, // This is the father operand
    inputs1: &[NodeKind<T>],
    op2: &OpKind<T>, // This is the child operand
    inputs2: &[NodeKind<T>],
    skip_input_idx: usize, // Skips one of the inputs2 Nodes
) -> Option<Fusion<T>>
where
    T: NumberLike,
{
    match op1 {
        OpKind::ScalarOp(s1) => match op2 {
            OpKind::ScalarOp(s2) => Some(fuse_scalars(s1, inputs1, s2)),
            _ => None,
        },
        OpKind::FusedScalar(ops) => match op2 {
            OpKind::ScalarOp(s2) => Some(fuse_scalar_combination(ops, inputs1, s2)),
            _ => None,
        },

        _ => None,
    }
}
