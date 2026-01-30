use std::boxed::Box;
use std::cell::OnceCell;
use std::sync::Arc;

use crate::tensor::definitions::NumberLike;
use crate::tensor::layout::Layout;
use crate::tensor::ops::fusion::try_fuse;
use crate::tensor::ops::impl_compute_op::OpKind;
use crate::tensor::ops::{ComputeWrapperSpec, compute_layout, cpu_compute};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Promising;

#[derive(Clone, Debug)]
pub enum NodeKind<T: Copy> {
    Edge(Arc<TensorGraphEdge<T>>),
    Cache(Arc<TensorGraphCacheNode<T>>),
    Node(Arc<TensorGraphNode<T>>),
}

//////////////////////////////////////////////////////////////////////////////////

pub fn get_inputs_layout<T: NumberLike>(inputs: &[NodeKind<T>]) -> Box<[&Layout]> {
    inputs
        .iter()
        .map(|node| match &node {
            NodeKind::Edge(edge) => edge.get().layout(),
            NodeKind::Node(node) => &node.layout,
            NodeKind::Cache(cache) => &cache.get_node().layout,
        })
        .collect()
}

//////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct TensorGraphEdge<T: Copy> {
    data: TensorData<T>,
}

impl<T: Copy> TensorGraphEdge<T> {
    pub fn from_tensor_data(data: TensorData<T>) -> Self {
        Self { data }
    }

    pub fn get(&self) -> &TensorData<T> {
        &self.data
    }
}

impl<T: Copy> Promising for TensorGraphEdge<T> {
    type Output = T;

    #[inline]
    fn compute(&self) -> TensorData<T> {
        self.data.clone_reference()
    }

    #[inline]
    fn layout(&self) -> &Layout {
        self.data.layout()
    }
}

impl<T: Copy> Clone for TensorGraphEdge<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone_reference(),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct TensorGraphNode<T: Copy> {
    pub(crate) op: OpKind<T>,
    pub(crate) inputs: Box<[NodeKind<T>]>,
    pub(crate) layout: Layout,
}

impl<T: NumberLike> TensorGraphNode<T> {
    // Panics if the layout is fucked for the specified operation
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Self {
        let fused = try_fuse(op, inputs);

        let layouts = get_inputs_layout(&fused.inputs);
        let layout = compute_layout(&fused.op, &layouts);

        if let Err(err) = layout {
            panic!("{}", err);
        }

        let unchecked_layout = unsafe { layout.unwrap_unchecked() };

        Self {
            op: fused.op,
            inputs: fused.inputs,
            layout: unchecked_layout,
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        let fused = try_fuse(op, inputs);

        Self {
            op: fused.op,
            inputs: fused.inputs,
            layout,
        }
    }
}

impl<T: NumberLike + ComputeWrapperSpec> Promising for TensorGraphNode<T> {
    type Output = T;

    // TODO: Make this not recursive so that big chains won't cause war crimes
    fn compute(&self) -> TensorData<T> {
        let inputs: Box<[TensorData<T>]> = self
            .inputs
            .iter()
            .map(|x| match &x {
                NodeKind::Node(node) => node.compute(),
                NodeKind::Edge(edge) => edge.compute(),
                NodeKind::Cache(cache) => cache.compute(),
            })
            .collect();

        cpu_compute(&self.op, &inputs)
    }

    #[inline]
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

//////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct TensorGraphCacheNode<T: Copy> {
    node: TensorGraphNode<T>,
    cache: OnceCell<TensorData<T>>,
}

impl<T: NumberLike> TensorGraphCacheNode<T> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Self {
        Self {
            node: TensorGraphNode::new(op, inputs),
            cache: OnceCell::new(),
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        Self {
            node: TensorGraphNode::with_layout(op, inputs, layout),
            cache: OnceCell::new(),
        }
    }

    pub fn from_node(node: TensorGraphNode<T>) -> Self {
        Self {
            node,
            cache: OnceCell::new(),
        }
    }

    pub fn get_node(&self) -> &TensorGraphNode<T> {
        &self.node
    }
}

impl<T: NumberLike + ComputeWrapperSpec> Promising for TensorGraphCacheNode<T> {
    type Output = T;

    fn compute(&self) -> TensorData<T> {
        if let Some(tensor) = self.cache.get() {
            tensor.clone_reference()
        } else {
            let tensor = self.node.compute();
            let _ = self.cache.set(tensor.clone_reference());

            tensor
        }
    }

    #[inline]
    fn layout(&self) -> &Layout {
        &self.get_node().layout
    }
}

//////////////////////////////////////////////////////////////////////////////////
