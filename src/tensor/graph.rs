use std::boxed::Box;
use std::cell::OnceCell;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::definitions::NumberLike;
use crate::tensor::layout::Layout;
use crate::tensor::ops::fusion::try_fuse;
use crate::tensor::ops::impl_compute_op::OpKind;
use crate::tensor::ops::{ComputeWrapperSpec, compute_layout, cpu_compute};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Promising;

static NEXT_ID: AtomicUsize = const { AtomicUsize::new(0) };

//////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub enum NodeKind<T: Copy> {
    Edge(Arc<TensorGraphEdge<T>>),
    Cache(Arc<TensorGraphCacheNode<T>>),
    Node(Arc<TensorGraphNode<T>>),
}

//////////////////////////////////////////////////////////////////////////////////

#[inline]
pub fn get_id<T: Copy>(node: &NodeKind<T>) -> usize {
    match node {
        NodeKind::Edge(edge) => edge.id,
        NodeKind::Node(node) => node.id,
        NodeKind::Cache(cache) => cache.node.id,
    }
}

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

fn get_inputs_tensor_data<T: Copy>(
    inputs: &[NodeKind<T>],
    computation_cache: &mut HashMap<usize, TensorData<T>>,
    reference_counter: &mut HashMap<usize, usize>,
) -> Vec<TensorData<T>> {
    let mut inputs_data: Vec<TensorData<T>> = Vec::with_capacity(inputs.len());
    for n in inputs.iter() {
        let id = get_id(n);

        // If this panics, you fucked up the topological sort, congrats!
        let tensor_data = if let Some(count) = reference_counter.get_mut(&id) {
            if *count == 1 {
                *count = 0;
                computation_cache.remove(&id).unwrap()
            } else {
                *count -= 1;
                computation_cache.get(&id).unwrap().clone_reference()
            }
        } else {
            unreachable!("if this panics you fucked up the topological sort, congrats!")
        };

        inputs_data.push(tensor_data);
    }

    inputs_data
}
//////////////////////////////////////////////////////////////////////////////////

pub struct TensorGraphEdge<T: Copy> {
    pub(crate) id: usize,
    data: TensorData<T>,
}

impl<T: Copy> TensorGraphEdge<T> {
    pub fn from_tensor_data(data: TensorData<T>) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data,
        }
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
            id: self.id,
            data: self.data.clone_reference(),
        }
    }
}

impl<T: Copy> Debug for TensorGraphEdge<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorGraphEdge {{ id: {}, data: [...] }}", self.id)
    }
}

//////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct TensorGraphNode<T: Copy> {
    pub(crate) id: usize,
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
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            op: fused.op,
            inputs: fused.inputs,
            layout: unchecked_layout,
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        let fused = try_fuse(op, inputs);

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            op: fused.op,
            inputs: fused.inputs,
            layout,
        }
    }

    // Performs a DFS topological sort on the current DAG that this leaf (sink) is part of.
    //  It should be iterated from right to left.
    // NOTE: This node is not added to the returning vec.
    //  but, naturally, would be the first element if added.
    // NOTE 2: If a cache and non-cache node with the same id are present in the same DAG,
    //  the cache will not be used. That will not be fixed as it would require
    //  invalidating some elements in the sorted.
    //  It's the user responsibility to use the cached node correctly.
    fn topological_sort(&self) -> (Vec<&NodeKind<T>>, HashMap<usize, usize>) {
        let mut sorted: Vec<&NodeKind<T>> = Vec::with_capacity(64);
        let mut reference_counter: HashMap<usize, usize> = HashMap::new();

        let mut current_inputs = VecDeque::from_iter(self.inputs.iter());

        while let Some(inp) = current_inputs.pop_front() {
            let id = get_id(inp);
            if let Some(count) = reference_counter.get_mut(&id) {
                *count += 1;
                continue;
            } else {
                reference_counter.insert(id, 1);
            }

            match inp {
                NodeKind::Edge(_) => {}
                NodeKind::Node(node) => current_inputs.extend(node.inputs.iter()),
                NodeKind::Cache(cache) => {
                    if !cache.is_cache_filled() {
                        current_inputs.extend(cache.get_node().inputs.iter())
                    }
                }
            }

            sorted.push(inp);
        }

        (sorted, reference_counter)
    }
}

impl<T: NumberLike + ComputeWrapperSpec> Promising for TensorGraphNode<T> {
    type Output = T;

    fn compute(&self) -> TensorData<T> {
        let (sorted_dag, mut reference_counter) = self.topological_sort();
        let mut computation_cache: HashMap<usize, TensorData<T>> = HashMap::new();

        for node in sorted_dag.into_iter().rev() {
            match node {
                NodeKind::Edge(edge) => {
                    computation_cache.insert(edge.id, edge.compute());
                }
                NodeKind::Node(node) => {
                    let inputs: Vec<TensorData<T>> = get_inputs_tensor_data(
                        &node.inputs,
                        &mut computation_cache,
                        &mut reference_counter,
                    );

                    let result = cpu_compute(&node.op, &inputs);
                    computation_cache.insert(node.id, result);
                }
                NodeKind::Cache(cache) => {
                    let tensor_data = if cache.is_cache_filled() {
                        unsafe { cache.cache.get().unwrap_unchecked().clone_reference() }
                    } else {
                        let inputs: Vec<TensorData<T>> = get_inputs_tensor_data(
                            &cache.node.inputs,
                            &mut computation_cache,
                            &mut reference_counter,
                        );

                        let result = cpu_compute(&cache.node.op, &inputs);
                        let _ = cache.cache.set(result.clone_reference());
                        result
                    };

                    computation_cache.insert(cache.node.id, tensor_data);
                }
            }
        }

        let inputs: Vec<TensorData<T>> =
            get_inputs_tensor_data(&self.inputs, &mut computation_cache, &mut reference_counter);

        cpu_compute(&self.op, &inputs)
    }

    #[inline]
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Copy + Debug> Debug for TensorGraphNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorGraphNode {{ id: {:?}, op: {:?},  inputs: [...] }}",
            self.id, self.op
        )
    }
}

//////////////////////////////////////////////////////////////////////////////////

pub struct TensorGraphCacheNode<T: Copy> {
    node: TensorGraphNode<T>,
    cache: OnceCell<TensorData<T>>,
}

impl<T: Copy> TensorGraphCacheNode<T> {
    pub fn from_node(node: TensorGraphNode<T>) -> Self {
        Self {
            node,
            cache: OnceCell::new(),
        }
    }

    pub fn get_node(&self) -> &TensorGraphNode<T> {
        &self.node
    }

    pub fn is_cache_filled(&self) -> bool {
        self.cache.get().is_some()
    }
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

impl<T: Copy + Debug> Debug for TensorGraphCacheNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TensorGraphNode {{ id: {:?}, op: {:?},  inputs: [...], cached: {} }}",
            self.node.id,
            self.node.op,
            self.is_cache_filled()
        )
    }
}

//////////////////////////////////////////////////////////////////////////////////
