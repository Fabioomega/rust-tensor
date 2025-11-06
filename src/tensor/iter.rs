use std::{iter::FusedIterator, marker::PhantomData, ptr::NonNull};

use crate::{
    debug_assert_positive,
    tensor::{
        Dimension,
        device::{CPU, DeviceInfo},
        mat::RawTensor,
        traits::{LinearIndexMemory, MutLinearIndexMemory, MutSliceable, Sliceable},
    },
};

pub struct ContiguousIter<'a, T, D: DeviceInfo> {
    tensor: &'a RawTensor<T, D>,
    pos: usize,
}

impl<'a, T: Copy, D: DeviceInfo> ContiguousIter<'a, T, D> {
    pub fn new(tensor: &'a RawTensor<T, D>) -> Self {
        Self { tensor, pos: 0 }
    }
}

impl<'a, T: Copy> Iterator for ContiguousIter<'a, T, CPU> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.tensor.len() {
            return None;
        }

        self.pos += 1;
        return Some(self.tensor.index_memory(self.pos - 1));
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.tensor.len() - self.pos;

        (len, Some(len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for ContiguousIter<'a, T, CPU> {}

impl<'a, T: Copy> FusedIterator for ContiguousIter<'a, T, CPU> {}

/////////////////////////////////////////////////////////////

pub struct MutContiguousIter<'a, T> {
    ptr: NonNull<T>,
    end: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: Copy> MutContiguousIter<'a, T> {
    pub fn new(ptr: NonNull<T>, len: usize) -> Self {
        Self {
            ptr,
            end: unsafe { ptr.add(len) },
            _marker: PhantomData {},
        }
    }
}

impl<'a, T: Copy> Iterator for MutContiguousIter<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr >= self.end {
            return None;
        }

        unsafe {
            self.ptr = self.ptr.offset(1);

            Some(self.ptr.offset(-1).as_mut())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe { self.end.offset_from_unsigned(self.ptr) };
        (len, Some(len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for MutContiguousIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for MutContiguousIter<'a, T> {}

/////////////////////////////////////////////////////////////

pub struct SliceIter<'a, T: Copy> {
    ptr: *const T,
    shape: &'a [i32],
    adj_stride: &'a [i32],
    counter: Box<[i32]>,
    left_over: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: Copy> SliceIter<'a, T> {
    pub fn new(ptr: *const T, data_len: usize, shape: &'a [i32], adj_stride: &'a [i32]) -> Self {
        let counter = vec![0; shape.len()].into_boxed_slice();

        Self {
            ptr,
            shape,
            adj_stride,
            counter: counter,
            left_over: data_len,
            _marker: PhantomData {},
        }
    }
}

impl<'a, T: Copy> Iterator for SliceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let last = self.counter.len() - 1;
        self.counter[last] += 1;
        let mut step_dim = last;

        for dim in (1..self.counter.len()).rev() {
            if self.counter[dim] == self.shape[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim - 1;
                continue;
            }
            break;
        }

        unsafe {
            let item = self.ptr;
            self.ptr = self.ptr.offset(self.adj_stride[step_dim] as isize);
            self.left_over -= 1;

            Some(&*item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Copy> ExactSizeIterator for SliceIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for SliceIter<'a, T> {}

///////////////////////////////////////////////////////////////

pub struct MutSliceIter<'a, T: Copy> {
    ptr: NonNull<T>,
    shape: &'a [i32],
    adj_stride: &'a [i32],
    counter: Box<[i32]>,
    left_over: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: Copy> MutSliceIter<'a, T> {
    pub fn new(ptr: NonNull<T>, data_len: usize, shape: &'a [i32], adj_stride: &'a [i32]) -> Self {
        let counter = vec![0; shape.len()].into_boxed_slice();

        Self {
            ptr,
            shape,
            adj_stride,
            counter,
            left_over: data_len,
            _marker: PhantomData {},
        }
    }
}

impl<'a, T: Copy> Iterator for MutSliceIter<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let last = self.counter.len() - 1;
        self.counter[last] += 1;
        let mut step_dim = last;

        for dim in (1..self.counter.len()).rev() {
            if self.counter[dim] == self.shape[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim - 1;
                continue;
            }
            break;
        }

        let step = self.adj_stride[step_dim];
        unsafe {
            let mut item_ptr = self.ptr;
            self.ptr = self.ptr.offset(step as isize);
            self.left_over -= 1;

            Some(item_ptr.as_mut())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Copy> ExactSizeIterator for MutSliceIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for MutSliceIter<'a, T> {}

/////////////////////////////////////////////////////////////

pub enum StepInfo<'a, T: Copy> {
    EnterDimension(usize),
    ExitDimension(usize),
    Value(&'a T),
    End,
}

pub struct InformedSliceIter<'a, T: Copy, C: Sliceable> {
    slice: &'a C,
    current_state: StepInfo<'a, T>,
    pos: i64,
    counter: Vec<i32>,
}

impl<'a, T: Copy, C: Sliceable> InformedSliceIter<'a, T, C> {
    pub fn new(slice: &'a C) -> Self {
        Self {
            slice,
            current_state: StepInfo::<T>::EnterDimension(0),
            pos: 0,
            counter: vec![0; slice.shape().len()],
        }
    }
}

impl<'a, T: Copy, C: Sliceable<Output = T>> Iterator for InformedSliceIter<'a, T, C> {
    type Item = StepInfo<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current_state {
            StepInfo::EnterDimension(dim) => {
                if dim == self.slice.shape().len() - 1 {
                    debug_assert_positive!(self.pos);

                    self.current_state =
                        StepInfo::Value(self.slice.index_memory(self.pos as usize));

                    return Some(StepInfo::EnterDimension(dim));
                }

                self.current_state = StepInfo::EnterDimension(dim + 1);

                Some(StepInfo::EnterDimension(dim))
            }
            StepInfo::ExitDimension(dim) => {
                if dim == 0 {
                    self.current_state = StepInfo::End;
                    return Some(StepInfo::ExitDimension(dim));
                }

                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                if self.counter[dim - 1] == self.slice.shape()[dim - 1] {
                    self.current_state = StepInfo::ExitDimension(dim - 1);
                    return Some(StepInfo::ExitDimension(dim));
                }

                self.pos += self.slice.adj_stride()[dim - 1] as i64;
                self.current_state = StepInfo::EnterDimension(dim);

                Some(StepInfo::ExitDimension(dim))
            }
            StepInfo::Value(v) => {
                let counter_last = self.counter.len() - 1;

                if *self.counter.last().unwrap() == *self.slice.shape().last().unwrap() - 1 {
                    self.current_state = StepInfo::ExitDimension(self.counter.len() - 1);
                    self.counter[counter_last] = 0;

                    return Some(StepInfo::Value(v));
                }

                self.pos += *self.slice.adj_stride().last().unwrap() as i64;
                self.counter[counter_last] += 1;

                debug_assert_positive!(self.pos);

                self.current_state = StepInfo::Value(self.slice.index_memory(self.pos as usize));

                Some(StepInfo::Value(v))
            }
            StepInfo::End => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.slice.len() - self.pos as usize;

        (len, Some(len))
    }
}

impl<'a, T: Copy, C: Sliceable<Output = T>> ExactSizeIterator for InformedSliceIter<'a, T, C> {}

impl<'a, T: Copy, C: Sliceable<Output = T>> FusedIterator for InformedSliceIter<'a, T, C> {}
