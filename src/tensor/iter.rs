use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::iter::FusedIterator;
use std::ptr::NonNull;

use crate::debug_assert_positive;
use crate::tensor::Dimension;
use crate::tensor::layout::Layout;

pub struct ContiguousIter<'a, T: Copy> {
    data: RwLockReadGuard<'a, Box<[T]>>,
    index: usize,
}

impl<'a, T: Copy> ContiguousIter<'a, T> {
    pub fn new(lock: &'a RwLock<Box<[T]>>) -> Self {
        let data: RwLockReadGuard<'_, Box<[T]>> = lock.read();
        Self { data, index: 0 }
    }
}

impl<'a, T: Copy> Iterator for ContiguousIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let item = &self.data[self.index] as *const T;
        self.index += 1;

        return Some(unsafe { &*item });
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.data.len() - self.index;

        (len, Some(len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for ContiguousIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for ContiguousIter<'a, T> {}

// /////////////////////////////////////////////////////////////

pub struct MutContiguousIter<'a, T: Copy> {
    data: RwLockWriteGuard<'a, Box<[T]>>,
    index: usize,
}

impl<'a, T: Copy> MutContiguousIter<'a, T> {
    pub fn new(lock: &'a RwLock<Box<[T]>>) -> Self {
        let data: RwLockWriteGuard<'_, Box<[T]>> = lock.write();
        Self { data, index: 0 }
    }
}

impl<'a, T: Copy> Iterator for MutContiguousIter<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let mut item = NonNull::new(&mut self.data[self.index] as *mut T).unwrap();
        self.index += 1;

        return Some(unsafe { item.as_mut() });
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.data.len() - self.index;

        (len, Some(len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for MutContiguousIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for MutContiguousIter<'a, T> {}

// /////////////////////////////////////////////////////////////

pub struct SliceIter<'a, T: Copy> {
    data: RwLockReadGuard<'a, Box<[T]>>,
    pos: isize,
    counter: Box<[i32]>,
    shape: &'a [i32],
    adj_stride: &'a [i32],
    left_over: usize,
}

impl<'a, T: Copy> SliceIter<'a, T> {
    pub fn new(
        lock: &'a RwLock<Box<[T]>>,
        data_len: usize,
        shape: &'a [i32],
        adj_stride: &'a [i32],
    ) -> Self {
        let counter = vec![0; shape.len()].into_boxed_slice();

        Self {
            data: lock.read(),
            pos: 0,
            shape,
            adj_stride,
            counter: counter,
            left_over: data_len,
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

        let pos = self.pos as usize;

        unsafe {
            let item = &self.data[pos] as *const T;
            self.pos += self.adj_stride[step_dim] as isize;
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
    data: RwLockWriteGuard<'a, Box<[T]>>,
    pos: isize,
    counter: Box<[i32]>,
    shape: &'a [i32],
    adj_stride: &'a [i32],
    left_over: usize,
}

impl<'a, T: Copy> MutSliceIter<'a, T> {
    pub fn new(
        lock: &'a RwLock<Box<[T]>>,
        data_len: usize,
        shape: &'a [i32],
        adj_stride: &'a [i32],
    ) -> Self {
        let counter = vec![0; shape.len()].into_boxed_slice();

        Self {
            data: lock.write(),
            pos: 0,
            shape,
            adj_stride,
            counter: counter,
            left_over: data_len,
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
            let item_ptr = &mut self.data[self.pos as usize] as *mut T;
            self.pos += step as isize;
            self.left_over -= 1;

            Some(&mut *item_ptr)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Copy> ExactSizeIterator for MutSliceIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for MutSliceIter<'a, T> {}

/////////////////////////////////////////////////////////////

pub enum StepInfo<T: Copy> {
    EnterDimension(usize),
    ExitDimension(usize),
    Value(T),
    End,
}

pub struct InformedSliceIter<'a, T: Copy> {
    buffer: RwLockReadGuard<'a, Box<[T]>>,
    layout: &'a Layout,
    current_state: StepInfo<T>,
    pos: i64,
    len: usize,
    counter: Vec<i32>,
}

impl<'a, T: Copy> InformedSliceIter<'a, T> {
    pub fn new(buffer: RwLockReadGuard<'a, Box<[T]>>, iter_len: usize, layout: &'a Layout) -> Self {
        let len = layout.shape().len();

        Self {
            buffer,
            layout,
            current_state: StepInfo::<T>::EnterDimension(0),
            pos: 0,
            len: iter_len,
            counter: vec![0; len],
        }
    }
}

impl<'a, T: Copy> Iterator for InformedSliceIter<'a, T> {
    type Item = StepInfo<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current_state {
            StepInfo::EnterDimension(dim) => {
                if dim == self.layout.shape().len() - 1 {
                    debug_assert_positive!(self.pos);

                    self.current_state = StepInfo::Value(self.buffer[self.pos as usize]);

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

                if self.counter[dim - 1] == self.layout.shape()[dim - 1] {
                    self.current_state = StepInfo::ExitDimension(dim - 1);
                    return Some(StepInfo::ExitDimension(dim));
                }

                self.pos += self.layout.adj_stride()[dim - 1] as i64;
                self.current_state = StepInfo::EnterDimension(dim);

                Some(StepInfo::ExitDimension(dim))
            }
            StepInfo::Value(v) => {
                let counter_last = self.counter.len() - 1;

                if *self.counter.last().unwrap() == *self.layout.shape().last().unwrap() - 1 {
                    self.current_state = StepInfo::ExitDimension(self.counter.len() - 1);
                    self.counter[counter_last] = 0;

                    return Some(StepInfo::Value(v));
                }

                self.pos += *self.layout.adj_stride().last().unwrap() as i64;
                self.counter[counter_last] += 1;

                debug_assert_positive!(self.pos);

                self.current_state = StepInfo::Value(self.buffer[self.pos as usize]);

                Some(StepInfo::Value(v))
            }
            StepInfo::End => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len - self.pos as usize;

        (len, Some(len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for InformedSliceIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for InformedSliceIter<'a, T> {}
