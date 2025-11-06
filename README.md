# ARTL (A Rust Tensor Library)
The aim of this project is to both implement a simple CNN like neural net with it's own tensor library with automatic differentiation and CUDA support via [CUDARC](https://github.com/coreylowman/cudarc).

## The Vision
As of right now this library only implements simple iterators and some matrix operations; inspired by [ndarray](https://github.com/rust-ndarray/ndarray). In the future it will have full support for BLAS operations and graphs trough the use of [Intel OneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) and [CUDARC](https://github.com/coreylowman/cudarc).
While safety and ergonomics is a goal of this project it will not be it's main goal given that is mainly for learning purposes. 
