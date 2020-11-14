# HPG: high performance gridder

**Confidential**, at least for the time being. Please do not share
access to this repository or any clone of this repository without
obtaining prior approval from [Martin
Pokorny](mailto:mpokorny@nrao.edu)

High performance gridding (kernel) implementation library.

## Build requirements

* [CMake](https://cmake.org) (v3.14 or later)
* [Kokkos](https://github.com/kokkos/kokkos) (v3.2.00 or later)
* [FFTW](http://fftw.org) (v3.3.8)
* [CUDA](https://developer.nvidia.com/cuda-toolkit) (optional, v11.0.2 or later)
* cuFFT (part of CUDA toolkit)

Compilers:
* [gcc](https://gcc.gnu.org), v9.3.0 or later
* [clang](https://clang.llvm.org), v10.0.0 or later
* *nvcc*, *via* *Kokkos'* `nvcc_wrapper` if *gcc* and *CUDA*

## Build instructions

CMake build options:
* `Hpg_ENABLE_SERIAL`: enable serial implementation
* `Hpg_ENABLE_OPENMP`: enable OpenMP implementation
* `Hpg_ENABLE_CUDA`: enable CUDA implementation

Note that the selected implementation(s) must be enabled in the *Kokkos*
library installation. When building for *CUDA* using *gcc*, *Kokkos'*
`nvcc_wrapper` must be used to build *HPG*.

## Using libhpg

The *HPG* installation includes a *CMake* package configuration file
that defines the `hpg` target. To build an executable with the target
name `MyProgram` against the *libhpg* library, simply include the
following statement in `CMakeLists.txt`:

``` cmake
target_link_libraries(MyProgram hpg)
```

Client code should include only the header `hpg.hpp` (in particular, such code must not include `hpg_impl.hpp`).
