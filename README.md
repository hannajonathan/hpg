# HPG: high performance gridder

High performance gridding (kernel) implementation library.

## Build requirements

* [CMake](https://cmake.org) (v3.14 or later)
* [Kokkos](https://github.com/kokkos/kokkos) (v4.0.00 or later)
* [FFTW](http://fftw.org) (v3.3.8 or later)
* [CUDA](https://developer.nvidia.com/cuda-toolkit) (optional, v11.0.2 or later)
* cuFFT (part of CUDA toolkit)

Compilers:
* [gcc](https://gcc.gnu.org), v9.3.0 or later
* [clang](https://clang.llvm.org), v10.0.0 or later
* *nvcc*, *via* *Kokkos'* `nvcc_wrapper` if using *gcc* and *CUDA*

## Build instructions

CMake build options:
* `Hpg_ENABLE_ALL`: enable all devices enabled by Kokkos installation
  (default: `ON`)
* `Hpg_ENABLE_SERIAL`: enable serial implementation (default: value of
  `Hpg_ENABLE_ALL`)
* `Hpg_ENABLE_OPENMP`: enable OpenMP implementation (default: value of
  `Hpg_ENABLE_ALL`)
* `Hpg_ENABLE_CUDA`: enable CUDA implementation (default: value of
  `Hpg_ENABLE_ALL`)
* `Hpg_API`: C++ standard for top-level HPG API (default: 11)
* `Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS`: enable experimental
  compute kernel implementations (advanced feature, default: OFF)
* `Hpg_MAX_NUM_CF_GROUPS`: maximum number of convolution function
  groups (default: 1000)

Note that the selected implementation(s) must be enabled in the *Kokkos*
library installation. When building for *CUDA* using *gcc*, *Kokkos'*
`nvcc_wrapper` must be used to build *HPG*.

The value of `Hpg_MAX_NUM_CF_GROUPS` sets a compile-time limit on the
size of a vector in the implementation. The "groups" in the name
refers to the rightmost index of the `hpg::CFArray::operator()`
method. There is currently no reliable check on whether that
configured maximum is exceeded by an instance of `hpg::CFArray` at
runtime; therefore, err on the side of caution when setting this
value.

Building *libhpg* requires a compiler that supports C++17 or later,
although the C++ language standard of the *HPG* API can be selected by
the user as noted above.

### Installation using Spack

Installation of *HPG* *via* [Spack](https://spack.io) is
supported. Building and installing *HPG* by this path is, in many
ways, the simplest alternative, although *Spack* itself is naturally a
prerequisite. The *Spack* package file for *HPG* is available in the
[hpg-spack](https://gitlab.nrao.edu/mpokorny/hpg-spack) repository. To
build and install *HPG* it is sufficient to copy the package file
alone to a *Spack* repository, although cloning the
[hpg-spack](https://gitlab.nrao.edu/mpokorny/hpg-spack) git repository
into the *Spack* repository is nearly as simple. Alternatively, if one
is intending to do development on *HPG*, cloning the *HPG* git
repository is also necessary. As an example, the following single
command is sufficient to build and install *HPG* built with serial,
OpenMP and CUDA devices, together with all its dependencies (including
*CMake*, *CUDA*, *FFTW*, *Kokkos*, plus all transitive dependencies),
starting from a fresh *Spack* installation configured with *gcc*
version 9.3.0

``` shell
$ spack install \
    hpg@main%gcc@9.3.0+openmp+serial+cuda api=11 \
        ^fftw~mpi \
        ^kokkos cuda_arch=70 +wrapper \
        ^kokkos-nvcc-wrapper~mpi
```
Users are encouraged to find out more at the [Spack web site](https://spack.io).

#### Tests

There's some weirdness about Spack unit testing and HDF5 that I can't
track, exactly, but there is a workaround. HDF5 is needed by an HPG
unit test because that test uses an HDF5 file containing a reference grid for
comparison. The workaround is to build *and load* `hdf5 +cxx ~mpi`
using Spack before attempting to build HPG. At least this works in a
Spack environment, which then looks something like the following

``` shell
$ spack install hdf5 +cxx ~mpi
$ spack load hdf5
$ spack env create hpg
$ spackactivate hpg
$ spack develop hpg@main -p [PATH_TO_MY_HPG_REPO] # optional, but it's what I do
$ spack add hpg
$ spack concretize # optional, but it's worth checking
$ spack install --test root
```

## Using libhpg

The *HPG* installation includes a *CMake* package configuration file
that defines the `hpg::hpg` target. To build *CMake* targets against
the *libhpg* library, simply include something like the following
statements in `CMakeLists.txt`:

``` cmake
# find the package
find_package(hpg CONFIG)

# link to executable "MyProgram"
add_executable(MyProgram)
target_link_libraries(MyProgram hpg::hpg)

# link to library "MyLibrary" (HPG API not exposed by MyLibrary)
add_library(MyLibrary)
target_link_libraries(MyLibrary PRIVATE hpg::hpg)
```

Most client code should include only the header `hpg.hpp` (using
`#include <hpg/hpg.hpp>`), which hides from the client code all use
by the *HPG* implementation of *Kokkos* types and concepts. However,
if direct access to the *Kokkos* kernels is desired, the header
`hpg_impl.h` may be used.
