// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once

#include "hpg_config.hpp"
#include "hpg_version.hpp"
#include "hpg_rval.hpp"
#include "hpg_error.hpp"

#include <cassert>
#include <complex>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#if HPG_API >= 17 || defined(HPG_INTERNAL)
# include <optional>
# include <variant>
#endif

#include "hpg_export.h"

/** @file hpg.hpp
 *
 * Main header file for top-level HPG API
 */

/** top-level HPG API namespace
 */
namespace hpg {

/** initialization arguments
 *
 * This type is currently identical to the Kokkos type of the same name,
 * although that could change in future versions.
 *
 * If you set num_threads or num_numa to zero or less, Kokkos will try to
 * determine default values if possible or otherwise set them to 1. In
 * particular, Kokkos can use the hwloc library to determine default settings
 * using the assumption that the process binding mask is unique, i.e., that this
 * process does not share any cores with another process. Note that the default
 * value of each parameter is -1.
 */
struct InitArguments {

  InitArguments()
    : num_threads(-1)
    , num_numa(-1)
    , device_id(-1)
    , ndevices(-1)
    , skip_device(01)
    , disable_warnings(false) {}

  /** number of threads per NUMA region
   *
   * Used in conjunction with num_numa
   */
  int num_threads;

  /** number of NUMA regions used by process
   */
  int num_numa;

  /** device id to be used */
  int device_id;

  /** number of devices to be used per node
   *
   * Used when running MPI jobs. Process to device mapping happens by obtaining
   * the local MPI rank and assigning devices round-robin.
   */
  int ndevices;

  /** device to ignore
   *
   * Used in conjunction with ndevices.  This is most useful
   * on workstations with multiple GPUs, one of which is used to drive screen
   * output.
   */
  int skip_device;

  /** disable Kokkos warnings
   */
  bool disable_warnings;
};

/** global initialization of hpg
 *
 * Function is idempotent, but should not be called by a process after a call to
 * hpg::finalize(). All objects created by hpg must only exist between an
 * initialize()/finalize() pair; in particular, any hpg object destructors must
 * be called before the call to finalize(). A common approach is to access hpg
 * within a new scope after the call to initialize():
 * @code{.cpp}
 *     int main() {
 *       hpg::initialize();
 *       {
 *         Gridder g(...);
 *       }
 *       hpg::finalize();
 *     }
 * @endcode
 * Another approach involves the use of a ScopeGuard instance.
 *
 * @return true, if and only if initialization succeeded
 */
HPG_EXPORT bool initialize();

/** global initialization of hpg, with arguments
 *
 * @param args initialization parameters
 */
HPG_EXPORT bool initialize(const InitArguments& args);

/** global finalization of hpg
 *
 * Function is idempotent, but should only be called by a process after a call
 * to hpg::initialize()
 */
HPG_EXPORT void finalize();

/** query whether hpg has been initialized
 *
 * Note the result will remain "true" after finalization.
 */
HPG_EXPORT bool is_initialized() noexcept;

/** backend device type
 */
enum class HPG_EXPORT Device {
  Serial, /**< serial device */
  OpenMP, /**< OpenMP device */
  Cuda, /**< CUDA device */
};

/** supported devices */
HPG_EXPORT const std::set<Device>&
devices() noexcept;

/** supported host devices */
HPG_EXPORT const std::set<Device>&
host_devices() noexcept;

/** identification string for unspecified CF layout */
extern const char * const cf_layout_unspecified_version;

/** hpg scope object
 *
 * Intended to help avoid errors caused by objects that exist after the call
 * to hpg::finalize(). For example,
 * @code{.cpp}
 *     int main() {
 *       // Don't do this!
 *       hpg::initialize();
 *       Gridder g();
 *       hpg::finalize(); // Error! g is still in scope,
 *                        // ~Gridder is called after finalize()
 *     }
 * @endcode
 * however, use of a ScopeGuard value as follows helps avoid this error:
 * @code{.cpp}
 *     int main() {
 *       hpg::ScopeGuard hpg_guard;
 *       Gridder g();
 *       // OK, because g is destroyed prior to hpg_guard
 *     }
 * @endcode
 */
struct HPG_EXPORT ScopeGuard {

private:
  bool init;

public:
  /** default constructor
   *
   * calls hpg::initialize()
   */
  ScopeGuard();

  /** constructor with initialization argument
   *
   * calls hpg::initialize(const InitArguments&)
   */
  ScopeGuard(const InitArguments& args);

  ~ScopeGuard();

  ScopeGuard(const ScopeGuard&) = delete;

  ScopeGuard&
  operator=(const ScopeGuard&) = delete;
};

/** convolution function array value floating point type */
using cf_fp = float;
/** visibility value floating point type */
using visibility_fp = float;
/** gridded value floating point type */
using grid_value_fp = double;
/** grid scale floating point type */
using grid_scale_fp = float;
/** visibility (U, V, W) coordinate floating point type */
using vis_uvw_fp = double;
/** visibility weight floating point type */
using vis_weight_fp = float;
/** visibility frequency floating point type */
using vis_frequency_fp = float;
/** visibility phase floating point type */
using vis_phase_fp = double;
/** CF phase gradient floating point type */
using cf_phase_gradient_fp = float;

// vis_uvw_t can be any type that supports std::get<N>() for element access
/** UVW coordinate type */
using vis_uvw_t = std::array<vis_uvw_fp, 3>;

// cf_phase_gradient_t can be any type that supports std::get<N>() for element
// access
/** CF phase gradient type */
using cf_phase_gradient_t = std::array<cf_phase_gradient_fp, 2>;

/** type to represent an optional value */
#if HPG_API >= 17
template <typename T>
using opt_t = std::optional<T>;
#else // HPG_API < 17
template <typename T>
using opt_t = std::shared_ptr<T>;
#endif //HPG_API >= 17

/** type to represent a possible error */
using opt_error_t = opt_t<Error>;

/** representation of visibility data
 *
 * @tparam N number of polarizations
 */
template <unsigned N>
struct VisData {

  /** number of polarizations */
  static constexpr unsigned npol = N;

  /** visibility values, ordered by polarization*/
  std::array<std::complex<visibility_fp>, N> m_visibilities;
  /** visibility weights, ordered by polarization*/
  std::array<vis_weight_fp, N> m_weights;
  /** visibility frequency */
  vis_frequency_fp m_frequency;
  /** visibility phase */
  vis_phase_fp m_phase;
  /** visibility UVW coordinates */
  vis_uvw_t m_uvw;
  /** grid cube index */
  unsigned m_grid_cube;
  /** cube and grp CFArray index components */
  std::array<unsigned, 2> m_cf_index;
  /** phase gradient */
  cf_phase_gradient_t m_cf_phase_gradient;

  /** constructor, with CF phase gradient values */
  VisData(
    const std::array<std::complex<visibility_fp>, N>& visibilities,
    const std::array<vis_weight_fp, N>& weights,
    const vis_frequency_fp& frequency,
    const vis_phase_fp& phase,
    const vis_uvw_t& uvw,
    const unsigned& grid_cube,
    const std::array<unsigned, 2>& cf_index,
    const cf_phase_gradient_t& cf_phase_gradient)
    : m_visibilities(visibilities)
    , m_weights(weights)
    , m_frequency(frequency)
    , m_phase(phase)
    , m_uvw(uvw)
    , m_grid_cube(grid_cube)
    , m_cf_index(cf_index)
    , m_cf_phase_gradient(cf_phase_gradient) {}

  /** constructor, without CF phase gradient values */
  VisData(
    const std::array<std::complex<visibility_fp>, N>& visibilities,
    const std::array<vis_weight_fp, N>& weights,
    const vis_frequency_fp& frequency,
    const vis_phase_fp& phase,
    const vis_uvw_t& uvw,
    const unsigned& grid_cube,
    const std::array<unsigned, 2>& cf_index)
    : m_visibilities(visibilities)
    , m_weights(weights)
    , m_frequency(frequency)
    , m_phase(phase)
    , m_uvw(uvw)
    , m_grid_cube(grid_cube)
    , m_cf_index(cf_index)
    , m_cf_phase_gradient({0, 0}) {}

  /** default constructor */
  VisData() {}

  /** equality operator */
  bool
  operator==(const VisData& rhs) {
    return m_visibilities == rhs.m_visibilities
      && m_weights == rhs.m_weights
      && m_frequency == rhs.m_frequency
      && m_phase == rhs.m_phase
      && m_uvw == rhs.m_uvw
      && m_grid_cube == rhs.m_grid_cube
      && m_cf_index == rhs.m_cf_index
      && m_cf_phase_gradient == rhs.m_cf_phase_gradient;
  }
};

namespace Impl {
struct HPG_EXPORT State;
struct HPG_EXPORT GridderState;

/** sign of a value
 *
 * @return -1, if less than 0; +1, if greater than 0; 0, if equal to 0
 */
template <typename T>
inline constexpr int
sgn(T val) {
  return (T(0) < val) - (val < T(0));
}
} // end namespace Impl

/** vector of elements parameterized by number of polarizations
 *
 * Erases "number of polarizations" template parameter
 */
template <template <unsigned> typename E>
struct VectorNPol {

  /** number of polarizations in elements of contained vector */
  unsigned m_npol;

  /** contained vector */
  union {
    std::unique_ptr<std::vector<E<1>>> m_v1;
    std::unique_ptr<std::vector<E<2>>> m_v2;
    std::unique_ptr<std::vector<E<3>>> m_v3;
    std::unique_ptr<std::vector<E<4>>> m_v4;
  };

  /** default constructor */
  VectorNPol()
    : m_npol(0)
    , m_v1() {}

  /** construct instance by moving vector elements with N=1 */
  VectorNPol(std::vector<E<1>>&& v)
    : m_npol(1)
    , m_v1(new std::vector<E<1>>(std::move(v))) {}

  /** construct instance by copying vector elements with N=1 */
  VectorNPol(const std::vector<E<1>>& v)
    : m_npol(1)
    , m_v1(new std::vector<E<1>>(v)) {}

  /** construct instance by moving vector elements with N=2 */
  VectorNPol(std::vector<E<2>>&& v)
    : m_npol(2)
    , m_v2(new std::vector<E<2>>(std::move(v))) {}

  /** construct instance by copying vector elements with N=2 */
  VectorNPol(const std::vector<E<2>>& v)
    : m_npol(2)
    , m_v2(new std::vector<E<2>>(v)) {}

  /** construct instance by moving vector elements with N=3 */
  VectorNPol(std::vector<E<3>>&& v)
    : m_npol(3)
    , m_v3(new std::vector<E<3>>(std::move(v))) {}

  /** construct instance by copying vector elements with N=3 */
  VectorNPol(const std::vector<E<3>>& v)
    : m_npol(3)
    , m_v3(new std::vector<E<3>>(v)) {}

  /** construct instance by moving vector elements with N=4 */
  VectorNPol(std::vector<E<4>>&& v)
    : m_npol(4)
    , m_v4(new std::vector<E<4>>(std::move(v))) {}

  /** construct instance by copying vector elements with N=4 */
  VectorNPol(const std::vector<E<4>>& v)
    : m_npol(4)
    , m_v4(new std::vector<E<4>>(v)) {}

  /** copy constructor */
  VectorNPol(const VectorNPol& other)
    : m_npol(other.m_npol)
    , m_v1() {
    switch (m_npol) {
    case 0:
      break;
    case 1:
      m_v1 =
        std::unique_ptr<std::vector<E<1>>>(new std::vector<E<1>>(*other.m_v1));
      break;
    case 2:
      m_v2 =
        std::unique_ptr<std::vector<E<2>>>(new std::vector<E<2>>(*other.m_v2));
      break;
    case 3:
      m_v3 =
        std::unique_ptr<std::vector<E<3>>>(new std::vector<E<3>>(*other.m_v3));
      break;
    case 4:
      m_v4 =
        std::unique_ptr<std::vector<E<4>>>(new std::vector<E<4>>(*other.m_v4));
      break;
    default:
      assert(false);
      break;
    }
  }

  /** move constructor */
  VectorNPol(VectorNPol&& other)
    : m_npol(other.m_npol)
    , m_v1() {
    switch (m_npol) {
    case 0:
      break;
    case 1:
      m_v1 = std::move(other).m_v1;
      break;
    case 2:
      m_v2 = std::move(other).m_v2;
      break;
    case 3:
      m_v3 = std::move(other).m_v3;
      break;
    case 4:
      m_v4 = std::move(other).m_v4;
      break;
    default:
      assert(false);
      break;
    }
  }

  /** copy assignment operator */
  VectorNPol&
  operator=(const VectorNPol& rhs) {
    VectorNPol tmp(rhs);
    swap(tmp);
    return *this;
  }

  /** move assignment operator */
  VectorNPol&
  operator=(VectorNPol&& rhs) {
    VectorNPol tmp(std::move(rhs));
    swap(tmp);
    return *this;
  }

  /** number of elements of vector */
  size_t
  size() const {
    switch (m_npol) {
    case 0:
      return 0;
      break;
    case 1:
      return m_v1->size();
      break;
    case 2:
      return m_v2->size();
      break;
    case 3:
      return m_v3->size();
      break;
    case 4:
      return m_v4->size();
      break;
    default:
      assert(false);
      return 0;
      break;
    }
  }

  /** total number of stored values
   *
   * number of polarization multiplied by size of vector
   */
  size_t
  num_elements() const {
    switch (m_npol) {
    case 0:
      return 0;
      break;
    case 1:
      return size();
      break;
    case 2:
      return 2 * size();
      break;
    case 3:
      return 3 * size();
      break;
    case 4:
      return 4 * size();
      break;
    default:
      assert(false);
      return 0;
      break;
    }
  }

  virtual ~VectorNPol() {
    switch (m_npol) {
    case 0:
    case 1:
      m_v1.reset();
      break;
    case 2:
      m_v2.reset();
      break;
    case 3:
      m_v3.reset();
      break;
    case 4:
      m_v4.reset();
      break;
    default:
      assert(false);
      break;
    }
  }

private:

  /** take over the vector from another instance */
  void
  takev(VectorNPol& other) {
    switch (other.m_npol) {
    case 0:
    case 1:
      m_v1 = std::move(other.m_v1);
      break;
    case 2:
      m_v2 = std::move(other.m_v2);
      break;
    case 3:
      m_v3 = std::move(other.m_v3);
      break;
    case 4:
      m_v4 = std::move(other.m_v4);
      break;
    default:
      assert(false);
      break;
    }
    m_npol = other.m_npol;
    other.m_npol = 0;
  }

  /** swap contents with another instance */
  void
  swap(VectorNPol& other) {
    switch (other.m_npol) {
    case 0: {
      auto ov1 = std::move(other).m_v1;
      other.takev(*this);
      m_v1 = std::move(ov1);
      m_npol = 0;
      break;
    }
    case 1: {
      auto ov1 = std::move(other).m_v1;
      other.takev(*this);
      m_v1 = std::move(ov1);
      m_npol = 1;
      break;
    }
    case 2: {
      auto ov2 = std::move(other).m_v2;
      other.takev(*this);
      m_v2 = std::move(ov2);
      m_npol = 2;
      break;
    }
    case 3: {
      auto ov3 = std::move(other).m_v3;
      other.takev(*this);
      m_v3 = std::move(ov3);
      m_npol = 3;
      break;
    }
    case 4: {
      auto ov4 = std::move(other).m_v4;
      other.takev(*this);
      m_v4 = std::move(ov4);
      m_npol = 4;
      break;
    }
    default:
      assert(false);
      break;
    }
  }
};

/** vector of VisData<.> elements */
using VisDataVector = VectorNPol<VisData>;

/** helper type for definition of IArrayVector */
template <unsigned N>
using iarray = std::array<int, N>;

/** vector of std::array<int, .> elements */
using IArrayVector = VectorNPol<iarray>;

/** array layout enumeration */
enum class HPG_EXPORT Layout {
  Right, /**< C order: rightmost index has smallest stride */
  Left, /**< FORTRAN order: leftmost index has smallest stride */
};

/** shape of a convolution function
 *
 * This class is primarily of use as an argument to describe the maximum
 * expected size of a CFArray. Note that CFArray is a subclass of this class.
 *
 * @sa CFArray
 */
class HPG_EXPORT CFArrayShape {
public:

  /** rank of array */
  static constexpr unsigned rank = 5;

  /** oversampling factor */
  virtual unsigned
  oversampling() const = 0;

  /** number of CF groups */
  virtual unsigned
  num_groups() const = 0;

  /** array extents for a given group */
  virtual std::array<unsigned, rank - 1>
  extents(unsigned grp) const = 0;

  /** destructor */
  virtual ~CFArrayShape() {}
};

/** base class for convolution functions */
class HPG_EXPORT CFArray
  : public CFArrayShape {
public:

  /** element value type */
  using value_type = std::complex<cf_fp>;

  /** padding, in units of major increments (not oversampled), on every edge of
   * CF support domain */
  static constexpr unsigned padding = 2;

  /** ordered index axis names */
  // note: changes in this must be coordinated with changes in the element
  // access operators
  enum Axis {x, y, mueller, cube, group};

  /** CF element layout identification */
  virtual const char*
  layout() const {
    return cf_layout_unspecified_version;
  }

  /** element access operator
   *
   * @param x X coordinate, relative to padded domain edge (oversampled units)
   * @param y Y coordinate, relative to padded domain edge (oversampled units)
   * @param mueller Mueller element index; selects an element of a Mueller
   * matrix
   * @param cube cube index
   * @param group group index
   */
  virtual std::complex<cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned cube,
    unsigned group)
    const = 0;

  /** half-widths of CF support domain for a given group index */
  std::array<unsigned, 2>
  radii(unsigned grp) const {
    auto os = oversampling();
    auto ext = extents(grp);
    return {
      ((ext[0] - 2 * padding * os) / os) / 2,
      ((ext[1] - 2 * padding * os) / os) / 2};
  }

  /** destructor */
  virtual ~CFArray() {}

  /** copy values into a buffer with optimal layout for a given device
   *
   * @param device target device
   * @param host_device host device to use for converting layout
   * @param grp group index
   * @param dst buffer into which to copy values
   *
   * @return layout version string or error
   */
  rval_t<std::string>
  copy_to(Device device, Device host_device, unsigned grp, value_type* dst)
    const;

  /** minimum size of buffer for destination of copy_to()
   *
   * @param device target device
   * @param grp group index
   *
   * @return number of elements required in a destination buffer or error
   */
  rval_t<size_t>
  min_buffer_size(Device device, unsigned grp) const;
};

/** CFArray sub-class for stored (cached) values in optimized layout for
 * devices
 */
class HPG_EXPORT DeviceCFArray
  : public CFArray {
public:

  /** create a DeviceCFArray (sub-class) instance
   *
   * @param version version string
   * @param oversampling CF oversampling factor
   * @param arrays extent and values in given layout for every CFArray group
   */
  static rval_t<std::unique_ptr<DeviceCFArray>>
  create(
    const std::string& version,
    unsigned oversampling,
    std::vector<
      std::tuple<
        std::array<unsigned, rank - 1>,
        std::vector<value_type>>>&&
      arrays);

  /** associated device type */
  virtual Device
  device() const = 0;

  virtual ~DeviceCFArray() {}
};

/** wrapper for access to copy of grid values
 *
 * @todo: replace with mdspan?
 */
class HPG_EXPORT GridValueArray {
public:

  /** rank of array */
  static constexpr unsigned rank = 4;

  /** element value type */
  using value_type = std::complex<grid_value_fp>;

  /** ordered index axis names */
  // note: changes in this must be coordinated with changes in the element
  // access operators
  enum Axis {x, y, mrow, cube};

  /** size of array on given dimension */
  virtual unsigned
  extent(unsigned dim) const = 0;

  /** const element access operator */
  virtual const value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube) const = 0;

  /** non-const element access operator */
  virtual value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube) = 0;

  /** copy values to a buffer in requested layout
   *
   * @param host_device host device to use for copying values
   * @param dst destination buffer
   * @param layout array layout of values copied into dst
   *
   * @return an Error, iff host_device names a disabled host device
   */
  opt_t<Error>
  copy_to(
    Device host_device,
    value_type* dst,
    Layout layout = Layout::Left) const;

  /** minimum size of buffer for destination of copy_to()
   *
   * @return number of elements required in a destination buffer
   */
  virtual size_t
  min_buffer_size() const {
    return extent(0) * extent(1) * extent(2) * extent(3);
  }

  /** destructor */
  virtual ~GridValueArray() {}

  /** create GridValueArray instance from values in a buffer
   *
   * @param name name of Kokkos::View underlying return value (implementation
   * detail)
   * @param target_device device for which result value is intended to be used
   * @param host_device host device to use for copying values
   * @param src source buffer
   * @param extents grid size
   * @param layout array layout of values copied from src
   */
  static std::unique_ptr<GridValueArray>
  copy_from(
    const std::string& name,
    Device target_device,
    Device host_device,
    value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout layout = Layout::Left);

protected:

  /** unsafe version of copy_to()
   *
   * Assumes host_device names an enabled host device
   */
  virtual void
  unsafe_copy_to(Device host_device, value_type* dst, Layout layout) const = 0;
};

/** wrapper for access to copy of grid weights
 *
 * @todo: replace with mdspan?
 */
class HPG_EXPORT GridWeightArray {
public:

  /** rank of array */
  static constexpr unsigned rank = 2;

  /** element value type */
  using value_type = grid_value_fp;

  /** ordered index axis names */
  // note: changes in this must be coordinated with changes in the element
  // access operators
  enum Axis {mrow, cube};

  /** size of array on given dimension */
  virtual unsigned
  extent(unsigned dim) const = 0;

  /** const element access operator */
  virtual const value_type&
  operator()(unsigned mrow, unsigned cube) const = 0;

  /** non-const element access operator */
  virtual value_type&
  operator()(unsigned mrow, unsigned cube) = 0;

  /** copy values to a buffer in requested layout
   *
   * @param host_device host device to use for copying values
   * @param dst destination buffer
   * @param layout array layout of values copied into dst
   *
   * @return an Error, iff host_device names a disabled host device
   */
  opt_t<Error>
  copy_to(
    Device host_device,
    value_type* dst,
    Layout layout = Layout::Left) const;

  /** minimum size of buffer for destination of copy_to()
   *
   * @return number of elements required in a destination buffer
   */
  virtual size_t
  min_buffer_size() const {
    return extent(0) * extent(1);
  }

  /** destructor */
  virtual ~GridWeightArray() {}

  /** create GridWeightArray instance from values in a buffer
   *
   * @param name name of Kokkos::View underlying return value (implementation
   * detail)
   * @param target_device device for which result value is intended to be used
   * @param host_device host device to use for copying values
   * @param src source buffer
   * @param extents grid size
   * @param layout array layout of values copied from src
   */
  static std::unique_ptr<GridWeightArray>
  copy_from(
    const std::string& name,
    Device target_device,
    Device host_device,
    value_type* src,
    const std::array<unsigned, rank>& extents,
    Layout layout = Layout::Left);

protected:

  /** unsafe version of copy_to()
   *
   * Assumes host_device names an enabled host device
   */
  virtual void
  unsafe_copy_to(Device host_device, value_type* dst, Layout layout) const = 0;
};

class HPG_EXPORT Gridder;

/** sign of imaginary unit in exponent of FFT kernel */
enum class HPG_EXPORT FFTSign {
  POSITIVE, /**< +1 */
  NEGATIVE  /**< -1 */
};

/** default value of sign of imaginary unit in exponent of FFT kernel for
 * apply_grid_fft() */
constexpr FFTSign grid_fft_sign_dflt = FFTSign::POSITIVE;

/** default value of sign of imaginary unit in exponent of FFT kernel for
 * apply_model_fft() */
constexpr FFTSign model_fft_sign_dflt =
  ((grid_fft_sign_dflt == FFTSign::POSITIVE)
   ? FFTSign::NEGATIVE
   : FFTSign::POSITIVE);

/** future
 *
 * A type similar to a std::future, but with limitations to account for the fact
 * that GridderState progress to fulfill a future can only occur during calls to
 * GridderState methods. In this situation a regular std::future is prone to
 * deadlocks, which is the reason for the existence of this class.
 *
 * Note that, by design, these futures are never resolved by an exception, and
 * it is a requirement that the underlying std::future behaves similarly.
 */
template <typename T>
class HPG_EXPORT future final {
public:

  /** default constructor */
  future() {}

  /** constructor */
  future(const std::function<opt_t<T>&()>& f)
    : m_f(f) {}

  future(future&& f)
    : m_f(std::move(f).m_f) {}

  future(const future& f)
    : m_f(f.m_f) {}

  future&
  operator=(const future& f) {
    m_f = f.m_f;
    return *this;
  }

  future&
  operator=(future&& f) {
    m_f = std::move(f).m_f;
    return *this;
  }

  /** get value
   *
   * @return value of the future if it has been resolved, or nothing
   */
  opt_t<T>&
  get() noexcept {
    return m_f();
  }

  template <typename U>
  future<U>
  map(const std::function<U(T&&)>& f) && {
    return future<U>(
      [f, mf=std::move(m_f), result=opt_t<U>()]() mutable -> opt_t<U>& {
        if (!result) {
          auto& ot = mf();
#if HPG_API >= 17
          if (ot)
            result = f(std::move(ot).value());
#else
          if (ot)
            result = opt_t<U>(new U(f(std::move(*ot))));
#endif
        }
        return result;
      });
  }

  virtual ~future() {}

protected:

  std::function<opt_t<T>&()> m_f; /**< contained std::function */
};

/** gridder state
 *
 * A container for the entire state needed to do gridding as a value, including
 * gridded visibility data and possibly a convolution function array. Used by
 * the Gridder class, but may also be used directly for its greater flexibility.
 *
 * Depending on the device used for gridding, methods may schedule tasks for
 * asynchronous execution. When an instance is constructed, the user may provide
 * a maximum number of concurrent, asynchronous tasks to run on the requested
 * device. Note that the actual number of concurrent tasks that the new instance
 * supports may be less than the number requested in the constructor. In
 * particular, several devices support no asynchronous execution. Class methods
 * that submit work to a device, that is, for either data movement or
 * computation, may block when the number of asynchronously running tasks is at
 * its limit, until one of those running tasks completes. Otherwise, the methods
 * that create device tasks will return as soon the task has been submitted to
 * the device.
 *
 * In general, using a GridderState method on an instance creates a new (value)
 * copy of the target. For example,
 * @code{.cpp}
 *    GridderState s0;
 *    GridderState s1 = s0.fence();
 * @endcode
 * will create a copy of s0. Note that a copy will include the grid and the
 * convolution function array. To avoid the copy, the following pattern can be
 * used instead:
 * @code{.cpp}
 *    GridderState s0;
 *    GridderState s1 = std::move(s0).fence();
 * @endcode
 * Note, however, that the value of s0 in the previous example after the call to
 * fence() will be in the null state, which is likely not of much further use to
 * the caller.
 */
class HPG_EXPORT GridderState {
protected:
  friend class Gridder;
  friend class Impl::GridderState;

  // state cannot be a unique_ptr since Impl::State is here an incomplete type
  std::shared_ptr<Impl::State> impl; /**< state implementation */

public:
  /** default constructor
   *
   * the null state, most methods will fail when called on a target with this
   * value
   */
  GridderState();

protected:

  /** constructor
   *
   * create a GridderState
   *
   * @param device gridder device type
   * @param max_added_tasks maximum number of additional tasks (actual number
   * may be less than requested)
   * @param max_visibility_batch_size maximum number of VisData<.> values for
   * calls to grid_visibilities()
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size in logical axis order: X, Y, mrow, cube
   * @param grid_scale in X, Y order
   * @param mueller_indexes CFArray Mueller element indexes, by mrow
   * @param conjugate_mueller_indexes CFArray conjugate Mueller element indexes,
   * by mrow
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is
   * employed, but some devices support additional, concurrent tasks.
   *
   * The value of max_added_tasks and max_visibility_batch_size has an effect on
   * the amount of memory allocated on the selected gridder device. The total
   * amount of memory allocated for visibilities will be approximately equal to
   * max_added_tasks multiplied by sizeof(VisData<N>) for the appropriate value
   * of N.
   *
   * @sa Gridder::Gridder()
   */
  GridderState(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions = {0, 0, 0, 0}
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    );

public:

  /** GridderState factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @sa GridderState()
   */
  static rval_t<GridderState>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    IArrayVector&& mueller_indexes,
    IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept;

  /** GridderState factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @tparam N number of polarizations in visibilities to be gridded
   *
   * @sa GridderState()
   */
  template <unsigned N>
  static rval_t<GridderState>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions = {0, 0, 0, 0}
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create(
        device,
        max_added_tasks,
        max_visibility_batch_size,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        , implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        );
  }

  /** copy constructor
   *
   * Copies all state. Invokes fence() on argument.
   */
  GridderState(const GridderState&);

  /** move constructor
   */
  GridderState(GridderState&&);

  virtual ~GridderState();

  /** copy assignment
   *
   * Copies all state. Invokes fence() on argument.
   */
  GridderState&
  operator=(const GridderState&);

  /** move assignment
   */
  GridderState&
  operator=(GridderState&&);

  /** device */
  Device
  device() const noexcept;

  /** maximum additional tasks
   *
   * This value may differ from the value provided to the constructor, depending
   * on device limitations */
  unsigned
  max_added_tasks() const noexcept;

  /** maximum number of visibilities passed to gridding kernel at once */
  size_t
  max_visibility_batch_size() const noexcept;

  /** grid size */
  const std::array<unsigned, 4>&
  grid_size() const noexcept;

  /** grid scale */
  std::array<grid_scale_fp, 2>
  grid_scale() const noexcept;

  /** number of visibility polarizations */
  unsigned
  num_polarizations() const noexcept;

  /** null state query */
  bool
  is_null() const noexcept;

  /** size (in bytes) of region allocated for CFArray elements
   *
   * Memory allocations for convolution function regions are made per device
   * task. Values returned for the size of the currently allocated region refer
   * only to the most recently allocated region; multiplying this value by the
   * number of device tasks may be an inaccurate measure of the total allocated
   * region, as asynchronously executing tasks may be using different
   * convolution functions.
   *
   * @param shape if non-null, the memory needed for a CFArray of the given
   * shape; if null, the size of the currently allocated region in the
   * target
   */
  size_t
  convolution_function_region_size(const CFArrayShape* shape) const noexcept;

  /** allocate memory for convolution function
   *
   * Increasing memory allocations for convolution functions are handled
   * automatically by set_convolution_function(), but in a sequence of calls to
   * set_convolution_function() in which later calls require a larger allocation
   * than earlier calls, it may be advantageous to use this method in order to
   * allocate the maximum memory that will be required by the sequence before
   * starting the sequence, which will then permit the sequence to proceed
   * without any reallocations. To release all memory allocated for the
   * convolution function, the caller may pass a null pointer for the method
   * argument. Invokes fence() on the target.
   *
   * @param shape shape of CFArray for which to allocate memory (per task)
   *
   * @return new GridderState that is a copy of the target, but with memory
   * allocated for convolution function, or error
   */
  rval_t<GridderState>
  allocate_convolution_function_region(const CFArrayShape* shape) const &;

  /** allocate memory for convolution function
   *
   * Increasing memory allocations for convolution functions are handled
   * automatically by set_convolution_function(), but in a sequence of calls to
   * set_convolution_function() in which later calls require a larger allocation
   * than earlier calls, it may be advantageous to use this method in order to
   * allocate the maximum memory that will be required by the sequence before
   * starting the sequence, which will then permit the sequence to proceed
   * without any reallocations. To release all memory allocated for the
   * convolution function, the caller may pass a null pointer for the method
   * argument. Invokes fence() on the target.
   *
   * @param shape shape of CFArray for which to allocate memory (per task)
   *
   * @return new GridderState that has overwritten the target, but with memory
   * allocated for convolution function, or error
   */
  rval_t<GridderState>
  allocate_convolution_function_region(const CFArrayShape* shape) &&;

  /** set convolution function
   *
   * May invoke fence() on target.
   *
   * @return new GridderState that is a copy of the target, but with provided
   * convolution function for subsequent gridding
   *
   * @param host_device device to use for changing array layout
   * @param cf convolution function array
   *
   * @sa Gridder::set_convolution_function()
   */
  rval_t<GridderState>
  set_convolution_function(Device host_device, CFArray&& cf) const &;

  /** set convolution function
   *
   * May invoke fence() on target.
   *
   * @return new GridderState that has overwritten the target, but with provided
   * convolution function for subsequent gridding
   *
   * @param host_device device to use for changing array layout
   * @param cf convolution function array
   *
   * @sa Gridder::set_convolution_function()
   */
  rval_t<GridderState>
  set_convolution_function(Device host_device, CFArray&& cf) &&;

  /** set visibility model
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after setting model
   *
   * @param host_device device to use for copying model values
   * @param gv visibility model
   *
   * @sa Gridder::set_model()
   */
  rval_t<GridderState>
  set_model(Device host_device, GridValueArray&& gv) const &;

  /** set visibility model
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after setting model
   *
   * @param host_device device to use for copying model values
   * @param gv visibility model
   *
   * @sa Gridder::set_model()
   */
  rval_t<GridderState>
  set_model(Device host_device, GridValueArray&& gv) &&;

protected:

  friend class Gridder;

  /** narrow type of future containing VisDataVector to future containing
   * std::vector<VisData<N>>
   *
   * @tparam N number of polarizations in visibilities
   */
  template <unsigned N>
  static future<std::vector<VisData<N>>>
  future_visibilities_narrow(future<VisDataVector>&& fvs) {
    std::abort();
  }

public:

  /** degridding/gridding base method (const version)
   *
   * May invoke fence() on target. This method is the most general of all the
   * "grid_visibilities() const &" methods, and the one that is called by the
   * implementations of all other variants.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of a potentially empty vector of returned (residual or
   * predicted) visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param do_degrid do degridding
   * @param return_visibilities return residual or predicted visibilities
   * @param do_grid do gridding
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  grid_visibilities_base(
    Device host_device,
    VisDataVector&& visibilities,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) const &;

  /** degridding/gridding base method (rvalue reference version)
   *
   * May invoke fence() on target. This method is the most general of all the
   * "grid_visibilities() &&" methods, and the one that is called by the
   * implementations of all other variants.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of a potentially empty vector of returned (residual or
   * predicted) visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param do_degrid do degridding
   * @param return_visibilities return residual or predicted visibilities
   * @param do_grid do gridding
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  grid_visibilities_base(
    Device host_device,
    VisDataVector&& visibilities,
    bool do_grid,
    bool return_visibilities,
    bool do_degrid) &&;

  /** grid visibilities, without degridding (template-free, const version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities) const &;

  /** grid visibilities, without degridding (template-free, rvalue reference
   * version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities) &&;

  /** grid visibilities, without degridding (templated, const version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) const & {

    return
      grid_visibilities(host_device, VisDataVector(std::move(visibilities)));
  };

  /** grid visibilities, without degridding (templated, rvalue reference
   * version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) && {

    return
      std::move(*this)
      .grid_visibilities(host_device, VisDataVector(std::move(visibilities)));
  };

  /** degrid and grid visibilities (template-free, const version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<GridderState>
  degrid_grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities) const &;

  /** degrid and grid visibilities (template-free, rvalue reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<GridderState>
  degrid_grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities) &&;

  /** degrid and grid visibilities (templated, const version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<GridderState>
  degrid_grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) const & {

    return
      degrid_grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
  };

  /** degrid and grid visibilities (templated, rvalue reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<GridderState>
  degrid_grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) && {

    return
      std::move(*this).degrid_grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
  };

  /** degrid visibilities, returning predicted visibilities (template-free,
   * const version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of predicted visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  degrid_get_predicted_visibilities(
    Device host_device,
    VisDataVector&& visibilities) const &;

  /** degrid visibilities, returning predicted visibilities (template-free,
   * rvalue reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of predicted visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  degrid_get_predicted_visibilities(
    Device host_device,
    VisDataVector&& visibilities) &&;

  /** degrid visibilities, returning predicted visibilities (templated, const
   * version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of predicted visibilities
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<std::tuple<GridderState, future<std::vector<VisData<N>>>>>
  degrid_get_predicted_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) const & {

    auto tpl_or_err =
      degrid_get_predicted_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
    if (hpg::is_value(tpl_or_err)) {
      GridderState gs;
      future<VisDataVector> fvs;
      std::tie(gs, fvs) = hpg::get_value(std::move(tpl_or_err));
      return
        std::make_tuple(
          std::move(gs),
          future_visibilities_narrow<N>(std::move(fvs)));
    } else {
      return hpg::get_error(std::move(tpl_or_err));
    }
  };

  /** degrid visibilities, returning predicted visibilities (templated, rvalue
   * reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of predicted visibilities
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<std::tuple<GridderState, future<std::vector<VisData<N>>>>>
  degrid_get_predicted_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) && {

    auto tpl_or_err =
      std::move(*this).degrid_get_predicted_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
    if (hpg::is_value(tpl_or_err)) {
      GridderState gs;
      future<VisDataVector> fvs;
      std::tie(gs, fvs) = hpg::get_value(std::move(tpl_or_err));
      return
        std::make_tuple(
          std::move(gs),
          future_visibilities_narrow<N>(std::move(fvs)));
    } else {
      return hpg::get_error(std::move(tpl_or_err));
    }
  };

  /** degrid and grid visibilities, returning residual visibilities
   * (template-free, const version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of residual visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    VisDataVector&& visibilities) const &;

  /** degrid and grid visibilities, returning residual visibilities
   * (template-free, rvalue reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of residual visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    VisDataVector&& visibilities) &&;

  /** degrid and grid visibilities, returning residual visibilities (templated,
   * const version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of residual visibilities
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<std::tuple<GridderState, future<std::vector<VisData<N>>>>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) const & {

    auto tpl_or_err =
      degrid_grid_get_residual_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
    if (hpg::is_value(tpl_or_err)) {
      GridderState gs;
      future<VisDataVector> fvs;
      std::tie(gs, fvs) = hpg::get_value(std::move(tpl_or_err));
      return
        rval<std::tuple<GridderState, future<std::vector<VisData<N>>>>>(
          std::make_tuple(
            std::move(gs),
            future_visibilities_narrow<N>(std::move(fvs))));
    } else {
      return
        rval<std::tuple<GridderState, future<std::vector<VisData<N>>>>>(
          hpg::get_error(std::move(tpl_or_err)));
    }
  };

  /** degrid and grid visibilities, returning residual visibilities (templated,
   * rvalue reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of residual visibilities
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<std::tuple<GridderState, future<std::vector<VisData<N>>>>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) && {

    auto tpl_or_err =
      std::move(*this).degrid_grid_get_residual_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
    if (hpg::is_value(tpl_or_err)) {
      GridderState gs;
      future<VisDataVector> fvs;
      std::tie(gs, fvs) = hpg::get_value(std::move(tpl_or_err));
      return
        rval<std::tuple<GridderState, future<std::vector<VisData<N>>>>>(
          std::make_tuple(
            std::move(gs),
            future_visibilities_narrow<N>(std::move(fvs))));
    } else {
      return
        rval<std::tuple<GridderState, future<std::vector<VisData<N>>>>>(
          hpg::get_error(std::move(tpl_or_err)));
    }
  };


  /** device execution fence
   *
   * @return new GridderState that is a copy of the target, but one in which all
   * tasks on the device have completed
   *
   * Call is rarely explicitly required by users. In fact, any value copy of the
   * target includes an implicit fence.
   *
   * @sa Gridder::fence()
   */
  GridderState
  fence() const &;

  /** device execution fence
   *
   * @return new GridderState that has overwritten the target, but only after
   * all tasks on the device have completed
   *
   * Call is rarely explicitly required by users.
   *
   * @sa Gridder::fence()
   */
  GridderState
  fence() &&;

  /** get copy of grid plane weights
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
  grid_weights() const &;

  /** get copy of grid plane weights
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
  grid_weights() &&;

  /** get copy of grid values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  grid_values() const &;

  /** get copy of grid values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  grid_values() &&;

  /** get copy of model values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  model_values() const &;

  /** get copy of model values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  model_values() &&;

  /** reset grid values to zero
   *
   * Also resets grid plane weights to zero. May invoke fence() on target.
   */
  GridderState
  reset_grid() const &;

  /** reset grid values to zero
   *
   * Also resets grid plane weights to zero. May invoke fence() on target.
   */
  GridderState
  reset_grid() &&;

  /** reset model visibilities to zero
   *
   * May invoke fence() on target
   */
  GridderState
  reset_model() const &;

  /** reset model visibilities to zero
   *
   * May invoke fence() on target
   */
  GridderState
  reset_model() &&;

  /** normalize grid values by scaled weights
   *
   * May invoke fence() on target.
   *
   * @param wgt_factor multiplicative factor applied to weights before
   * normalization
   */
  GridderState
  normalize_by_weights(grid_value_fp wgt_factor = 1) const &;

  /** normalize grid values by scaled weights
   *
   * May invoke fence() on target.
   *
   * @param wgt_factor multiplicative factor applied to weights before
   * normalization
   */
  GridderState
  normalize_by_weights(grid_value_fp wgt_factor = 1) &&;

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param norm post-FFT normalization divisor
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  rval_t<GridderState>
  apply_grid_fft(
    grid_value_fp norm = 1,
    FFTSign sign = grid_fft_sign_dflt,
    bool in_place = true) const &;

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param norm post-FFT normalization divisor
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  rval_t<GridderState>
  apply_grid_fft(
    grid_value_fp norm = 1,
    FFTSign sign = grid_fft_sign_dflt,
    bool in_place = true) &&;

  /** apply FFT to model array planes
   *
   * May invoke fence() on target.
   *
   * @param norm post-FFT normalization divisor
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  rval_t<GridderState>
  apply_model_fft(
    grid_value_fp norm = 1,
    FFTSign sign = model_fft_sign_dflt,
    bool in_place = true) const &;

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param norm post-FFT normalization divisor
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  rval_t<GridderState>
  apply_model_fft(
    grid_value_fp norm = 1,
    FFTSign sign = model_fft_sign_dflt,
    bool in_place = true) &&;

  /** shift grid planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  GridderState
  shift_grid() const &;

  /** shift grid planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  GridderState
  shift_grid() &&;

  /** shift model planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  GridderState
  shift_model() const &;

  /** shift model planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  GridderState
  shift_model() &&;

protected:
  friend class Gridder;

  /** swap member values with another GridderState instance */
  void
  swap(GridderState& other) noexcept;
};

/** specialization of GridderState::future_visibilities_narrow<1> */
template <>
HPG_EXPORT future<std::vector<VisData<1>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs);

/** specialization of GridderState::future_visibilities_narrow<2> */
template <>
HPG_EXPORT future<std::vector<VisData<2>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs);

/** specialization of GridderState::future_visibilities_narrow<3> */
template <>
HPG_EXPORT future<std::vector<VisData<3>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs);

/** specialization of GridderState::future_visibilities_narrow<4> */
template <>
HPG_EXPORT future<std::vector<VisData<4>>>
GridderState::future_visibilities_narrow(future<VisDataVector>&& fvs);

/** Gridder class
 *
 * Instances of this class may be used for a pure object-oriented interface to
 * the functional interface of GridderState. Note that the object-oriented
 * interface of Gridder is slightly more constrained than the functional
 * interface of GridderState, as the GridderState member of a Gridder instance
 * is often modified by Gridder methods through move construction/assignment,
 * and is thus never copied. Managing GridderState instances directly provides
 * greater flexibility, in that the caller has complete control over when
 * GridderState values are copied vs moved. However, the flexibility provided by
 * GridderState consequently makes it easy to create copies of those values when
 * a moved value would have been more efficient (both in resource usage and
 * performance).
 *
 * Depending on the device used for gridding, methods may schedule tasks for
 * asynchronous execution or may block. Even when asynchronous execution is
 * supported by a device, a call may nevertheless block at times, depending upon
 * the internal state of the Gridder instance. See GridderState for more
 * information.
 *
 * @sa GridderState
 */
class HPG_EXPORT Gridder {
protected:

  mutable GridderState state; /**< state maintained by instances */

public:

  /** default constructur */
  Gridder();

protected:

  /** constructor
   *
   * @param device gridder device type
   * @param max_added_tasks maximum number of concurrent tasks (actual
   * number may be less than requested)
   * @param max_visibility_batch_size maximum number of VisData<.> values for
   * calls to grid_visibilities()
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size in logical axis order: X, Y, mrow, cube
   * @param grid_scale in X, Y order
   * @param mueller_indexes CFArray Mueller element indexes, by mrow
   * @param conjugate_mueller_indexes CFArray conjugate Mueller element indexes,
   * by mrow
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is employed,
   * but some devices support additional, concurrent tasks.
   *
   * The value of max_added_tasks and max_visibility_batch_size has an effect on
   * the amount of memory allocated on the selected gridder device. The total
   * amount of memory allocated for visibilities will be approximately equal to
   * max_added_tasks multiplied by sizeof(VisData<N>) for the appropriate value
   * of N.
   */
  Gridder(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    IArrayVector&& mueller_indexes,
    IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    );

public:

  /** Gridder factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @sa Gridder()
   */
  static rval_t<Gridder>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    IArrayVector&& mueller_indexes,
    IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept;

  /** Gridder factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @tparam N number of polarization in visibilities to be gridded
   *
   * @sa Gridder()
   */
  template <unsigned N>
  static rval_t<Gridder>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions = {0, 0, 0, 0}
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create(
        device,
        max_added_tasks,
        max_visibility_batch_size,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        , implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      );
  }

  /** copy constructor
   *
   * Invokes fence() on argument.
   */
  Gridder(const Gridder& other);

  /** move constructor */
  Gridder(Gridder&& other);

  /** copy assignment
   *
   * Invokes fence() on argument
   */
  Gridder&
  operator=(const Gridder&);

  /** move assignment*/
  Gridder&
  operator=(Gridder&&);

  virtual ~Gridder();

  /** device */
  Device
  device() const noexcept;

  /** maximum additional tasks
   *
   * This value may differ from the value provided to the constructor, depending
   * on device limitations */
  unsigned
  max_added_tasks() const noexcept;

  /** maximum number of visibilities passed to gridding kernel at once */
  size_t
  max_visibility_batch_size() const noexcept;

  /** grid size */
  const std::array<unsigned, 4>&
  grid_size() const noexcept;

  /** grid scale */
  std::array<grid_scale_fp, 2>
  grid_scale() const noexcept;

  /** number of visibility polarizations */
  unsigned
  num_polarizations() const noexcept;

  /** null state query */
  bool
  is_null() const noexcept;


  /** size (in bytes) of region allocated for CFArray elements
   *
   * Memory allocations for convolution function regions are made per device
   * task. Values returned for the size of the currently allocated region refer
   * only to the most recently allocated region; multiplying this value by the
   * number of device tasks may be an inaccurate measure of the total allocated
   * region, as asynchronously executing tasks may be using different
   * convolution functions.
   *
   * @param shape if non-null, the memory needed for a CFArray of the given
   * shape; if null, the size of the currently allocated region in the target
   */
  size_t
  convolution_function_region_size(const CFArrayShape* shape) const noexcept;

  /** allocate memory for convolution function
   *
   * Increasing memory allocations for convolution functions are handled
   * automatically by set_convolution_function(), but in a sequence of calls to
   * set_convolution_function() in which later calls require a larger allocation
   * than earlier calls, it may be advantageous to use this method in order to
   * allocate the maximum memory that will be required by the sequence before
   * starting the sequence, which will then permit the sequence to proceed
   * without any reallocations. To release all memory allocated for the
   * convolution function, the caller may pass a null pointer for the method
   * argument. Invokes fence() on the target.
   *
   * @param shape shape of CFArray for which to allocate memory (per task)
   *
   * @return new GridderState that is a copy of the target, but with memory
   * allocated for convolution function, or error
   */
  opt_t<Error>
  allocate_convolution_function_region(const CFArrayShape* shape);

  /** set convolution function
   *
   * May invoke fence() on target.
   *
   * the provided convolution function will be used for gridding until this
   * function is called again
   *
   * @param host_device device to use for changing array layout
   * @param cf convolution function array
   */
  opt_t<Error>
  set_convolution_function(Device host_device, CFArray&&);

  /** set visibility model
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after setting model
   *
   * @param host_device device to use for copying model values
   * @param gv visibility model
   *
   * @sa Gridder::set_model()
   */
  opt_t<Error>
  set_model(Device host_device, GridValueArray&& gv);

  /** grid visibilities (template-free version)
   *
   * May invoke fence() on target.
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  opt_t<Error>
  grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities);

  /** grid visibilities (template version)
   *
   * May invoke fence() on target.
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  opt_t<Error>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) {

    return
      grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
  }

  /** degrid and grid visibilities (template-free version)
   *
   * May invoke fence() on target.
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  opt_t<Error>
  degrid_grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities);

  /** degrid and grid visibilities (template version)
   *
   * May invoke fence() on target.
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  opt_t<Error>
  degrid_grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) {

    return
      degrid_grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
  }

  /** degrid visibilities, returning predicted visibilities (template-free
   * version)
   *
   * May invoke fence() on target.
   *
   * @return future of predicted visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<future<VisDataVector>>
  degrid_get_predicted_visibilities(
    Device host_device,
    VisDataVector&& visibilities);

  /** degrid visibilities, returning predicted visibilities (template version)
   *
   * May invoke fence() on target.
   *
   * @return future of predicted visibilities
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<future<std::vector<VisData<N>>>>
  degrid_get_predicted_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) {

    auto fvs_or_err =
      degrid_get_predicted_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
    if (hpg::is_value(fvs_or_err))
      return
        GridderState::future_visibilities_narrow<N>(
          hpg::get_value(std::move(fvs_or_err)));
    else
      return hpg::get_error(std::move(fvs_or_err));
  }

  /** degrid and grid visibilities, returning residual visibilities
   * (template-free version)
   *
   * May invoke fence() on target.
   *
   * @return future of residual visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  rval_t<future<VisDataVector>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    VisDataVector&& visibilities);

  /** degrid and grid visibilities, returning residual visibilities (template
   * version)
   *
   * May invoke fence() on target.
   *
   * @return future of residual visibilities
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   */
  template <unsigned N>
  rval_t<future<std::vector<VisData<N>>>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities) {

    auto fvs_or_err =
      degrid_grid_get_residual_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)));
    if (hpg::is_value(fvs_or_err))
      return
        GridderState::future_visibilities_narrow<N>(
          hpg::get_value(std::move(fvs_or_err)));
    else
      return hpg::get_error(std::move(fvs_or_err));
  }

  /** device execution fence
   *
   * Returns after all tasks on device have completed. Call is rarely explicitly
   * required by users.
   */
  void
  fence() const;

  /** get copy of grid plane weights
   *
   * Invokes fence() on target.
   */
  std::unique_ptr<GridWeightArray>
  grid_weights() const;

  /** get copy of grid values
   *
   * Invokes fence() on target.
   */
  std::unique_ptr<GridValueArray>
  grid_values() const;

  /** get copy of model values
   *
   * Invokes fence() on target.
   */
  std::unique_ptr<GridValueArray>
  model_values() const;

  /** reset grid values to zero
   *
   * Also resets grid plane weights to zero. May invoke fence() on target.
   */
  void
  reset_grid();

  /** reset model values to zero
   */
  void
  reset_model();

  /** normalize grid values by scaled weights
   *
   * May invoke fence() on target.
   *
   * @param wgt_factor multiplicative factor applied to weights before
   * normalization
   */
  void
  normalize_by_weights(grid_value_fp wgt_factor = 1);

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param norm post-FFT normalization divisor
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  opt_t<Error>
  apply_grid_fft(
    grid_value_fp norm = 1,
    FFTSign sign = grid_fft_sign_dflt,
    bool in_place = true);

  /** apply FFT to model array planes
   *
   * May invoke fence() on target.
   *
   * @param norm post-FFT normalization divisor
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  opt_t<Error>
  apply_model_fft(
    grid_value_fp norm = 1,
    FFTSign sign = model_fft_sign_dflt,
    bool in_place = true);

  /** shift grid planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  void
  shift_grid();

  /** shift model planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  void
  shift_model();

protected:

  /** move constructor */
  Gridder(GridderState&& st);
};

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
