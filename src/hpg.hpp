#pragma once

#include "hpg_config.hpp"
#include "hpg_rval.hpp"
#include "hpg_error.hpp"

#include <complex>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#if HPG_API >= 17
# include <optional>
#endif

#include "hpg_export.h"

/** @file hpg.hpp
 *
 * Main header file for top-level HPG API
 */

/** top-level HPG API namespace
 */
namespace hpg {

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
  HPX, /**< HIP device */
};

/** supported devices */
HPG_EXPORT const std::set<Device>&
devices() noexcept;

/** supported host devices */
HPG_EXPORT const std::set<Device>&
host_devices() noexcept;

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
  ScopeGuard();

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
using vis_uvw_fp = float;
/** visibility weight floating point type */
using vis_weight_fp = float;
/** visibility frequency floating point type */
using vis_frequency_fp = float;
/** visibility phase floating point type */
using vis_phase_fp = double;
/** CF phase screen floating point type */
using cf_phase_screen_fp = float;

// vis_uvw_t can be any type that supports std::get<N>() for element access
/** UVW coordinate type */
using vis_uvw_t = std::array<vis_uvw_fp, 3>;

// cf_phase_screen_t can be any type that supports std::get<N>() for element
// access
/** CF phase screen type */
using cf_phase_screen_t = std::array<cf_phase_screen_fp, 2>;

/** visibility CF index type
 *
 * in terms of full CFArray indexes, order is cube, grp
 */
using vis_cf_index_t = std::pair<unsigned, unsigned>;

/** type to represent a possible error */
#if HPG_API >= 17
using opt_error_t = std::optional<Error>;
#else // HPG_API < 17
using opt_error_t = std::unique_ptr<Error>;
#endif //HPG_API >= 17

template <unsigned N>
struct VisData {

  std::array<std::complex<visibility_fp>, N> m_visibilities;
  std::array<vis_weight_fp, N> m_weights;
  vis_frequency_fp m_frequency;
  vis_phase_fp m_phase;
  vis_uvw_t m_uvw;

  VisData(
    const std::array<std::complex<visibility_fp>, N>& visibilities,
    const std::array<vis_weight_fp, N>& weights,
    const vis_frequency_fp& frequency,
    const vis_phase_fp& phase,
    const vis_uvw_t& uvw)
    : m_visibilities(visibilities)
    , m_weights(weights)
    , m_frequency(frequency)
    , m_phase(phase)
    , m_uvw(uvw) {}

  VisData() {}
};

namespace Impl {
struct HPG_EXPORT State;
struct HPG_EXPORT GridderState;

template <typename T>
inline constexpr int
sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

struct VisDataVector {

  unsigned m_npol;

  union vis_data {
    std::vector<VisData<1>> vd1;
    std::vector<VisData<2>> vd2;
    std::vector<VisData<3>> vd3;
    std::vector<VisData<4>> vd4;

    vis_data(std::vector<VisData<1>>&& v)
      : vd1(std::move(v)) {}
    vis_data(std::vector<VisData<2>>&& v)
      : vd2(std::move(v)) {}
    vis_data(std::vector<VisData<3>>&& v)
      : vd3(std::move(v)) {}
    vis_data(std::vector<VisData<4>>&& v)
      : vd4(std::move(v)) {}

    ~vis_data() {}
  } m_vis_data;

  VisDataVector(std::vector<VisData<1>>&& v)
    : m_npol(1)
    , m_vis_data(vis_data(std::move(v))) {}

  VisDataVector(std::vector<VisData<2>>&& v)
    : m_npol(2)
    , m_vis_data(vis_data(std::move(v))) {}

  VisDataVector(std::vector<VisData<3>>&& v)
    : m_npol(3)
    , m_vis_data(vis_data(std::move(v))) {}

  VisDataVector(std::vector<VisData<4>>&& v)
    : m_npol(4)
    , m_vis_data(vis_data(std::move(v))) {}
};

} // end namespace Impl

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

  static constexpr unsigned rank = 5;

  virtual unsigned
  oversampling() const = 0;

  virtual unsigned
  num_groups() const = 0;

  virtual std::array<unsigned, 4>
  extents(unsigned grp) const = 0;

  virtual ~CFArrayShape() {}
};

/** base class for convolution functions */
class HPG_EXPORT CFArray
  : public CFArrayShape {
public:

  using value_type = std::complex<cf_fp>;

  static constexpr unsigned padding = 2;

  virtual const char*
  layout() const {
    return cf_layout_unspecified_version;
  }

  virtual std::complex<cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mrow,
    unsigned cube,
    unsigned grp)
    const = 0;

  std::array<unsigned, 2>
  radii(unsigned grp) const {
    auto os = oversampling();
    auto ext = extents(grp);
    return {
      ((ext[0] - 2 * padding * os) / os) / 2,
      ((ext[1] - 2 * padding * os) / os) / 2};
  }

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
    std::vector<std::tuple<std::array<unsigned, 4>, std::vector<value_type>>>&&
      arrays);

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

  static constexpr unsigned rank = 4;

  using value_type = std::complex<grid_value_fp>;

  virtual unsigned
  extent(unsigned dim) const = 0;

  virtual const value_type&
  operator()(unsigned x, unsigned y, unsigned mrow, unsigned cube) const = 0;

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
  opt_error_t
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

  static constexpr unsigned rank = 2;

  using value_type = grid_value_fp;

  virtual unsigned
  extent(unsigned dim) const = 0;

  virtual const value_type&
  operator()(unsigned mrow, unsigned cube) const = 0;

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
  opt_error_t
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

enum class HPG_EXPORT FFTSign {
  POSITIVE,
  NEGATIVE
};

constexpr FFTSign fft_sign_dflt = FFTSign::POSITIVE;

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
   * @param max_visibility_batch_size maximum number of visibilities to pass to
   * the gridding kernel at once
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size, in logical axis order: X, Y, mrow, cube
   * @param grid_scale, in X, Y order
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is
   * employed, but some devices support additional, concurrent tasks.
   *
   * The value of max_added_tasks and max_visibility_batch_size has an effect on
   * the amount of memory allocated on the selected gridder device. The total
   * amount of memory allocated for visibilities will be approximately equal to
   * max_added_tasks multiplied by
   * GridderState::visibility_batch_allocation(max_visibility_batch_size).
   *
   * @sa Gridder::Gridder()
   */
  GridderState(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions = {0, 0, 0, 0}
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    );

public:

  /** GridderState factory method
   *
   * does not throw an exception if device argument names an unsupported device
   */
  static rval_t<GridderState>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions = {0, 0, 0, 0}
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept;

  /** copy constructor
   *
   * Copies all state. Invokes fence() on argument.
   */
  GridderState(const GridderState&);

  /** move constructor
   */
  GridderState(GridderState&&);

  virtual ~GridderState();

  static size_t
  visibility_batch_allocation(size_t batch_size);

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

  rval_t<GridderState>
  set_model(Device host_device, GridValueArray&& gv) const &;

  rval_t<GridderState>
  set_model(Device host_device, GridValueArray&& gv) &&;

protected:

  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    Impl::VisDataVector&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes) const &;

public:

  /** grid some visibilities
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param grid_cubes visibility grid cube indexes
   * @param cf_indexes visibility convolution function indexes
   *
   * @sa Gridder::grid_visibilities()
   */
  template <unsigned N>
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes) const & {

    return
      grid_visibilities(
        host_device,
        Impl::VisDataVector(std::move(visibilities)),
        std::move(grid_cubes),
        std::move(cf_indexes));
  };

protected:

  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    Impl::VisDataVector&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes,
    std::vector<cf_phase_screen_t>&& cf_phase_screens) const &;

public:

  /** grid some visibilities
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param grid_cubes visibility grid cube indexes
   * @param cf_indexes visibility convolution function indexes
   * @param cf_phase_screens visibility CF phase screen parameters
   *
   * @sa Gridder::grid_visibilities()
   */
  template <unsigned N>
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes,
    std::vector<cf_phase_screen_t>&& cf_phase_screens) const & {

    return
      grid_visibilities(
        host_device,
        Impl::VisDataVector(std::move(visibilities)),
        std::move(grid_cubes),
        std::move(cf_indexes),
        std::move(cf_phase_screens));
  }

protected:

  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    Impl::VisDataVector&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes) &&;

public:

  /** grid some visibilities
   *
   * May invoke fence() on target.
   *
   * @return new GridderState that has overwritten the target, but after
   * gridding task has been submitted to device queue
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param grid_cubes visibility grid cube indexes
   * @param cf_indexes visibility convolution function indexes
   *
   * @sa Gridder::grid_visibilities()
   */
  template <unsigned N>
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes) && {

    return
      std::move(*this)
      .grid_visibilities(
        host_device,
        Impl::VisDataVector(std::move(visibilities)),
        std::move(grid_cubes),
        std::move(cf_indexes));
  }

protected:

  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    Impl::VisDataVector&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes,
    std::vector<cf_phase_screen_t>&& cf_phase_screens) &&;

public:

  /** grid some visibilities
   *
   * May invoke fence() on target.
   *
   * @return new GridderState that has overwritten the target, but after
   * gridding task has been submitted to device queue
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param grid_cubes visibility grid cube indexes
   * @param cf_indexes visibility convolution function indexes
   * @param cf_phase_screens visibility CF phase screen parameters
   *
   * @sa Gridder::grid_visibilities()
   */
  template <unsigned N>
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes,
    std::vector<cf_phase_screen_t>&& cf_phase_screens) && {

    return
      std::move(*this)
      .grid_visibilities(
        host_device,
        Impl::VisDataVector(std::move(visibilities)),
        std::move(grid_cubes),
        std::move(cf_indexes),
        std::move(cf_phase_screens));
  }

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
  normalize(grid_value_fp wgt_factor = 1) const &;

  /** normalize grid values by scaled weights
   *
   * May invoke fence() on target.
   *
   * @param wgt_factor multiplicative factor applied to weights before
   * normalization
   */
  GridderState
  normalize(grid_value_fp wgt_factor = 1) &&;

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  rval_t<GridderState>
  apply_fft(FFTSign sign = fft_sign_dflt, bool in_place = true) const  &;

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  rval_t<GridderState>
  apply_fft(FFTSign sign = fft_sign_dflt, bool in_place = true) &&;

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

protected:
  friend class Gridder;

  void
  swap(GridderState& other) noexcept;
};

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
   * @param max_visibility_batch_size maximum number of visibilities to pass to
   * the gridding kernel at once
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size, in logical axis order: X, Y, mrow, cube
   * @param grid_scale, in X, Y order
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is employed,
   * but some devices support additional, concurrent tasks.
   *
   * The value of max_added_tasks and max_visibility_batch_size has an effect on
   * the amount of memory allocated on the selected gridder device. The total
   * amount of memory allocated for visibilities will be approximately equal to
   * max_added_tasks multiplied by
   * GridderState::visibility_batch_allocation(max_visibility_batch_size).
   */
  Gridder(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale);

public:

  /** Gridder factory method
   *
   * does not throw an exception if device argument names an unsupported device
   */
  static rval_t<Gridder>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions = {0, 0, 0, 0}
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept;

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
  opt_error_t
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
  opt_error_t
  set_convolution_function(Device host_device, CFArray&&);

  opt_error_t
  set_model(Device host_device, GridValueArray&& gv);

protected:

  opt_error_t
  grid_visibilities(
    Device host_device,
    Impl::VisDataVector&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes);

public:

  /** grid visibilities
   *
   * May invoke fence() on target.
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param grid_cubes visibility grid cube indexes
   * @param cf_indexes visibility convolution function indexes
   */
  template <unsigned N>
  opt_error_t
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes) {

    return
      grid_visibilities(
        host_device,
        Impl::VisDataVector(std::move(visibilities)),
        std::move(grid_cubes),
        std::move(cf_indexes));
  }

protected:

  opt_error_t
  grid_visibilities(
    Device host_device,
    Impl::VisDataVector&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes,
    std::vector<cf_phase_screen_t>&& cf_phase_screens);

public:

  /** grid visibilities
   *
   * May invoke fence() on target.
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param grid_cubes visibility grid cube indexes
   * @param cf_indexes visibility convolution function indexes
   * @param cf_phase_screens visibility CF phase screen parameters
   */
  template <unsigned N>
  opt_error_t
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    std::vector<unsigned>&& grid_cubes,
    std::vector<vis_cf_index_t>&& cf_indexes,
    std::vector<cf_phase_screen_t>&& cf_phase_screens) {

    return
      grid_visibilities(
        host_device,
        Impl::VisDataVector(std::move(visibilities)),
        std::move(grid_cubes),
        std::move(cf_indexes),
        std::move(cf_phase_screens));
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

  /** reset grid values to zero
   *
   * Also resets grid plane weights to zero. May invoke fence() on target.
   */
  void
  reset_grid();

  void
  reset_model();

  /** normalize grid values by weights
   *
   * May invoke fence() on target.
   *
   * @param wgt_factor multiplicative factor applied to weights before
   * normalization
   */
  void
  normalize(grid_value_fp wgt_factor = 1);

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  opt_error_t
  apply_fft(FFTSign sign = fft_sign_dflt, bool in_place = true);

  /** shift grid planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  void
  shift_grid();

protected:

  Gridder(GridderState&& st);
};

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
