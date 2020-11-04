#pragma once

#include <complex>
#include <memory>
#include <vector>

#include "hpg_export.h"

namespace hpg {

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
using vis_frequency_fp = double;
/** visibility phase floating point type */
using vis_phase_fp = double;

// vis_uvw_t can be any type that supports std::get<N>() for element access
using vis_uvw_t = std::array<vis_uvw_fp, 3>;

/**
 * backend device type
 */
enum class HPG_EXPORT Device {
#ifdef HPG_ENABLE_SERIAL
  Serial,
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  OpenMP,
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  Cuda,
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
  HPX,
#endif // HPG_ENABLE_HPX
};

namespace Impl {
struct HPG_EXPORT State;
} // end namespace Impl

/** base class for 2d convolution functions */
class HPG_EXPORT CF2 {
public:

  unsigned oversampling;

  std::array<unsigned, 2> extent;

  virtual std::complex<cf_fp>
    operator()(unsigned x, unsigned y) const = 0;
};

struct HPG_EXPORT Gridder;

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
 * particular, note that several devices support no asynchronous
 * execution. Class methods that submit work to a device, that is, for either
 * data movement or computation, may block when the number of asynchronously
 * running tasks is at its limit, until one of those running tasks
 * completes. Otherwise, the methods that create device tasks will return as
 * soon the task has been submitted to the device.
 *
 * In general, using a GridderState method on an instance creates a new (value)
 * copy of the target. For example,
 *    GridderState s0;
 *    GridderState s1 = s0.fence();
 * will create a copy of s0. Note that a copy will include the grid and the
 * convolution function array. (Many of these methods are non-const on the
 * target since a fence operation may be involved.) To avoid the copy, the
 * following pattern can be used instead:
 *    GridderState s0;
 *    GridderState s1 = std::move(s0).fence();
 * Note, however, that the value of s0 in the previous example after the call to
 * fence() will be in the null state, which is likely not of much further use to
 * the caller.
 */
struct GridderState {
protected:
  friend class Gridder;

  // state cannot be a unique_ptr since Impl::State is here an incomplete type
  std::shared_ptr<Impl::State> impl; /**< state implementation */

public:
  /** default constructor
   *
   * the null state, most methods will fail when called on a target with this
   * value
   */
  GridderState();

  /** constructor
   *
   * create a GridderState
   *
   * @param device gridder device type
   * @param max_async_tasks maximum number of asynchronous tasks (actual number
   * may be less than requested)
   * @param grid_size, in logical axis order: X, Y, plane
   * @param grid_scale, in X, Y order
   *
   * @sa Gridder::Gridder()
   */
  GridderState(
    Device device,
    unsigned max_async_tasks,
    const std::array<unsigned, 3>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale);

  /** copy constructor
   *
   * copies all state
   */
  GridderState(const volatile GridderState&);

  /** move constructor
   */
  GridderState(GridderState&&);

  /** copy assignment
   *
   * copies all state
   */
  GridderState&
  operator=(const volatile GridderState&);

  /** move assignment
   */
  GridderState&
  operator=(GridderState&&);

  /** device */
  Device
  device() const;

  /** maximum active tasks
   *
   * This value may differ from the value provided to the constructor, depending
   * on device limitations */
  unsigned
  max_async_tasks() const;

  /** grid size */
  const std::array<unsigned, 3>&
  grid_size() const;

  /** grid scale */
  const std::array<grid_scale_fp, 2>&
  grid_scale() const;

  /** set convolution function
   *
   * @return new GridderState that is a copy of the target, but with provided
   * convolution function for subsequent gridding
   *
   * @param host_device device to use for changing array layout
   * @param cf 2d convolution function array
   *
   * @sa Gridder::set_convolution_function()
   */
  GridderState
  set_convolution_function(Device host_device, const CF2& cf) const volatile &;

  /** set convolution function
   *
   * @return new GridderState that has overwritten the target, but with provided
   * convolution function for subsequent gridding
   *
   * @param host_device device to use for changing array layout
   * @param cf 2d convolution function array
   *
   * @sa Gridder::set_convolution_function()
   */
  GridderState
  set_convolution_function(Device host_device, const CF2& cf) &&;

  /** grid some visibilities
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at visibility_weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param visibility_weights visibility weights
   * @param visibility_frequencies visibility frequencies
   * @param visibility_phase visibility phase difference
   * @param visibility_coordinates visibility coordinates
   *
   * @sa Gridder::grid_visibilities()
   */
  GridderState
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phase,
    const std::vector<vis_uvw_t>& visibility_coordinates)
    const volatile &;

  /** grid some visibilities
   *
   * @return new GridderState that has overwritten the target, but after
   * gridding task has been submitted to device queue
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at visibility_weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param visibility_weights visibility weights
   * @param visibility_frequencies visibility frequencies
   * @param visibility_phase visibility phase difference
   * @param visibility_coordinates visibility coordinates
   *
   * @sa Gridder::grid_visibilities()
   */
  GridderState
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phase,
    const std::vector<vis_uvw_t>& visibility_coordinates) &&;

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
  fence() const volatile &;

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

  /** get normalization factor */
  std::pair<GridderState, std::complex<grid_value_fp>>
  get_normalization() const volatile &;

  /** get normalization factor */
  std::pair<GridderState, std::complex<grid_value_fp>>
  get_normalization() &&;

  /** set normalization factor
   *
   * @return normalization factor value before setting new value
   */
  std::pair<GridderState, std::complex<grid_value_fp>>
  set_normalization(const std::complex<grid_value_fp>& val = 0) &;

  /** set normalization factor
   *
   * @return normalization factor value before setting new value
   */
  std::pair<GridderState, std::complex<grid_value_fp>>
  set_normalization(const std::complex<grid_value_fp>& val = 0) &&;

protected:
  friend class Gridder;

  void
  swap(GridderState& other);
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
class Gridder {
public:

  mutable GridderState state; /**< state maintained by instances */

  /** constructor
   *
   * @param device gridder device type
   * @param max_async_tasks maximum number of concurrent tasks (actual number
   * may be less than requested)
   * @param grid_size, in logical axis order: X, Y, plane
   * @param grid_scale, in X, Y order
   */
  Gridder(
    Device device,
    unsigned max_async_tasks,
    const std::array<unsigned, 3>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale)
    : state(GridderState(device, max_async_tasks, grid_size, grid_scale)) {}

  /** copy constructor */
  Gridder(const volatile Gridder& other)
    : state(other.state) {}

  /** move constructor */
  Gridder(Gridder&& other)
    : state(std::move(other).state) {}

  /** copy assignment */
  Gridder&
  operator=(const volatile Gridder&);

  /** move assignment*/
  Gridder&
  operator=(Gridder&&);

  /** device */
  Device
  device() const {
    return state.device();
  }

  /** maximum active tasks
   *
   * This value may differ from the value provided to the constructor, depending
   * on device limitations */
  unsigned
  max_async_tasks() const {
    return state.max_async_tasks();
  }

  /** grid size */
  const std::array<unsigned, 3>&
  grid_size() const {
    return state.grid_size();
  }

  /** grid scale */
  const std::array<grid_scale_fp, 2>&
  grid_scale() const {
    return state.grid_scale();
  }

  /** set convolution function
   *
   * the provided convolution function will be used for gridding until this
   * function is called again
   *
   * @param host_device device to use for changing array layout
   * @param cf 2d convolution function array
   */
  void
  set_convolution_function(Device host_device, const CF2& cf) {
    state = std::move(state).set_convolution_function(host_device, cf);
  }

  /** grid visibilities
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at visibility_weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param visibility_weights visibility weights
   * @param visibility_frequencies visibility frequencies
   * @param visibility_phase visibility phase difference
   * @param visibility_coordinates visibility coordinates
   */
  void
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phase,
    const std::vector<vis_uvw_t>& visibility_coordinates) {

    state =
      std::move(state)
      .grid_visibilities(
        host_device,
        visibilities,
        visibility_weights,
        visibility_frequencies,
        visibility_phase,
        visibility_coordinates);
  }

  /** device execution fence
   *
   * Returns after all tasks on device have completed. Call is rarely explicitly
   * required by users.
   */
  void
  fence() const volatile {
    const_cast<Gridder*>(this)->state =
      std::move(const_cast<Gridder*>(this)->state).fence();
  }

  /** get normalization factor */
  std::complex<grid_value_fp>
  get_normalization() const volatile {
    std::complex<grid_value_fp> result;
    std::tie(const_cast<Gridder*>(this)->state, result) =
      std::move(const_cast<Gridder*>(this)->state).get_normalization();
    return result;
  }

  /** set normalization factor
   *
   * @return normalization factor value before setting new value
   */
  std::complex<grid_value_fp>
  set_normalization(const std::complex<grid_value_fp>& val = 0) {
    std::complex<grid_value_fp> result;
    std::tie(const_cast<Gridder*>(this)->state, result) =
      std::move(const_cast<Gridder*>(this)->state).set_normalization(val);
    return result;
  }
};

/** global initialization of hpg
 *
 * Function is idempotent, but should not be called by a process after a call to
 * hpg::finalize()
 */
void initialize();

/** global finalization of hpg
 *
 * Function is idempotent, but should only be called by a process after a call
 * to hpg::initialize()
 */
void finalize();

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
