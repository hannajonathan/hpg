#pragma once

#include <complex>
#include <memory>

#include "hpg_export.h"

namespace hpg {

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

  virtual std::complex<float>
    operator()(unsigned x, unsigned y) const = 0;
};

struct HPG_EXPORT Gridder;

/** gridder state
 *
 * A container for the entire state needed to do gridding as a value, including
 * gridded visibility data and possibly a convolution function array. Used by
 * the Gridder class, but may also be used directly for its greater flexibility.
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
   * @param grid_size, in logical axis order: X, Y, channel
   * @param grid_scale, in X, Y order
   *
   * @sa Gridder::Gridder()
   */
  GridderState(
    Device device,
    const std::array<unsigned, 3>& grid_size,
    const std::array<float, 2>& grid_scale);

  /** copy constructor
   *
   * copies all state
   */
  GridderState(GridderState&);

  /** move constructor
   */
  GridderState(GridderState&&);

  /** copy assignment
   *
   * copies all state
   */
  GridderState&
  operator=(GridderState&);

  /** move assignment
   */
  GridderState&
  operator=(GridderState&&);

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
  set_convolution_function(
    Device host_device,
    const CF2& cf) &;

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
  set_convolution_function(
    Device host_device,
    const CF2& cf) &&;

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
  fence() &;

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
};

/** Gridder class
 *
 * Instances of this class may be used for a pure object-oriented interface to
 * the functional interface of GridderState. Note that the object-oriented
 * interface of Gridder is slightly more constrained than the functional
 * interface of GridderState, as the GridderState member of a Gridder instance
 * is often modified by Gridder methods through move construction/assignment,
 * and is thus never copied. Managing GridderState instances directly provides
 * greate flexibility, in that the caller has complete control over when
 * GridderState values are copied vs moved. However, the flexibility provided by
 * GridderState consequently makes it easy to create copies of those values when
 * a moved value would have been more efficient (both in resource usage and
 * performance).
 */
class Gridder {
public:

  GridderState state; /**< state maintained by instances */

  // NB: grid_size logical axis order: X, Y, ch

  /** constructor
   *
   * @param device gridder device type
   * @param grid_size, in logical axis order: X, Y, channel
   * @param grid_scale, in X, Y order
   */
  Gridder(
    Device device,
    const std::array<unsigned, 3>& grid_size,
    const std::array<float, 2>& grid_scale)
    : state(GridderState(device, grid_size, grid_scale)) {}

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

  /** device execution fence
   *
   * Returns after all tasks on device have completed. Call is rarely explicitly
   * required by users.
   */
  void
  fence() {
    state = std::move(state).fence();
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
