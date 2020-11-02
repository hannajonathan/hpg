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

struct HPG_EXPORT Gridder;

/** gridder state
 *
 * Used by the Gridder class, but may also be used to directly access the
 * functional (static) interface of hpg::Gridder
 */
struct GridderState {
protected:
  friend class Gridder;

  // state cannot be a unique_ptr since Impl::State is here an incomplete type
  std::shared_ptr<Impl::State> impl; /**< state implementation */

public:
  /** default constructor
   *
   * the null state, static hpg::Gridder functions will generally fail when
   * called with this value
   */
  GridderState();

  /** copy constructor
   *
   * copies all state, including current grid values
   */
  GridderState(GridderState&);

  /** move constructor
   */
  GridderState(GridderState&&);

  /** copy assignment
   *
   * copies all state, including grid values
   */
  GridderState&
  operator=(GridderState&);

  /** move assignment
   */
  GridderState&
  operator=(GridderState&&);
};

/** base class for 2d convolution functions */
class HPG_EXPORT CF2 {
public:

  unsigned oversampling;

  std::array<unsigned, 2> extent;

  virtual std::complex<float>
  operator()(unsigned x, unsigned y) const = 0;
};

/** Gridder class
 *
 * Instances of this class may be used for an object-oriented interface to the
 * functional, static interface. Note that the object-oriented interface is
 * slightly more constrained than the functional interface, as GridderState
 * member values are often modified by Gridder methods through move
 * construction/assignment, and are thus never copied. Using the functional
 * interface via the static class methods with GridderState value arguments is
 * more flexible, in that the caller has complete control over when GridderState
 * values are copied vs moved. However, the flexibility provided by the
 * functional interface consequently makes it easy to create copies of
 * GridderState values when a moved value would be more efficient (both in
 * resource usage and performance).
 *
 * In general, in the functional interface, all GridderState arguments are
 * passed by value, which can cause an implicit copy to be made when calling
 * those functions. For example,
 *    GridderState s;
 *    Gridder::fence(s);
 * will implicitly create a copy of s. To avoid the implicit copy, the following
 * pattern can be used instead:
 *    GridderState s;
 *    Gridder::fence(std::move(s));
 * Note, however, that the value of s in the previous example after the call to
 * Gridder::fence() will be in the null state, which is likely not of much
 * further use to the caller.
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
    : state(init(device, grid_size, grid_scale)) {}

  void
  resample_to_grid(Device host_device, const CF2& cf) {
    state = resample_to_grid(std::move(state), host_device, cf);
  }

  /** device execution fence
   *
   * Returns after all tasks on device have completed. Call is rarely explicitly
   * required by users.
   */
  void
  fence() {
    state = fence(std::move(state));
  }

  /** GridderState constructor
   *
   * Functional interface: create a GridderState
   *
   * @sa Gridder::Gridder() for argument descriptions, and an object-oriented
   * equivalent
   */
  static GridderState
  init(
    Device device,
    const std::array<unsigned, 3>& grid_size,
    const std::array<float, 2>& grid_scale);

  static GridderState
  resample_to_grid(GridderState state, Device host_device, const CF2& cf);

  /** device execution fence
   *
   * @sa Gridder::fence() for a description, and an object-oriented equivalent
   *
   */
  static GridderState
  fence(GridderState state);
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
