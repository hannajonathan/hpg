#pragma once

#include <complex>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
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
  Serial, /**< serial device */
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  OpenMP, /**< OpenMP device */
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  Cuda, /**< CUDA device */
#endif // HPG_ENABLE_CUDA
#ifdef HPG_ENABLE_HPX
  HPX, /**< HIP device */
#endif // HPG_ENABLE_HPX
};

namespace Impl {
struct HPG_EXPORT State;
} // end namespace Impl

/** base class for convolution functions */
class HPG_EXPORT CFArray {
public:

  virtual unsigned
  oversampling() const = 0;

  virtual unsigned
  extent(unsigned dim) const = 0;

  virtual std::complex<cf_fp>
  operator()(unsigned x, unsigned y, unsigned cf_sto, unsigned cube)
    const = 0;

  virtual ~CFArray() {}
};

/** wrapper for read-only access to grid values
 *
 * @todo: replace with mdspan?
 */
class HPG_EXPORT GridValueArray {
public:

  virtual unsigned
  extent(unsigned dim) const = 0;

  virtual std::complex<grid_value_fp>
  operator()(unsigned x, unsigned y, unsigned stokes, unsigned cube) const = 0;

  virtual ~GridValueArray() {}
};

/** wrapper for read-only access to grid weights
 *
 * @todo: replace with mdspan?
 */
class HPG_EXPORT GridWeightArray {
public:

  virtual unsigned
  extent(unsigned dim) const = 0;

  virtual grid_value_fp
  operator()(unsigned stokes, unsigned cube) const = 0;

  virtual ~GridWeightArray() {}
};

class HPG_EXPORT Error {
private:

  std::string m_msg;

public:
  Error(const std::string& msg)
    : m_msg(msg) {}

  const std::string&
  message() const {
    return m_msg;
  }
};

struct HPG_EXPORT Gridder;

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
 *    GridderState s0;
 *    GridderState s1 = s0.fence();
 * will create a copy of s0. Note that a copy will include the grid and the
 * convolution function array. (Many of these methods are const volatile on the
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
   * @param max_added_tasks maximum number of additional tasks (actual number
   * may be less than requested)
   * @param grid_size, in logical axis order: X, Y, stokes, cube
   * @param grid_scale, in X, Y order
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is
   * employed, but some devices support additional, concurrent tasks.
   *
   * @sa Gridder::Gridder()
   */
  GridderState(
    Device device,
    unsigned max_added_tasks,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale);

  /** copy constructor
   *
   * Copies all state. Invokes fence() on argument.
   */
  GridderState(const volatile GridderState&);

  /** move constructor
   */
  GridderState(GridderState&&);

  /** copy assignment
   *
   * Copies all state. Invokes fence() on argument.
   */
  GridderState&
  operator=(const volatile GridderState&);

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

  /** grid size */
  const std::array<unsigned, 4>&
  grid_size() const noexcept;

  /** grid scale */
  const std::array<grid_scale_fp, 2>&
  grid_scale() const noexcept;

  /** null state query */
  bool
  is_null() const noexcept;

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
  std::variant<Error, GridderState>
  set_convolution_function(Device host_device, const CFArray& cf) &;

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
  std::tuple<std::optional<Error>, GridderState>
  set_convolution_function(Device host_device, const CFArray& cf) &&;

  /** grid some visibilities
   *
   * May invoke fence() on target.
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
   * @param visibility_grid_cubes visibility grid cube indexes
   * @param visibility_cf_cubes visibility convolution function cube indexes
   * @param visibility_weights visibility weights
   * @param visibility_frequencies visibility frequencies
   * @param visibility_phases visibility phase differences
   * @param visibility_coordinates visibility coordinates
   *
   * @sa Gridder::grid_visibilities()
   */
  GridderState
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<unsigned> visibility_grid_cubes,
    const std::vector<unsigned> visibility_cf_cubes,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phases,
    const std::vector<vis_uvw_t>& visibility_coordinates) &;

  /** grid some visibilities
   *
   * May invoke fence() on target.
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
   * @param visibility_grid_cubes visibility grid cube indexes
   * @param visibility_cf_cubes visibility convolution function cube indexes
   * @param visibility_weights visibility weights
   * @param visibility_frequencies visibility frequencies
   * @param visibility_phases visibility phase differences
   * @param visibility_coordinates visibility coordinates
   *
   * @sa Gridder::grid_visibilities()
   */
  GridderState
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<unsigned> visibility_grid_cubes,
    const std::vector<unsigned> visibility_cf_cubes,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phases,
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

  /** get grid plane weights
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
  grid_weights() const volatile &;

  /** get grid plane weights
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridWeightArray>>
  grid_weights() &&;

  /** get grid values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  grid_values() const volatile &;

  /** get grid values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  grid_values() &&;

  /** reset grid values to zero
   *
   * May invoke fence() on target
   */
  GridderState
  reset_grid() &;

  /** reset grid values to zero
   *
   * May invoke fence() on target
   */
  GridderState
  reset_grid() &&;

  /** normalize grid values by scaled weights
   *
   * May invoke fence() on target.
   *
   * @param wgt_factor multiplicative factor applied to weights before
   * normalization
   */
  GridderState
  normalize(grid_value_fp wgt_factor = 1) &;

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
  std::variant<Error, GridderState>
  apply_fft(FFTSign sign = fft_sign_dflt, bool in_place = true) &;

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  std::tuple<std::optional<Error>, GridderState>
  apply_fft(FFTSign sign = fft_sign_dflt, bool in_place = true) &&;

  /** rotate grid planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  GridderState
  rotate_grid() &;

  /** rotate grid planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  GridderState
  rotate_grid() &&;

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
class Gridder {
public:

  mutable GridderState state; /**< state maintained by instances */

  /** constructor
   *
   * @param device gridder device type
   * @param max_added_tasks maximum number of concurrent tasks (actual
   * number may be less than requested)
   * @param grid_size, in logical axis order: X, Y, stokes, cube
   * @param grid_scale, in X, Y order
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is employed,
   * but some devices support additional, concurrent tasks.
   */
  Gridder(
    Device device,
    unsigned max_added_tasks,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale)
    : state(GridderState(device, max_added_tasks, grid_size, grid_scale)) {}

  /** copy constructor
   *
   * Invokes fence() on argument.
   */
  Gridder(const volatile Gridder& other)
    : state(other.state) {}

  /** move constructor */
  Gridder(Gridder&& other)
    : state(std::move(other).state) {}

  /** copy assignment
   *
   * Invokes fence() on argument
   */
  Gridder&
  operator=(const volatile Gridder&);

  /** move assignment*/
  Gridder&
  operator=(Gridder&&);

  /** device */
  Device
  device() const noexcept {
    return state.device();
  }

  /** maximum additional tasks
   *
   * This value may differ from the value provided to the constructor, depending
   * on device limitations */
  unsigned
  max_added_tasks() const noexcept {
    return state.max_added_tasks();
  }

  /** grid size */
  const std::array<unsigned, 4>&
  grid_size() const noexcept {
    return state.grid_size();
  }

  /** grid scale */
  const std::array<grid_scale_fp, 2>&
  grid_scale() const noexcept {
    return state.grid_scale();
  }

  /** null state query */
  bool
  is_null() const noexcept {
    return state.is_null();
  }

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
  std::optional<Error>
  set_convolution_function(Device host_device, const CFArray& cf) {
    std::optional<Error> result;
    std::tie(result, const_cast<Gridder*>(this)->state) =
      std::move(const_cast<Gridder*>(this)->state)
      .set_convolution_function(host_device, cf);
    return result;
  }

  /** grid visibilities
   *
   * May invoke fence() on target.
   *
   * The indexing of visibilities and all other metadata vectors must be
   * consistent. For example the weight for the visibility value visibilities[i]
   * must be located at visibility_weights[i].
   *
   * @param host_device device to use for changing array layout
   * @param visibilities visibilities
   * @param visibility_grid_cubes visibility grid cube indexes
   * @param visibility_cf_cubes visibility convolution function cube indexes
   * @param visibility_weights visibility weights
   * @param visibility_frequencies visibility frequencies
   * @param visibility_phases visibility phase differences
   * @param visibility_coordinates visibility coordinates
   */
  void
  grid_visibilities(
    Device host_device,
    const std::vector<std::complex<visibility_fp>>& visibilities,
    const std::vector<unsigned> visibility_grid_cubes,
    const std::vector<unsigned> visibility_cf_cubes,
    const std::vector<vis_weight_fp>& visibility_weights,
    const std::vector<vis_frequency_fp>& visibility_frequencies,
    const std::vector<vis_phase_fp>& visibility_phases,
    const std::vector<vis_uvw_t>& visibility_coordinates) {

    state =
      std::move(state)
      .grid_visibilities(
        host_device,
        visibilities,
        visibility_grid_cubes,
        visibility_cf_cubes,
        visibility_weights,
        visibility_frequencies,
        visibility_phases,
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

  /** get grid plane weights
   *
   * Invokes fence() on target.
   */
  std::unique_ptr<GridWeightArray>
  grid_weights() const volatile {
    std::unique_ptr<GridWeightArray> result;
    std::tie(const_cast<Gridder*>(this)->state, result) =
      std::move(const_cast<Gridder*>(this)->state).grid_weights();
    return result;
  }

  /** get access to grid values */
  std::unique_ptr<GridValueArray>
  grid_values() const volatile {
    std::unique_ptr<GridValueArray> result;
    std::tie(const_cast<Gridder*>(this)->state, result) =
      std::move(const_cast<Gridder*>(this)->state).grid_values();
    return result;
  }

  /** reset grid values to zero
   *
   * May invoke fence() on target.
   */
  void
  reset_grid() {
    state = std::move(state).reset_grid();
  }

  /** normalize grid values by weights
   *
   * May invoke fence() on target.
   *
   * @param wgt_factor multiplicative factor applied to weights before
   * normalization
   */
  void
  normalize(grid_value_fp wgt_factor = 1) {
    state = std::move(state).normalize(wgt_factor);
  }

  /** apply FFT to grid array planes
   *
   * May invoke fence() on target.
   *
   * @param sign sign of imaginary unit in FFT kernel
   * @param in_place run FFT in-place, without allocation of another grid
   */
  std::optional<Error>
  apply_fft(FFTSign sign = fft_sign_dflt, bool in_place = true) {
    std::optional<Error> result;
    std::tie(result, const_cast<Gridder*>(this)->state) =
      std::move(const_cast<Gridder*>(this)->state).apply_fft(sign, in_place);
    return result;
  }

  /** rotate grid planes by half
   *
   * Primarily for use after application of FFT. May invoke fence() on target.
   */
  void
  rotate_grid() {
    state = std::move(state).rotate_grid();
  }
};

/** global initialization of hpg
 *
 * Function is idempotent, but should not be called by a process after a call to
 * hpg::finalize(). All objects created by hpg must only exist between an
 * initialize()/finalize() pair; in particular, any hpg object destructors must
 * be called before the call to finalize(). A common approach is to access hpg
 * within a new scope after the call to initialize():
 *     int main() {
 *       hpg::initialize();
 *       {
 *         Gridder g(...);
 *       }
 *       hpg::finalize();
 *     }
 * Another approach involves the use of a ScopeGuard instance.
 *
 * @return true, if and only if initialization succeeded
 */
bool initialize();

/** global finalization of hpg
 *
 * Function is idempotent, but should only be called by a process after a call
 * to hpg::initialize()
 */
void finalize();

/** query whether hpg has been initialized
 *
 * Note the result will remain "true" after finalization.
 */
bool
is_initialized();

/** hpg scope object
 *
 * Intended to help avoid errors caused by objects that exist after the call
 * to hpg::finalize(). For example,
 *     int main() {
 *       // Don't do this!
 *       hpg::initialize();
 *       Gridder g();
 *       hpg::finalize(); // Error! g is still in scope,
 *                        // ~Gridder is called after finalize()
 *     }
 * however, use of a ScopeGuard value as follows helps avoid this error:
 *     int main() {
 *       hpg::ScopeGuard hpg_guard;
 *       Gridder g();
 *       // OK, because g is destroyed prior to hpg_guard
 *     }
 */
struct ScopeGuard {

private:
  bool init;

public:
  ScopeGuard()
    : init(false) {
    if (!is_initialized()) {
      initialize();
      init = true;
    }
  }

  ~ScopeGuard() {
    if (is_initialized() && init)
      finalize();
  }

  ScopeGuard(const ScopeGuard&) = delete;

  ScopeGuard&
  operator=(const ScopeGuard&) = delete;
};

} // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
