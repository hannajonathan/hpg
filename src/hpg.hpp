#pragma once

#include "hpg_config.hpp"

#include <complex>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#if (HPG_API >= 17)
# include <optional>
# include <variant>
#endif

#include "hpg_export.h"

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
is_initialized() noexcept;

/** backend device type
 */
enum class HPG_EXPORT Device {
  Serial, /**< serial device */
  OpenMP, /**< OpenMP device */
  Cuda, /**< CUDA device */
  HPX, /**< HIP device */
};

/** query supported devices */
const std::set<Device>&
devices() noexcept;

/** query support host devices */
const std::set<Device>&
host_devices() noexcept;

/** error types
 */
enum class HPG_EXPORT ErrorType {
  DisabledDevice,
  DisabledHostDevice,
  IncompatibleVisVectorLengths,
  Other
};

class HPG_EXPORT Error {
private:

  ErrorType m_type;

  std::string m_msg;

public:

  Error(const std::string& msg, ErrorType err = ErrorType::Other);

  const std::string&
  message() const;

  ErrorType
  type() const;

  virtual ~Error();
};

/** type for containing a value or an error
 *
 * normally appears as a return value type for a method or function that can
 * fail
 */
#if HPG_API >= 17
template <typename T>
using rval_t = std::variant<Error, T>;
#else // HPG_API < 17
template <typename T>
using rval_t = std::tuple<std::unique_ptr<Error>, T>;
#endif // HPG_API >= 17

/** query whether rval_t value contains an error */
template <typename T>
HPG_EXPORT inline bool
is_error(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<Error>(rv);
#else // HPG_API < 17
  return bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** type trait to get value type from rval_t type */
template <typename T>
struct HPG_EXPORT rval_value {
  using type = void;
};
template <typename T>
struct rval_value<rval_t<T>> {
  using type = T;
};

/** query whether rval_t value contains a (non-error) value */
template <typename T>
HPG_EXPORT inline bool
is_value(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<T>(rv);
#else // HPG_API < 17
  return !bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** get value from an rval_t value */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_value(RV&& rv) {
  return std::get<1>(std::forward<RV>(rv));
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_value(const RV& rv) {
  return std::get<1>(rv);
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_value(RV&& rv) {
  return std::get<1>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** get error from an rval_t value */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_error(RV&& rv) {
# if HPG_API >= 17
  return std::get<0>(std::forward<RV>(rv));
# else
  return *std::get<0>(std::forward<RV>(rv));
# endif
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_error(const RV& rv) {
# if HPG_API >= 17
#  error "Unsupported c++ standard and HPG API version"
# else // HPG_API < 17
  return *std::get<0>(rv);
# endif // HPG_API >= 17
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_error(RV&& rv) {
  return *std::get<0>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** create an rval_t value from a (non-error) value */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(T&& t) {
#if HPG_API >= 17
  return rval_t<T>(std::forward<T>(t));
#else // HPG_API < 17
  return {std::unique_ptr<Error>(), std::forward<T>(t)};
#endif // HPG_API >= 17
}

/** create an rval_t value from an Error
 */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(const Error& err) {
#if HPG_API >= 17
  return rval_t<T>(err);
#else // HPG_API < 17
  return {std::unique_ptr<Error>(new Error(err)), T()};
#endif // HPG_API >= 17
}

#if __cplusplus >= 201402L
/** apply function that returns a plain value to an rval_t */
template <typename RV, typename F>
HPG_EXPORT auto
map(RV&& rv, F f) {
#if HPG_API >= 17
  using T = std::invoke_result_t<F, typename rval_value<RV>::type>;
#else // HPG_API < 17
  using T = typename std::result_of<F(typename rval_value<RV>::type)>::type;
#endif // HPG_API >= 17
  if (is_value(rv))
    return rval<T>(f(get_value(std::forward<RV>(rv))));
  else
    return rval<T>(get_error(std::forward<RV>(rv)));
}

/** apply function that returns an rval_t value to an rval_t */
template <typename RV, typename F>
HPG_EXPORT auto
flatmap(RV&& rv, F f) {
#if HPG_API >= 17
  using T =
    typename
    rval_value<std::invoke_result_t<F, typename rval_value<RV>::type>>::type;
#else // HPG_API < 17
  using T =
    typename rval_value<
      typename std::result_of<F(typename rval_value<RV>::type)>::type>::type;
#endif // HPG_API >= 17

  if (is_value(rv))
    return f(get_value(std::forward<RV>(rv)));
  else
    return rval<T>(get_error(std::forward<RV>(rv)));
}

/** apply function depending on contained value type with common result type */
template <typename RV, typename ValF, typename ErrF>
HPG_EXPORT auto
fold(RV&& rv, ValF vf, ErrF ef) {
#if HPG_API >= 17
  static_assert(
    std::is_same_v<
      std::invoke_result_t<ValF, typename rval_value<RV>::type>,
      std::invoke_result_t<ErrF, Error>>);
#else // hpg_api < 17
  static_assert(
    std::is_same<
      typename std::result_of<ValF(typename rval_value<RV>::type)>::type,
      typename std::result_of<ErrF(Error)>::type>::value);
#endif // hpg_api >= 17

  if (is_value(rv))
    return vf(get_value(std::forward<RV>(rv)));
  else
    return ef(get_error(std::forward<RV>(rv)));
}
#endif // __cplusplus >= 201402L

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

// vis_uvw_t can be any type that supports std::get<N>() for element access
using vis_uvw_t = std::array<vis_uvw_fp, 3>;

namespace Impl {
struct HPG_EXPORT State;
struct HPG_EXPORT GridderState;
} // end namespace Impl

/** base class for convolution functions */
class HPG_EXPORT CFArray {
public:

  static constexpr unsigned rank = 5;

  using scalar_type = std::complex<cf_fp>;

  virtual unsigned
  oversampling() const = 0;

  virtual unsigned
  num_supports() const = 0;

  virtual std::array<unsigned, 4>
  extents(unsigned supp) const = 0;

  virtual std::complex<cf_fp>
  operator()(unsigned x, unsigned y, unsigned sto, unsigned cube, unsigned supp)
    const = 0;

  virtual ~CFArray() {}
};

/** wrapper for read-only access to grid values
 *
 * @todo: replace with mdspan?
 */
class HPG_EXPORT GridValueArray {
public:

  static constexpr unsigned rank = 4;

  using scalar_type = std::complex<grid_value_fp>;

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

  static constexpr unsigned rank = 2;

  using scalar_type = grid_value_fp;

  virtual unsigned
  extent(unsigned dim) const = 0;

  virtual grid_value_fp
  operator()(unsigned stokes, unsigned cube) const = 0;

  virtual ~GridWeightArray() {}
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
 * convolution function array. (Many of these methods are const volatile on the
 * target since a fence operation may be involved.) To avoid the copy, the
 * following pattern can be used instead:
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
   * @param grid_size, in logical axis order: X, Y, stokes, cube
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
  GridderState(const volatile GridderState&);

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

  /** maximum number of visibilities passed to gridding kernel at once */
  size_t
  max_visibility_batch_size() const noexcept;

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
   *
   * @todo const?
   */
  rval_t<GridderState>
  set_convolution_function(Device host_device, CFArray&& cf)
    const volatile &;

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
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<std::complex<visibility_fp>>&& visibilities,
    std::vector<unsigned>&& visibility_grid_cubes,
    std::vector<unsigned>&& visibility_cf_cubes,
    std::vector<vis_weight_fp>&& visibility_weights,
    std::vector<vis_frequency_fp>&& visibility_frequencies,
    std::vector<vis_phase_fp>&& visibility_phases,
    std::vector<vis_uvw_t>&& visibility_coordinates) const volatile &;

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
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<std::complex<visibility_fp>>&& visibilities,
    std::vector<unsigned>&& visibility_grid_cubes,
    std::vector<unsigned>&& visibility_cf_cubes,
    std::vector<vis_weight_fp>&& visibility_weights,
    std::vector<vis_frequency_fp>&& visibility_frequencies,
    std::vector<vis_phase_fp>&& visibility_phases,
    std::vector<vis_uvw_t>&& visibility_coordinates) &&;

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
  reset_grid() const volatile &;

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
  normalize(grid_value_fp wgt_factor = 1) const volatile &;

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
  apply_fft(FFTSign sign = fft_sign_dflt, bool in_place = true)
    const volatile &;

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
  shift_grid() const volatile &;

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
   * @param grid_size, in logical axis order: X, Y, stokes, cube
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
  Gridder(const volatile Gridder& other);

  /** move constructor */
  Gridder(Gridder&& other);

  /** copy assignment
   *
   * Invokes fence() on argument
   */
  Gridder&
  operator=(const volatile Gridder&);

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
  const std::array<grid_scale_fp, 2>&
  grid_scale() const noexcept;

  /** null state query */
  bool
  is_null() const noexcept;

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
#if HPG_API >= 17
  std::optional<Error>
#else // HPG_API < 17
  std::unique_ptr<Error>
#endif //HPG_API >= 17
  set_convolution_function(Device host_device, CFArray&&);

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
#if HPG_API >= 17
  std::optional<Error>
#else // HPG_API < 17
  std::unique_ptr<Error>
#endif // HPG_API >= 17
  grid_visibilities(
    Device host_device,
    std::vector<std::complex<visibility_fp>>&& visibilities,
    std::vector<unsigned>&& visibility_grid_cubes,
    std::vector<unsigned>&& visibility_cf_cubes,
    std::vector<vis_weight_fp>&& visibility_weights,
    std::vector<vis_frequency_fp>&& visibility_frequencies,
    std::vector<vis_phase_fp>&& visibility_phases,
    std::vector<vis_uvw_t>&& visibility_coordinates);

  /** device execution fence
   *
   * Returns after all tasks on device have completed. Call is rarely explicitly
   * required by users.
   */
  void
  fence() const volatile;

  /** get grid plane weights
   *
   * Invokes fence() on target.
   */
  std::unique_ptr<GridWeightArray>
  grid_weights() const volatile;

  /** get access to grid values */
  std::unique_ptr<GridValueArray>
  grid_values() const volatile;

  /** reset grid values to zero
   *
   * May invoke fence() on target.
   */
  void
  reset_grid();

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
#if HPG_API >= 17
  std::optional<Error>
#else // HPG_API < 17
  std::unique_ptr<Error>
#endif //HPG_API >= 17
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
