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

#include "hpg.hpp"

#ifdef HPG_ENABLE_MPI

#include <mpi.h>
#include <mutex>

namespace hpg::mpi {

/** MPI error types
 */
enum class HPG_EXPORT ErrorType {
  InvalidTopology,
  InvalidCartesianRank,
  IdenticalPartitionIndex,
  InvalidPartitionIndex,
  NullCommunicator,
  InvalidPartitionSize,
  InvalidGroupSize,
  Other
};

/** error class
 */
class HPG_EXPORT Error
  : public ::hpg::Error {
private:

  ErrorType m_mpi_type;

public:

  /** error constructor */
  Error(const std::string& msg, ErrorType err = ErrorType::Other);

  Error() {}

  /** error type */
  ErrorType
    mpi_type() const;

  /** destructor */
  virtual ~Error();
};


struct InvalidTopologyError
  : public Error {

  InvalidTopologyError();
};

struct InvalidCartesianRankError
  : public Error {

  InvalidCartesianRankError();
};

struct IdenticalPartitionIndexError
  : public Error {

  IdenticalPartitionIndexError();
};

struct InvalidPartitionIndexError
  : public Error {

  InvalidPartitionIndexError();
};

struct NullCommunicatorError
  : public Error {

  NullCommunicatorError();
};

struct InvalidPartitionSizeError
  : public Error {

  InvalidPartitionSizeError();
};

struct InvalidGroupSizeError
  : public Error {

  InvalidGroupSizeError();
};

template <typename T>
constexpr MPI_Datatype
mpi_datatype() {
  return MPI_DATATYPE_NULL;
}
template <>
constexpr MPI_Datatype
mpi_datatype<char>() {
  return MPI_CHAR;
}
template <>
constexpr MPI_Datatype
mpi_datatype<signed short int>() {
  return MPI_SHORT;
}
template <>
constexpr MPI_Datatype
mpi_datatype<signed int>() {
  return MPI_INT;
};
template <>
constexpr MPI_Datatype
mpi_datatype<signed long int>() {
  return MPI_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<signed long long int>() {
  return MPI_LONG_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<signed char>() {
  return MPI_SIGNED_CHAR;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned char>() {
  return MPI_UNSIGNED_CHAR;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned short int>() {
  return MPI_UNSIGNED_SHORT;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned int>() {
  return MPI_UNSIGNED;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned long int>() {
  return MPI_UNSIGNED_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned long long int>() {
  return MPI_UNSIGNED_LONG_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<float>() {
  return MPI_FLOAT;
};
template <>
constexpr MPI_Datatype
mpi_datatype<double>() {
  return MPI_DOUBLE;
};
template <>
constexpr MPI_Datatype
mpi_datatype<long double>() {
  return MPI_LONG_DOUBLE;
};
template <>
constexpr MPI_Datatype
mpi_datatype<wchar_t>() {
  return MPI_WCHAR;
};
template <>
constexpr MPI_Datatype
mpi_datatype<bool>() {
  return MPI_CXX_BOOL;
};
template <>
constexpr MPI_Datatype
mpi_datatype<std::complex<float>>() {
  return MPI_CXX_COMPLEX;
};
template <>
constexpr MPI_Datatype
mpi_datatype<std::complex<double>>() {
  return MPI_CXX_DOUBLE_COMPLEX;
};
template <>
constexpr MPI_Datatype
mpi_datatype<std::complex<long double>>() {
  return MPI_CXX_LONG_DOUBLE_COMPLEX;
};

namespace runtime {
struct HPG_EXPORT State;
struct HPG_EXPORT GridderState;
} // end namespace runtime

class HPG_EXPORT Gridder;

class HPG_EXPORT GridderState {
protected:
  friend class Gridder;
  friend class runtime::GridderState;

  // state cannot be a unique_ptr since runtime::State is here an incomplete
  // type
  std::shared_ptr<runtime::State> impl; /**< state implementation */

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
   * @param comm MPI communicator
   * @param vis_part_index index of visibility partition axis of communicator
   * topology (-1 iff no visibility partition)
   * @param grid_part_index index of grid partition axis of communicator
   * topology (-1 iff no grid partition)
   * @param device gridder device type
   * @param max_added_tasks maximum number of additional tasks (actual number
   * may be less than requested)
   * @param visibility_batch_size batch size for number of VisData elements
   * (N.B: this value is currently a hard limit of the implementation that
   * governs the maximum number of elements in a vector of VisData elements
   * accepted by the grid_visibilities() family of methods)
   * @param max_avg_channels_per_vis maximum average (over
   * visibilities in a batch) number of channels onto which visibilities
   * are mapped
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size in logical axis order: X, Y, mrow, channel
   * @param grid_scale in X, Y order
   * @param mueller_indexes CFArray Mueller element indexes, by mrow
   * @param conjugate_mueller_indexes CFArray conjugate Mueller element indexes,
   * by mrow
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is
   * employed, but some devices support additional, concurrent tasks.
   *
   * The values of max_added_tasks, visibility_batch_size and
   * max_avg_channels_per_vis have effects on the amount of memory allocated
   * on the selected gridder device.
   *
   * @sa Gridder::Gridder()
   */
  GridderState(
    MPI_Comm comm,
    int vis_part_index,
    int grid_part_index,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
    ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    );

public:

  /** GridderState factory method
   *
   * @param comm MPI communicator with 2d topology, MPI_COMM_SELF, or
   * MPI_COMM_NULL
   * @param vis_part_index dimensional index in 2d topology of visibililty
   * partition (can be -1, if no visibility partition)
   * @param grid_part_index dimensional index in 2d topology of grid partition
   * (can be -1 if no grid partition)
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @sa GridderState()
   */
  static rval_t<GridderState>
  create(
    MPI_Comm comm,
    int vis_part_index,
    int grid_part_index,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
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
   * @param comm MPI communicator
   * @param vis_part_size size of visibility partition (i.e, number of subsets
   * in partition)
   * @param grid_part_size size of grid partition (i.e, number of subsets in
   * partition)
   *
   * Does not throw an exception if device argument names an unsupported
   * device. This function creates a new communicator from comm with a 2d
   * topology of the requested dimensions.
   */
  static rval_t<GridderState>
  create2d(
    MPI_Comm comm,
    unsigned vis_part_size,
    unsigned grid_part_size,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
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
    MPI_Comm comm,
    int vis_part_index,
    int grid_part_index,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
    ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create(
        comm,
        vis_part_index,
        grid_part_index,
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
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

  /** GridderState factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @tparam N number of polarizations in visibilities to be gridded
   */
  template <unsigned N>
  static rval_t<GridderState>
  create2d(
    MPI_Comm comm,
    unsigned vis_part_size,
    unsigned grid_part_size,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
    ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create2d(
        comm,
        vis_part_size,
        grid_part_size,
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
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
   * Deleted, to avoid shared MPI communicator
   */
  GridderState(const GridderState&) = delete;

  /** move constructor
   */
  GridderState(GridderState&&) noexcept = default;

  virtual ~GridderState();

  /** copy assignment
   *
   * Deleted, to avoid shared MPI communicator
   */
  GridderState&
  operator=(const GridderState&) = delete;

  /** move assignment
   */
  GridderState&
  operator=(GridderState&&) noexcept = default;

  /** query whether in root rank position of a grid partition subset
   */
  bool
  is_visibility_partition_root() const noexcept;

  /** query whether in root rank position of an visibility data partition subset
   */
  bool
  is_grid_partition_root() const noexcept;

  /** device */
  Device
  device() const noexcept;

  /** maximum additional tasks
   *
   * This value may differ from the value provided to the constructor, depending
   * on device limitations */
  unsigned
  max_added_tasks() const noexcept;

  /** number of visibilities passed to gridding kernel at once */
  size_t
  visibility_batch_size() const noexcept;

  /** maximum average number of channels mapped onto by visibilities */
  unsigned
  max_avg_channels_per_vis() const noexcept;

  /** grid size */
  const std::array<unsigned, 4>&
  grid_size() const noexcept;

  /** grid scale */
  std::array<grid_scale_fp, 2>
  grid_scale() const noexcept;

  unsigned
  grid_channel_offset() const noexcept;

  unsigned
  grid_channel_size() const noexcept;

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
   * @return new GridderState that has overwritten the target, but with memory
   * allocated for convolution function, or error
   */
  rval_t<GridderState>
  allocate_convolution_function_region(const CFArrayShape* shape) &&;

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
  set_model(Device host_device, GridValueArray&& gv) &&;

protected:

  // FIXME: what to do about this section???

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
   * @param update_grid_weights update grid weights or not
   * @param do_degrid do degridding
   * @param return_visibilities return residual or predicted visibilities
   * @param do_grid do gridding
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  grid_visibilities_base(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights,
    bool do_grid,
    bool return_visibilities,
    bool do_degrid) &&;

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
   * @param update_grid_weights update grid weights or not
   */
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights = true) &&;

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
   * @param update_grid_weights update grid weights or not
   */
  template <unsigned N>
  rval_t<GridderState>
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    bool update_grid_weights = true) && {

    return
      std::move(*this)
      .grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)),
        update_grid_weights);
  };

  /** degrid and grid visibilities (template-free, rvalue reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param update_grid_weights update grid weights or not
   */
  rval_t<GridderState>
  degrid_grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights = true) &&;

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
   * @param update_grid_weights update grid weights or not
   */
  template <unsigned N>
  rval_t<GridderState>
  degrid_grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    bool update_grid_weights = true) && {

    return
      std::move(*this).degrid_grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)),
        update_grid_weights);
  };

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
   * (template-free, rvalue reference version)
   *
   * May invoke fence() on target.
   *
   * @return new GridderState after gridding task has been submitted to device
   * queue, and a future of residual visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param update_grid_weights update grid weights or not
   */
  rval_t<std::tuple<GridderState, future<VisDataVector>>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights = true) &&;

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
  grid_weights() &&;

  /** get a pointer to the grid weights buffer
   *
   * WARNING: Use of this method requires great care; it's very easy to shoot
   * oneself in the foot! The returned pointer may not be dereferencable by the
   * calling process. There is no guarantee that fence() is invoked on the
   * target by this method. Recommended guidelines are to call the method
   * immediately after a call to any method that is guaranteed to fence the
   * target (fence() is a good choice), and free the pointer before calling any
   * non-const method.
   *
   * @return pointer to the current buffer of grid weights
   */
  std::shared_ptr<GridWeightArray::value_type>
  grid_weights_ptr() const &;

  /** get the number of elements in the span of the grid weights buffer
   *
   * It is recommended that this method is used to get the size of the grid
   * weights buffer (instead of using the product of the grid dimensions) to
   * account for potential padding in the buffer.
   *
   * @return the number of elements in the current buffer of grid weights
   */
  size_t
  grid_weights_span() const &;

  /** get copy of grid values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  grid_values() &&;

  /** get a pointer to the grid values buffer
   *
   * WARNING: Use of this method requires great care; it's very easy to shoot
   * oneself in the foot! The returned pointer may not be dereferencable by the
   * calling process. There is no guarantee that fence() is invoked on the
   * target by this method. Recommended guidelines are to call the method
   * immediately after a call to any method that is guaranteed to fence the
   * target (fence() is a good choice), and free the pointer before calling any
   * non-const method.
   *
   * @return pointer to the current buffer of grid values
   */
  std::shared_ptr<GridValueArray::value_type>
  grid_values_ptr() const &;

  /** get the number of elements in the span of the grid values buffer
   *
   * It is recommended that this method is used to get the size of the grid
   * values buffer (instead of using the product of the grid dimensions) to
   * account for potential padding in the buffer.
   *
   * @return the number of elements in the current buffer of grid values
   */
  size_t
  grid_values_span() const &;

  /** get copy of model values
   *
   * Invokes fence() on target.
   */
  std::tuple<GridderState, std::unique_ptr<GridValueArray>>
  model_values() &&;

  /** get a pointer to the model values buffer
   *
   * WARNING: Use of this method requires great care; it's very easy to shoot
   * oneself in the foot! The returned pointer may not be dereferencable by the
   * calling process. There is no guarantee that fence() is invoked on the
   * target by this method. Recommended guidelines are to call the method
   * immediately after a call to any method that is guaranteed to fence the
   * target (fence() is a good choice), and free the pointer before calling any
   * non-const method.
   *
   * @return pointer to the current buffer of model values
   */
  std::shared_ptr<GridValueArray::value_type>
  model_values_ptr() const &;

  /** get the number of elements in the span of the model values buffer
   *
   * It is recommended that this method is used to get the size of the model
   * values buffer (instead of using the product of the model dimensions) to
   * account for potential padding in the buffer.
   *
   * @return the number of elements in the current buffer of model values
   */
  size_t
  model_values_span() const &;

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
  reset_model() &&;

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
    bool in_place = true) &&;

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

  /** shift grid planes by half grid size on X, Y axes
   *
   * May invoke fence() on target.
   *
   * @param direction direction of shift
   */
  GridderState
  shift_grid(ShiftDirection direction) &&;

  /** shift model planes by half grid size on X, Y axes
   *
   * May invoke fence() on target.
   *
   * @param direction direction of shift
   */
  GridderState
  shift_model(ShiftDirection direction) &&;

protected:
  friend class Gridder;

  /** swap member values with another GridderState instance */
  void
  swap(GridderState& other) noexcept;

};

class HPG_EXPORT Gridder {
protected:

  mutable GridderState state; /**< state maintained by instances */

public:

  /** default constructor */
  Gridder();

protected:

  /** constructor
   *
   * @param comm MPI communicator
   * @param vis_part_index index of visibility partition axis of communicator
   * topology (-1 iff no visibility partition)
   * @param grid_part_index index of grid partition axis of communicator
   * topology (-1 iff no grid partition)
   * @param device gridder device type
   * @param max_added_tasks maximum number of concurrent tasks (actual
   * number may be less than requested)
   * @param visibility_batch_size batch size for number of VisData elements
   * (N.B: this value is currently a hard limit of the implementation that
   * governs the maximum number of elements in a vector of VisData elements
   * accepted by the grid_visibilities() family of methods)
   * @param max_avg_channels_per_vis maximum average (over
   * visibilities in a batch) number of channels onto which visibilities
   * are mapped
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size in logical axis order: X, Y, mrow, channel
   * @param grid_scale in X, Y order
   * @param mueller_indexes CFArray Mueller element indexes, by mrow
   * @param conjugate_mueller_indexes CFArray conjugate Mueller element indexes,
   * by mrow
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is employed,
   * but some devices support additional, concurrent tasks.
   *
   * The values of max_added_tasks, visibility_batch_size and
   * max_avg_channels_per_vis have effects on the amount of memory allocated
   * on the selected gridder device.
   */
  Gridder(
    MPI_Comm comm,
    int vis_part_index,
    int grid_part_index,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    IArrayVector&& mueller_indexes,
    IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
      ::hpg::GridderState::default_versions
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
    MPI_Comm comm,
    int vis_part_index,
    int grid_part_index,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
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
    MPI_Comm comm,
    int vis_part_index,
    int grid_part_index,
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
      ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create(
        comm,
        vis_part_index,
        grid_part_index,
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
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
  Gridder(const Gridder& other) = default;

  /** move constructor */
  Gridder(Gridder&& other) noexcept = default;

  /** copy assignment
   *
   * Invokes fence() on argument
   */
  Gridder&
  operator=(const Gridder&) = default;

  /** move assignment*/
  Gridder&
  operator=(Gridder&&) noexcept = default;

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

  /** number of visibilities passed to gridding kernel at once */
  size_t
  visibility_batch_size() const noexcept;

  /** maximum average number of channels mapped onto by visibilities */
  unsigned
  max_avg_channels_per_vis() const noexcept;

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
  opt_error_t
  set_model(Device host_device, GridValueArray&& gv);

  /** grid visibilities (template-free version)
   *
   * May invoke fence() on target.
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param update_grid_weights update grid weights or not
   */
  opt_error_t
  grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights = true);

  /** grid visibilities (template version)
   *
   * May invoke fence() on target.
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param update_grid_weights update grid weights or not
   */
  template <unsigned N>
  opt_error_t
  grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    bool update_grid_weights = true) {

    return
      grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)),
        update_grid_weights);
  }

  /** degrid and grid visibilities (template-free version)
   *
   * May invoke fence() on target.
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param update_grid_weights update grid weights or not
   */
  opt_error_t
  degrid_grid_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights = true);

  /** degrid and grid visibilities (template version)
   *
   * May invoke fence() on target.
   *
   * @tparam N number of polarizations in visibilities
   *
   * @param host_device device to use for copying visibilities
   * @param visibilities visibilities
   * @param update_grid_weights update grid weights or not
   */
  template <unsigned N>
  opt_error_t
  degrid_grid_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    bool update_grid_weights = true) {

    return
      degrid_grid_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)),
        update_grid_weights);
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
   * @param update_grid_weights update grid weights or not
   */
  rval_t<future<VisDataVector>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    VisDataVector&& visibilities,
    bool update_grid_weights = true);

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
   * @param update_grid_weights update grid weights or not
   */
  template <unsigned N>
  rval_t<future<std::vector<VisData<N>>>>
  degrid_grid_get_residual_visibilities(
    Device host_device,
    std::vector<VisData<N>>&& visibilities,
    bool update_grid_weights = true) {

    auto fvs_or_err =
      degrid_grid_get_residual_visibilities(
        host_device,
        VisDataVector(std::move(visibilities)),
        update_grid_weights);
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

  /** get a pointer to the grid weights buffer
   *
   * WARNING: Use of this method requires great care; it's very easy to shoot
   * oneself in the foot! The returned pointer may not be dereferencable by the
   * calling process. There is no guarantee that fence() is invoked on the
   * target by this method. Recommended guidelines are to call the method
   * immediately after a call to any method that is guaranteed to fence the
   * target (fence() is a good choice), and free the pointer before calling any
   * non-const method.
   *
   * @return pointer to the current buffer of grid weights
   */
  std::shared_ptr<GridWeightArray::value_type>
  grid_weights_ptr() const &;

  /** get the number of elements in the span of the grid weights buffer
   *
   * It is recommended that this method is used to get the size of the grid
   * values buffer (instead of using the product of the grid dimensions) to
   * account for potential padding in the buffer.
   *
   * @return the number of elements in the current buffer of grid weights
   */
  size_t
  grid_weights_span() const &;

  /** get copy of grid values
   *
   * Invokes fence() on target.
   */
  std::unique_ptr<GridValueArray>
  grid_values() const;

  /** get a pointer to the grid values buffer
   *
   * WARNING: Use of this method requires great care; it's very easy to shoot
   * oneself in the foot! The returned pointer may not be dereferencable by the
   * calling process. There is no guarantee that fence() is invoked on the
   * target by this method. Recommended guidelines are to call the method
   * immediately after a call to any method that is guaranteed to fence the
   * target (fence() is a good choice), and free the pointer before calling any
   * non-const method.
   *
   * @return pointer to the current buffer of grid values
   */
  std::shared_ptr<GridValueArray::value_type>
  grid_values_ptr() const &;

  /** get the number of elements in the span of the grid values buffer
   *
   * It is recommended that this method is used to get the size of the grid
   * values buffer (instead of using the product of the grid dimensions) to
   * account for potential padding in the buffer.
   *
   * @return the number of elements in the current buffer of grid values
   */
  size_t
  grid_values_span() const &;

  /** get copy of model values
   *
   * Invokes fence() on target.
   */
  std::unique_ptr<GridValueArray>
  model_values() const;

  /** get a pointer to the model values buffer
   *
   * WARNING: Use of this method requires great care; it's very easy to shoot
   * oneself in the foot! The returned pointer may not be dereferencable by the
   * calling process. There is no guarantee that fence() is invoked on the
   * target by this method. Recommended guidelines are to call the method
   * immediately after a call to any method that is guaranteed to fence the
   * target (fence() is a good choice), and free the pointer before calling any
   * non-const method.
   *
   * @return pointer to the current buffer of model values
   */
  std::shared_ptr<GridValueArray::value_type>
  model_values_ptr() const &;

  /** get the number of elements in the span of the model values buffer
   *
   * It is recommended that this method is used to get the size of the model
   * values buffer (instead of using the product of the model dimensions) to
   * account for potential padding in the buffer.
   *
   * @return the number of elements in the current buffer of model values
   */
  size_t
  model_values_span() const &;

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
  opt_error_t
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
  opt_error_t
  apply_model_fft(
    grid_value_fp norm = 1,
    FFTSign sign = model_fft_sign_dflt,
    bool in_place = true);

  /** shift grid planes by half grid size on X, Y axes
   *
   * May invoke fence() on target.
   *
   * @param direction direction of shift
   */
  void
  shift_grid(ShiftDirection direction);

  /** shift model planes by half grid size on X, Y axes
   *
   * May invoke fence() on target.
   *
   * @param direction direction of shift
   */
  void
  shift_model(ShiftDirection direction);

protected:

  /** move constructor */
  Gridder(GridderState&& st) noexcept;
};

} // end namespace hpg::mpi

#endif // HPG_ENABLE_MPI

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
