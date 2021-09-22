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
#define HPG_INTERNAL
#include "hpg.hpp"

#include "argparse.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <queue>
#include <set>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using namespace std::string_literals;
namespace K = Kokkos;

static const std::array<hpg::grid_scale_fp, 2> default_scale{0.001, -0.001};

/** fudge factor to shift the origin of the convolution functions for agreement
 * with CASA
 */
#ifndef CF_NUDGE
# define CF_NUDGE 1
#endif

/** wrapper class to allow argumentparser to return a vector from a
 * single-string-value option
 */
template <typename T>
struct argwrap {
  argwrap(const T& t)
    : val(t) {}
  argwrap(T&& t)
    : val(std::move(t)) {}
  T val;
};

/** split string into "sep"-delimited component strings, without splitting
 * within bracketed terms
 */
std::vector<std::string>
split_arg(const std::string& s, char sep = ',') {

  std::vector<std::string> result;
  unsigned depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '{') {
      ++depth;
    } else if (s[i] == '}') {
      --depth;
    } else if (depth == 0) {
      if (s[i] == sep) {
        auto t = s.substr(start, i - start);
        result.push_back(t);
        start = i + 1;
      }
    }
  }
  if (depth != 0)
    throw std::runtime_error("argument has unbalanced braces");
  auto t = s.substr(start, s.size() - start);
  result.push_back(t);
  return result;
}

/** parse a boolean argument value
 */
argwrap<std::vector<bool>>
parse_bool_args(const std::string& s) {

  std::vector<bool> result;
  for (auto& a : split_arg(s))
    result.push_back(std::min(std::stoul(a), 1ul));
  return argwrap(result);
}

/** parse an unsigned integer argument value
 */
argwrap<std::vector<unsigned>>
parse_unsigned_args(const std::string& s) {

  std::vector<unsigned> result;
  for (auto& a : split_arg(s))
    result.push_back(std::stoul(a));
  return argwrap(result);
}

/** parse a enumerated set argument value
 */
template <typename t>
argwrap<std::vector<t>>
parse_enumerated(
  const std::string& s,
  const std::string& name,
  const std::unordered_map<t, std::string>& codes,
  const std::vector<std::string>& all) {

  std::vector<std::string> args;
  if (s.find('*') != std::string::npos)
    args = all;
  else
    args = split_arg(s);
  std::vector<t> result;
  for (std::string& arg : args) {
    auto en_nm_p =
      std::find_if(codes.begin(), codes.end(),
                   [&arg](auto a_n) { return arg == a_n.second; });
    if (en_nm_p == codes.end())
      throw std::runtime_error("invalid " + name + " specification");
    result.push_back(en_nm_p->first);
  }
  return argwrap(result);
}

const std::unordered_map<hpg::Device, std::string> device_codes{
#ifdef HPG_ENABLE_SERIAL
  {hpg::Device::Serial, "serial"},
#endif
#ifdef HPG_ENABLE_OPENMP
  {hpg::Device::OpenMP, "omp"},
#endif
#ifdef HPG_ENABLE_CUDA
  {hpg::Device::Cuda, "cuda"}
#endif
};

argwrap<std::vector<hpg::Device>>
parse_devices(const std::string& s) {
  static const std::vector<std::string> all{
#ifdef HPG_ENABLE_SERIAL
    device_codes.at(hpg::Device::Serial),
#endif
#ifdef HPG_ENABLE_OPENMP
    device_codes.at(hpg::Device::OpenMP),
#endif
#ifdef HPG_ENABLE_CUDA
    device_codes.at(hpg::Device::Cuda)
#endif
  };

  return parse_enumerated(s, "device", device_codes, all);
}

enum class Op {
  Gridding,
  Degridding,
  DegriddingAndGridding
};

const std::unordered_map<Op, std::string> op_codes{
  {Op::Gridding, "grid"},
  {Op::Degridding, "degrid"},
  {Op::DegriddingAndGridding, "both"}
};

argwrap<std::vector<Op>>
parse_ops(const std::string& s) {
  static const std::vector<std::string> all{
    op_codes.at(Op::Gridding),
    op_codes.at(Op::Degridding),
    op_codes.at(Op::DegriddingAndGridding)
  };

  return parse_enumerated(s, "op", op_codes, all);
}

std::vector<std::vector<std::vector<int>>>
parse_mueller_indexes(const std::string& s) {
  std::vector<std::vector<std::vector<int>>> result;
  for (auto& a : split_arg(s)) {
    if (a.front() == 'I') {
      if (auto n = std::stoul(a.substr(1)); n == 1)
        result.push_back({{0}});
      else if (n == 2)
        result.push_back({{0, -1}, {-1, 1}});
      else if (n == 3)
        result.push_back({{0, -1, -1}, {-1, 1, -1}, {-1, -1, 2}});
      else if (n == 4)
        result.push_back({
          {0, -1, -1, -1},
          {-1, 1, -1, -1},
          {-1, -1, 2, -1},
          {-1, -1, -1, 3}});
      else
        throw std::runtime_error("invalid diagonal mueller size");
    } else if (a.front() == 'D') {
      if (auto n = std::stoul(a.substr(1)); n == 1)
        result.push_back({{0}});
      else if (n == 2)
        result.push_back({{0, 1}, {2, 3}});
      else if (n == 3)
        result.push_back({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
      else if (n == 4)
        result.push_back({
            {0, 1, 2, 3},
            {4, 5, 6, 7},
            {8, 9, 10, 11},
            {12, 13, 14, 15}});
      else
        throw std::runtime_error("invalid dense mueller size");
    } else if (a.front() == '{' && a.back() == '}') {
      std::vector<std::vector<int>> m;
      for (auto& mr : split_arg(a.substr(1, a.size() - 2))) {
        std::vector<int> mrow;
        if (mr.front() == '{' && mr.back() == '}') {
          for (auto& mc : split_arg(mr.substr(1, mr.size() - 2)))
            mrow.push_back(std::stoi(mc));
          m.push_back(std::move(mrow));
        } else {
          abort();
        }
      }
      result.push_back(std::move(m));
    } else {
      abort();
    }
  }
  return result;
}

/** parse a list unsigned-value vectors
 *
 * ex: a-b,c-d-e,f is three vectors [a, b], [c, d, e], [f]
 */
argwrap<std::vector<std::vector<unsigned>>>
parse_cfsizes(const std::string& s) {
  std::vector<std::vector<unsigned>> result;
  for (auto& g : split_arg(s)) {
    std::vector<unsigned> sizes;
    for (auto& sz : split_arg(g, '-'))
      sizes.push_back(std::stoul(sz));
    result.push_back(std::move(sizes));
  }
  return argwrap(result);
}

/** trial specification
 *
 * container for performance trial arguments, with some methods for derived
 * values and string representations
 */
struct TrialSpec {
  TrialSpec(
    unsigned index_,
    const hpg::Device& device_,
    const int& streams_,
    const Op& op_,
    bool sow_,
    const std::vector<std::vector<int>>& mueller_indexes_,
    const int& gsize_,
    const std::vector<unsigned>& cfsize_,
    const int& oversampling_,
    const int& visibilities_,
    const int& repeats_
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4> versions_
#endif
    )
    : index(index_)
    , device(device_)
    , streams(streams_)
    , op(op_)
    , sow(sow_)
    , mueller_indexes(mueller_indexes_)
    , gsize(gsize_)
    , oversampling(oversampling_)
    , visibilities(visibilities_)
    , repeats(repeats_)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , versions(versions_)
#endif
 {
   for (auto& s : cfsize_)
     cfsize.push_back(int(s));
 }

  unsigned index;
  hpg::Device device;
  Op op;
  bool sow;
  int streams;
  std::vector<std::vector<int>> mueller_indexes;
  int gsize;
  std::vector<int> cfsize;
  int oversampling;
  int visibilities;
  int repeats;
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  std::array<unsigned, 4> versions;
#endif

  // TODO: replace the pad_right() usage with io manipulators

  static const unsigned id_col_width = 10;
  static constexpr const char* id_names[]
  {"status",
   "dev",
   "str",
   "op",
   "sow",
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
   "vsn",
#endif
   "mueller",
   "grid",
   "cf",
   "osmp",
   "nvis",
   "rpt",
   "vis/s"};

  static std::string
  pad_right(const std::string& s) {
    auto col = (s.size() + id_col_width - 1) / id_col_width;
    return s + std::string(col * id_col_width - s.size(), ' ');
  }

  static std::string
  id_header() {
    std::ostringstream oss;
    unsigned i = 0;
    for (; i < sizeof(id_names) / sizeof(id_names[0]) - 1; ++i)
      oss << pad_right(id_names[i]);
    oss << id_names[i];
    return oss.str();
  }

  std::string
  mueller() const {
    std::ostringstream oss;
    bool is_diagonal = true;
    for (size_t mr = 0; is_diagonal && mr < mueller_indexes.size(); ++mr) {
      auto& mrow = mueller_indexes[mr];
      for (size_t mc = 0; is_diagonal && mc < mrow.size(); ++mc)
        is_diagonal = mrow[mc] == ((mc == mr) ? int(mr) : -1);
    }
    if (is_diagonal) {
      oss << "I" << mueller_indexes.size();
    } else {
      bool is_dense = true;
      int idx = 0;
      for (size_t mr = 0; is_dense && mr < mueller_indexes.size(); ++mr) {
        auto& mrow = mueller_indexes[mr];
        for (size_t mc = 0; is_dense && mc < mrow.size(); ++mc)
          is_dense = mrow[mc] == idx++;
      }
      is_dense =
        is_dense && mueller_indexes.size() == mueller_indexes[0].size();
      if (is_dense) {
        oss << "D" << mueller_indexes.size();
      } else {
        std::string sep = "";
        for (size_t mr = 0; mr < mueller_indexes.size(); ++mr) {
          auto& mrow = mueller_indexes[mr];
          for (size_t mc = 0; mc < mrow.size(); ++mc) {
            oss << sep << mrow[mc];
            sep = ",";
          }
          sep = ";";
        }
      }
    }
    return oss.str();
  }

  std::string
  id() const {
    std::ostringstream oss;
    std::array<char, id_col_width> nvis;
    std::snprintf(nvis.data(), nvis.size(), "%9.2g", double(visibilities));
    std::ostringstream cfsz;
    const char* sep = "";
    for (auto& s : cfsize) {
      cfsz << sep << s;
      sep = "-";
    }
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    std::ostringstream vsns;
    vsns << versions[0] << "," << versions[1] << ","
         << versions[2] << "," << versions[3];
#endif
    oss << pad_right(device_codes.at(device))
        << pad_right(std::to_string(streams))
        << pad_right(op_codes.at(op))
        << pad_right(sow ? "T" : "F")
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        << pad_right(vsns.str())
#endif
        << pad_right(mueller())
        << pad_right(std::to_string(gsize))
        << pad_right(cfsz.str())
        << pad_right(std::to_string(oversampling))
        << pad_right(nvis.data())
        << pad_right(std::to_string(repeats));
    return oss.str();
  }

  std::string
  skip(const std::string& reason) const {
    std::ostringstream oss;
    oss << pad_right("SKIP")
        << pad_right(id())
        << "'" << reason << "'";
    return oss.str();
  }

  std::string
  run(double seconds) const {

    std::ostringstream oss;
    std::array<char, id_col_width> vr;
    std::snprintf(
      vr.data(),
      vr.size(),
      "%-9.2e",
      total_visibilities() / seconds);
    oss << pad_right("RUN")
        << pad_right(id())
        << pad_right(vr.data());
    return oss.str();
  }

  double
  total_visibilities() const {
    return double(visibilities) * repeats;
  }
};

struct CFArray final
  : public hpg::CFArray {

  unsigned m_oversampling;
  std::vector<std::array<unsigned, rank - 1>> m_extents;
  std::vector<std::vector<std::complex<hpg::cf_fp>>> m_values;

  CFArray() {}

  CFArray(
    const std::vector<std::array<unsigned, rank - 1>>& sizes,
    unsigned oversampling,
    const std::vector<std::vector<std::complex<hpg::cf_fp>>>& values)
    : m_oversampling(oversampling)
    , m_values(values) {

    assert(sizes.size() == values.size());

    for (auto& sz : sizes)
      m_extents.push_back(
        {sz[0] * oversampling, sz[1] * oversampling, sz[2], sz[3]});
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return unsigned(m_extents.size());
  }

  std::array<unsigned, rank - 1>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  static_assert(
    CFArray::Axis::x == 0
    && CFArray::Axis::y == 1
    && CFArray::Axis::mueller == 2
    && CFArray::Axis::cube == 3
    && CFArray::Axis::group == 4);

  std::complex<hpg::cf_fp>
  operator()(
    unsigned x,
    unsigned y,
    unsigned mueller,
    unsigned cube,
    unsigned grp)
    const override {
    auto& vals = m_values[grp];
    auto& ext = m_extents[grp];
    return vals[((x * ext[1] + y) * ext[2] + mueller) * ext[3] + cube];
  }
};

/** representation of input data
 */
struct InputData {
  CFArray cf;
  std::array<unsigned, 4> gsize;
  int oversampling;

  hpg::VisDataVector visibilities;
  hpg::IArrayVector mueller_indexes;
  std::vector<hpg::GridValueArray::value_type> model_values;
};

template <unsigned N, typename Generator>
static void
init_visibilities(
  const std::vector<std::array<unsigned, 4>>& cf_sizes,
  int num_visibilities,
  const Generator& generator,
  InputData& input_data) {

  int num_visdata = (num_visibilities + N - 1) / N;
  std::vector<hpg::VisData<N>> visdata(num_visdata);
  auto visdata_p = visdata.data();
  auto cf_sizes_p = cf_sizes.data();
  unsigned ngrp = cf_sizes.size();

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  const double uscale = default_scale[0] * input_data.oversampling * inv_lambda;
  const double vscale = default_scale[1] * input_data.oversampling * inv_lambda;
  const auto x0 = (input_data.oversampling * input_data.gsize[0]) / 2;
  const auto y0 = (input_data.oversampling * input_data.gsize[1]) / 2;
  K::parallel_for(
    "init_vis",
    K::RangePolicy<K::OpenMP>(0, num_visdata),
    [=](const int i) {

      auto rstate = generator.get_state();
      auto grp = rstate.urand(0, ngrp);
      auto& cfsz = *(cf_sizes_p + grp);
      std::array<unsigned, 2> cf_index = {rstate.urand(0, cfsz[3]), grp};

      float ulim = (x0 - (input_data.oversampling * cfsz[0]) / 2) / uscale;
      float vlim = (y0 - (input_data.oversampling * cfsz[1]) / 2) / vscale;

      std::array<std::complex<hpg::visibility_fp>, N> visibilities;
      std::array<hpg::vis_weight_fp, N> weights;
      for (size_t i = 0; i < N; ++i) {
        visibilities[i] =
          std::complex<hpg::visibility_fp>(
            rstate.frand(-1, 1),
            rstate.frand(-1, 1));
        weights[i] = rstate.frand(0, 1);
      }
      hpg::cf_phase_gradient_t
        grad{rstate.frand(-1.0, 1.0), rstate.frand(-1.0, 1.0)};
      *(visdata_p + i) =
        hpg::VisData<N>(
          visibilities,
          weights,
          freq,
          rstate.frand(-3.14, 3.14),
          {rstate.frand(-ulim, ulim),
           rstate.frand(-vlim, vlim),
           0.0},
          rstate.urand(0, input_data.gsize[3]),
          cf_index,
          grad);

      generator.free_state(rstate);
    });

  input_data.visibilities = hpg::VisDataVector(visdata);
}

template <unsigned N>
hpg::IArrayVector
init_mueller_indexes(const std::vector<std::vector<int>>& mueller_indexes) {

  std::vector<std::array<int, N>> result;
  for (size_t mrow = 0; mrow < mueller_indexes.size(); ++mrow) {
    auto& mi_mrow = mueller_indexes[mrow];
    assert(mi_mrow.size() == N);
    std::array<int, N> mindexes;
    for (size_t mcol = 0; mcol < N; ++mcol)
      mindexes[mcol] = mi_mrow[mcol];
    result.push_back(mindexes);
  }
  return hpg::IArrayVector(result);
}

/** create visibility data
 */
template <typename Generator>
InputData
create_input_data(
  const std::vector<std::vector<int>>& mueller_indexes,
  unsigned glen,
  const std::vector<unsigned>& cflen,
  int oversampling,
  int num_visibilities,
  const Op& op,
  const Generator& generator) {

  std::array<unsigned, 4>
    gsize{glen, glen, unsigned(mueller_indexes.size()), 1};

  InputData result;
  result.gsize = gsize;
  result.oversampling = oversampling;

  int max_mindex = -1;
  for (const auto& mi_row : mueller_indexes)
    for (const auto& mi_col : mi_row)
      max_mindex = std::max(max_mindex, mi_col);
  assert(max_mindex >= 0);

  std::vector<std::array<unsigned, 4>> cf_sizes;
  std::vector<std::vector<std::complex<hpg::cf_fp>>> cf_values;
  for (auto& cfl : cflen) {
    cf_sizes.push_back({cfl, cfl, unsigned(max_mindex + 1), 1});
    cf_values.emplace_back(
      cfl * oversampling * cfl * oversampling * (max_mindex + 1));
  }

  auto const ngrp = cf_sizes.size();
  for (size_t grp = 0; grp < ngrp; ++grp) {
    auto cfs_p = cf_values[grp].data();
    K::parallel_for(
      "init_cf",
      K::RangePolicy<K::OpenMP>(0, cf_values[grp].size()),
      KOKKOS_LAMBDA(int i) {
        auto rstate = generator.get_state();
        *(cfs_p + i) =
          std::complex<hpg::cf_fp>(rstate.frand(-1, 1), rstate.frand(-1, 1));
        generator.free_state(rstate);
      });
  }
  result.cf = CFArray(cf_sizes, oversampling, cf_values);

  switch (mueller_indexes[0].size()) {
  case 1:
    init_visibilities<1>(cf_sizes, num_visibilities, generator, result);
    result.mueller_indexes = init_mueller_indexes<1>(mueller_indexes);
    break;
  case 2:
    init_visibilities<2>(cf_sizes, num_visibilities, generator, result);
    result.mueller_indexes = init_mueller_indexes<2>(mueller_indexes);
    break;
  case 3:
    init_visibilities<3>(cf_sizes, num_visibilities, generator, result);
    result.mueller_indexes = init_mueller_indexes<3>(mueller_indexes);
    break;
  case 4:
    init_visibilities<4>(cf_sizes, num_visibilities, generator, result);
    result.mueller_indexes = init_mueller_indexes<4>(mueller_indexes);
    break;
  default:
    assert(false);
    break;
  }

  if (op != Op::Gridding) {
    result.model_values.resize(
      std::accumulate(
        gsize.begin(),
        gsize.end(),
        (size_t)1,
        std::multiplies<size_t>()));
    auto vals_p = result.model_values.data();
    K::parallel_for(
      "init_model",
      K::RangePolicy<K::OpenMP>(0, result.model_values.size()),
      KOKKOS_LAMBDA(long i) {
        auto rstate = generator.get_state();
        *(vals_p + i) =
          hpg::GridValueArray::value_type(
            rstate.drand(-1, 1),
            rstate.drand(-1, 1));
        generator.free_state(rstate);
      });
  }
  return result;
}

/** gridding with hpg
 */
template <Op op, typename GridlikeFn>
void
run_hpg_trial_op(
  const TrialSpec& spec,
  const InputData& input_data,
  GridlikeFn gfn) {

  std::queue<InputData> ids;
  for (unsigned i = 0; i < spec.repeats; ++i)
    ids.push(input_data);

  std::unique_ptr<hpg::GridValueArray> model;
  if constexpr (op != Op::Gridding) {
    if (input_data.model_values.size() > 0)
      model =
        hpg::GridValueArray::copy_from(
          "model",
          spec.device,
          hpg::Device::OpenMP,
          input_data.model_values.data(),
          input_data.gsize);
  }

  auto time_trial =
    hpg::RvalM<void, hpg::GridderState>::pure(
      // create the GridderState instance
      [&]() {
        return
          hpg::GridderState::create(
            spec.device,
            spec.streams - 1,
            input_data.visibilities.size(),
            &input_data.cf,
            input_data.gsize,
            default_scale,
            hpg::IArrayVector(input_data.mueller_indexes),
            hpg::IArrayVector(input_data.mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
            , spec.versions
#endif
            );
      })
    .and_then(
      // set convolution function
      [&](hpg::GridderState&& gs) {
        return
          std::move(gs)
          .set_convolution_function(
            hpg::Device::OpenMP,
            CFArray(input_data.cf));
      })
    .and_then(
      // set visibility model
      [&](hpg::GridderState&& gs) {
        if (model)
          return
            std::move(gs)
            .set_model(hpg::Device::OpenMP, std::move(*model));
        else
          return hpg::rval(std::move(gs));
      })
    .map(
      // start timer after convolution function initialization completes
      [&](hpg::GridderState&& gs) {
        auto result = std::move(gs).fence();
        return
          std::make_tuple(std::chrono::steady_clock::now(), std::move(result));
      })
    .and_then_repeat(
      // grid visibilities a number of times, copying the start time into the
      // result tuple after each iteration
      spec.repeats,
      [&](auto&& t_gs) {
        InputData id = std::move(ids.front());
        ids.pop();
        return
          map(
            gfn(std::get<1>(std::move(t_gs)), std::move(id).visibilities),
            [&](hpg::GridderState&& gs) {
              return
                std::make_tuple(std::get<0>(std::move(t_gs)), std::move(gs));
            });
      })
    .map(
      // wait for completion, and return elapsed time
      [&](auto&& t_gs) {
        std::get<1>(std::move(t_gs)).fence();
        std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - std::get<0>(std::move(t_gs));
        return elapsed.count();
      });
  // run trial, and print result
  K::Profiling::pushRegion("Trial"s + std::to_string(spec.index));
  auto output =
    hpg::fold(
      time_trial(),
      [&spec](const double& t) {
        return spec.run(t);
      },
      [&spec](const hpg::Error& err) {
        return spec.skip(err.message());
      });
  K::Profiling::popRegion();
  std::cout << output << std::endl;
}

/** gridding with hpg
 */
template <Op op>
void
run_hpg_trial(const TrialSpec& spec, const InputData& input_data) {
  assert(false);
}

template <>
void
run_hpg_trial<Op::Gridding>(
  const TrialSpec& spec,
  const InputData& input_data) {

  run_hpg_trial_op<Op::Gridding>(
    spec,
    input_data,
    [update_grid_weights=spec.sow]
    (hpg::GridderState&& gs, hpg::VisDataVector&& vis) {
      return
        std::move(gs).grid_visibilities(
          hpg::Device::OpenMP,
          std::move(vis),
          update_grid_weights);
    });
}

template <>
void
run_hpg_trial<Op::Degridding>(
  const TrialSpec& spec,
  const InputData& input_data) {

  run_hpg_trial_op<Op::Degridding>(
    spec,
    input_data,
    [](hpg::GridderState&& gs, hpg::VisDataVector&& vis) {
      return
        map(
          std::move(gs).grid_visibilities_base(
            hpg::Device::OpenMP,
            std::move(vis),
            false,  // update_grid_weights
            true,   // do_degrid
            false,  // return_visibilities
            false), // do_grid
          [](auto&& gs_pv) {
            return std::get<0>(std::move(gs_pv));
          });
    });
}

template <>
void
run_hpg_trial<Op::DegriddingAndGridding>(
  const TrialSpec& spec,
  const InputData& input_data) {

  run_hpg_trial_op<Op::DegriddingAndGridding>(
    spec,
    input_data,
    [update_grid_weights=spec.sow]
    (hpg::GridderState&& gs, hpg::VisDataVector&& vis) {
      return
        std::move(gs).degrid_grid_visibilities(
          hpg::Device::OpenMP,
          std::move(vis),
          update_grid_weights);
    });
}

/** run gridding trials
 */
void
run_trials(
  const std::vector<std::vector<std::vector<int>>>& mueller_indexes,
  const std::vector<unsigned>& gsizes,
  const std::vector<std::vector<unsigned>>& cfsizes,
  const std::vector<unsigned>& oversamplings,
  const std::vector<unsigned>& visibilities,
  const std::vector<unsigned>& repeats,
  const std::vector<Op>& ops,
  const std::vector<bool>& update_grid_weights,
  const std::vector<hpg::Device>& devices,
  const std::vector<unsigned>& kernels,
  const std::vector<unsigned>& streams) {

  using rand_pool_type = typename K::Random_XorShift64_Pool<K::OpenMP>;

  unsigned trial_index = 0;
  std::cout << TrialSpec::id_header() << std::endl;
  for (auto& num_repeats : repeats) {
    for (auto& num_visibilities : visibilities) {
      for (auto& gsize : gsizes) {
        for (auto& mindexes : mueller_indexes) {
          for (auto& cfsize : cfsizes) {
            for (auto& oversampling : oversamplings) {
              for (auto& kernel : kernels) {
                for (auto& op : ops) {
                  for (auto sow : update_grid_weights) {
                    const auto input_data =
                      create_input_data(
                        mindexes,
                        gsize,
                        cfsize,
                        oversampling,
                        num_visibilities,
                        op,
                        rand_pool_type(348842));
                    for (auto& device : devices) {
                      for (auto& stream : streams) {
                        TrialSpec spec(
                          trial_index++,
                          device,
                          std::max(stream, 1u),
                          op,
                          sow,
                          mindexes,
                          gsize,
                          cfsize,
                          oversampling,
                          input_data.visibilities.num_elements(),
                          num_repeats
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
                          , {kernel, 0, 0, 0}
#endif
                          );
                        switch (op) {
                        case Op::Gridding:
                          run_hpg_trial<Op::Gridding>(spec, input_data);
                          break;
                        case Op::Degridding:
                          if (!sow)
                            run_hpg_trial<Op::Degridding>(spec, input_data);
                          break;
                        case Op::DegriddingAndGridding:
                          run_hpg_trial<Op::DegriddingAndGridding>(
                            spec,
                            input_data);
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

static std::string
wrap(const std::string& str) {
  if (str.size() == 0)
    return "";

  const std::string::size_type width = 72;
  std::ostringstream result;
  std::string::size_type pos = 0;
  do {
    auto substr = str.substr(pos, width);
    auto sp = substr.rfind(" ", substr.size());
    if (sp == std::string::npos || substr.size() < width)
      sp = substr.size();
    result << substr.substr(0, sp)
           << std::endl;
    pos += sp + 1;
  } while (pos < str.size());
  return result.str();
}

int
main(int argc, char* argv[]) {
  /* set up the argument parser */
  argparse::ArgumentParser args("minigridder", "0.0.1");
  args.add_description(
    wrap(
      "Measure achieved rate of visibilities gridded "s
      + "for various parameter values. "
      + "Many command line options can be expressed as comma-separated lists "
      + "to support running sweeps through different gridding trials "
      + "with a single invocation of the program. Note that options that can "
      + "take values in a small, enumerable set also accept `*` as a value "
      + "to indicate all the values in the set."));

  const unsigned default_num_vis = 1000000;
  {
    unsigned dflt = 10000;
    args
      .add_argument("-g", "--gsize")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("grid size ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    unsigned dflt = 31;
    args
      .add_argument("-c", "--cfsizes")
      .default_value(
        argwrap<std::vector<std::vector<unsigned>>>(
          std::vector<std::vector<unsigned>>{{dflt}}))
      .help("cf size ["s + std::to_string(dflt) + "]")
      .action(parse_cfsizes);
  }
  {
    unsigned dflt = 20;
    args
      .add_argument("-q", "--oversampling")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("oversampling factor ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    unsigned dflt = default_num_vis;
    args
      .add_argument("-v", "--visibilities")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("number of visibilities ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    unsigned dflt = 1;
    args
      .add_argument("-r", "--repeats")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("number of repeats ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    hpg::Device dflt = hpg::Device::OpenMP;
    args
      .add_argument("-d", "--device")
      .default_value(argwrap<std::vector<hpg::Device>>({dflt}))
      .help("device ["s + device_codes.at(dflt) + "]")
      .action(parse_devices);
  }
  {
    Op dflt = Op::Gridding;
    args
      .add_argument("-o", "--op")
      .default_value(argwrap<std::vector<Op>>({dflt}))
      .help("op (grid|degrid|both) ["s + op_codes.at(dflt) + "]")
      .action(parse_ops);
  }
  {
    bool dflt = false;
    args
      .add_argument("-w", "--weights")
      .default_value(argwrap<std::vector<bool>>({dflt}))
      .help("sum of weights (0|1) ["s + (dflt ? "1" : "0") + "]")
      .action(parse_bool_args);
  }
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  {
    unsigned dflt = 0;
    args
      .add_argument("-k", "--kernel")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("gridding kernel version ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
#endif
  {
    unsigned dflt = 0;
    args
      .add_argument("-i", "--id")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("device id ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    unsigned dflt = 4;
    args
      .add_argument("-s", "--streams")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("number of streams ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    std::string dflt = "I1";
    args
      .add_argument("-m", "--mueller")
      .default_value(dflt)
      .help("Mueller matrix indexes ["s + dflt + "]");
  }

  /* parse the command line arguments */
  try {
    args.parse_args(argc, argv);
  } catch (std::runtime_error& re) {
    std::cerr << re.what() << std::endl;
    return -1;
  }

  /* get the command line arguments */
  auto gsize = args.get<argwrap<std::vector<unsigned>>>("--gsize").val;
  auto cfsizes =
    args.get<argwrap<std::vector<std::vector<unsigned>>>>("--cfsizes").val;
  auto oversampling =
    args.get<argwrap<std::vector<unsigned>>>("--oversampling").val;
  auto visibilities =
    args.get<argwrap<std::vector<unsigned>>>("--visibilities").val;
  auto repeats = args.get<argwrap<std::vector<unsigned>>>("--repeats").val;
  auto devices =
    args.get<argwrap<std::vector<hpg::Device>>>("--device").val;
  auto ops = args.get<argwrap<std::vector<Op>>>("--op").val;
  auto wgts = args.get<argwrap<std::vector<bool>>>("--weights").val;
  auto device_ids = args.get<argwrap<std::vector<unsigned>>>("--id").val;
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  auto kernels = args.get<argwrap<std::vector<unsigned>>>("--kernel").val;
#else
  const std::vector<unsigned> kernels{0};
#endif
  auto streams = args.get<argwrap<std::vector<unsigned>>>("--streams").val;
  auto mueller_indexes =
    parse_mueller_indexes(args.get<std::string>("--mueller"));

  if (device_ids.size() != 1) {
    std::cerr << "Only a single device id may be specified" << std::endl;
    return -1;
  }

  hpg::InitArguments init_args;
  init_args.cleanup_fftw = true;
  init_args.device_id = device_ids[0];
  hpg::ScopeGuard hpg(init_args);
  if (hpg::host_devices().count(hpg::Device::OpenMP) > 0)
    run_trials(
      mueller_indexes,
      gsize,
      cfsizes,
      oversampling,
      visibilities,
      repeats,
      ops,
      wgts,
      devices,
      kernels,
      streams);
  else
    std::cerr << "OpenMP device is not enabled: no tests will be run"
              << std::endl;
  return 0;
}

// local variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// end:
