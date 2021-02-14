#define HPG_INTERNAL
#include "hpg.hpp"

#include "argparse.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <queue>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using namespace std::string_literals;
namespace K = Kokkos;

static const std::array<float, 2> default_scale{0.001, -0.001};

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

/** split string into "sep"-delimited component strings
 */
std::vector<std::string>
split_arg(const std::string& s, char sep = ',') {

  std::vector<std::string> result;
  size_t pos = 0;
  do {
    auto cm = s.find(sep, pos);
    if (cm != std::string::npos)
      cm -= pos;
    auto t = s.substr(pos, cm);
    result.push_back(t);
    pos += t.size() + 1;
  } while (pos < s.size());
  return result;
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
  {hpg::Device::Serial, "serial"},
  {hpg::Device::OpenMP, "omp"},
  {hpg::Device::Cuda, "cuda"}
};

argwrap<std::vector<hpg::Device>>
parse_devices(const std::string& s) {
  static const std::vector<std::string> all{
    device_codes.at(hpg::Device::Serial),
    device_codes.at(hpg::Device::OpenMP),
    device_codes.at(hpg::Device::Cuda)};

  return parse_enumerated(s, "device", device_codes, all);
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
    const hpg::Device& device_,
    const int& streams_,
    const unsigned& batch_size_,
    const int& gsize_,
    const std::vector<unsigned>& cfsize_,
    const int& oversampling_,
    bool phase_screen_,
    const int& visibilities_,
    const int& repeats_
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4> versions_
#endif
    )
    : device(device_)
    , streams(streams_)
    , batch_size(batch_size_)
    , gsize(gsize_)
    , oversampling(oversampling_)
    , phase_screen(phase_screen_)
    , visibilities(visibilities_)
    , repeats(repeats_)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , versions(versions_)
#endif
 {
   for (auto& s : cfsize_)
     cfsize.push_back(static_cast<int>(s));
 }

  hpg::Device device;
  int streams;
  unsigned batch_size;
  int gsize;
  std::vector<int> cfsize;
  int oversampling;
  bool phase_screen;
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
   "batch",
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
   "vsn",
#endif
   "grid",
   "cf",
   "osmp",
   "phscr",
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
  id() const {
    std::ostringstream oss;
    std::array<char, id_col_width - 1> nvis;
    std::snprintf(nvis.data(), nvis.size(), "%g", double(visibilities));
    std::array<char, id_col_width - 1> nbatch;
    std::snprintf(nbatch.data(), nbatch.size(), "%g", double(batch_size));
    std::ostringstream cfsz;
    const char* sep = "";
    for (auto& s : cfsize) {
      cfsz << sep << s;
      sep = "-";
    }
    std::snprintf(nbatch.data(), nbatch.size(), "%g", double(batch_size));
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    std::ostringstream vsns;
    vsns << versions[0] << "," << versions[1] << ","
         << versions[2] << "," << versions[3];
#endif
    oss << pad_right(device_codes.at(device))
        << pad_right(std::to_string(streams))
        << pad_right(nbatch.data())
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        << pad_right(vsns.str())
#endif
        << pad_right(std::to_string(gsize))
        << pad_right(cfsz.str())
        << pad_right(std::to_string(oversampling))
        << pad_right(phase_screen ? "T" : "F")
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
    std::array<char, id_col_width - 1> vr;
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
  std::vector<std::array<unsigned, 4>> m_extents;
  std::vector<std::vector<std::complex<hpg::cf_fp>>> m_values;

  CFArray() {}

  CFArray(
    const std::vector<std::array<unsigned, 4>>& sizes,
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
    return static_cast<unsigned>(m_extents.size());
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned copol, unsigned cube, unsigned grp)
    const override {
    auto& vals = m_values[grp];
    auto& ext = m_extents[grp];
    return vals[((x * ext[1] + y) * ext[2] + copol) * ext[3] + cube];
  }
};

/** representation of input data
 */
struct InputData {
  CFArray cf;
  std::array<unsigned, 4> gsize;
  int oversampling;

  std::vector<hpg::VisData<1>> visibilities;
  std::vector<hpg::vis_cf_index_t> cf_indexes;
  std::vector<hpg::cf_phase_screen_t> cf_phase_screens;
};

/** create visibility data
 */
template <typename Generator>
InputData
create_input_data(
  unsigned glen,
  unsigned polarizations,
  const std::vector<unsigned>& cflen,
  int oversampling,
  bool phase_screen,
  int num_visibilities,
  bool strictly_inner,
  const Generator& generator) {

  std::array<unsigned, 4> gsize{glen, glen, polarizations, 1};

  InputData result;
  result.gsize = gsize;
  result.oversampling = oversampling;

  std::vector<std::array<unsigned, 4>> cf_sizes;
  std::vector<std::vector<std::complex<hpg::cf_fp>>> cf_values;
  for (auto& cfl : cflen) {
    cf_sizes.push_back({cfl, cfl, polarizations, 1});
    cf_values.emplace_back(
      cfl * oversampling * cfl * oversampling * polarizations);
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

  result.visibilities.resize(num_visibilities);
  result.cf_indexes.resize(num_visibilities);
  if (phase_screen)
    result.cf_phase_screens.resize(num_visibilities);

  auto visibilities_p = result.visibilities.data();
  auto cf_indexes_p = result.cf_indexes.data();
  auto cf_phase_screens_p = result.cf_phase_screens.data();
  auto cf_sizes_p = cf_sizes.data();

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  const double uscale = default_scale[0] * oversampling * inv_lambda;
  const double vscale = default_scale[1] * oversampling * inv_lambda;
  const auto x0 = (oversampling * (gsize[0] - 2)) / 2;
  const auto y0 = (oversampling * (gsize[1] - 2)) / 2;
  K::parallel_for(
    "init_vis",
    K::RangePolicy<K::OpenMP>(0, num_visibilities),
    KOKKOS_LAMBDA(const int i) {

      auto rstate = generator.get_state();
      auto grp = rstate.urand(0, ngrp);
      auto& cfsz = *(cf_sizes_p + grp);
      *(cf_indexes_p + i) = {rstate.urand(0, cfsz[3]), grp};

      std::array<unsigned, 2> border;
      if (strictly_inner) {
        border[0] = (oversampling * cfsz[0]) / 2;
        border[1] = (oversampling * cfsz[1]) / 2;
      } else {
        border[0] = 0;
        border[1] = 0;
      }
      float ulim = (x0 - border[0]) / uscale;
      float vlim = (y0 - border[1]) / vscale;

      *(visibilities_p + i) =
        hpg::VisData<1>(
          {std::complex<hpg::visibility_fp>(
              rstate.frand(-1, 1),
              rstate.frand(-1, 1))},
          {rstate.frand(0, 1)},
          freq,
          rstate.frand(-3.14, 3.14),
          {rstate.frand(-ulim, ulim),
           rstate.frand(-vlim, vlim),
           0.0},
          rstate.urand(0, gsize[3]));
      if (phase_screen)
        *(cf_phase_screens_p + i) =
          {rstate.frand(-1.0, 1.0), rstate.frand(-1.0, 1.0)};

      generator.free_state(rstate);
    });
  return result;
}

auto
gridvis(hpg::GridderState&& gs, InputData&& id) {
  if (std::move(id).cf_phase_screens.size() > 0)
    return
      std::move(gs)
      .grid_visibilities(
        hpg::Device::OpenMP,
        std::move(id).visibilities,
        std::move(id).cf_indexes,
        std::move(id).cf_phase_screens);
  else
    return
      std::move(gs)
      .grid_visibilities(
        hpg::Device::OpenMP,
        std::move(id).visibilities,
        std::move(id).cf_indexes);
}


/** gridding with hpg
 */
void
run_hpg_trial(const TrialSpec& spec, const InputData& input_data) {

  std::queue<InputData> ids;
  for (unsigned i = 0; i <= spec.streams; ++i)
    ids.push(input_data);

  auto time_trial =
    hpg::RvalM<void, hpg::GridderState>::pure(
      // create the GridderState instance
      [&]() {
        return
          hpg::GridderState::create(
            spec.device,
            spec.streams - 1,
            spec.batch_size,
            &input_data.cf,
            input_data.gsize,
            default_scale
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
        InputData id;
        if (!ids.empty()) {
          id = std::move(ids.front());
          ids.pop();
        } else {
          id = input_data;
        }
        return
          map(
            gridvis(std::get<1>(std::move(t_gs)), std::move(id)),
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
  auto output =
    hpg::fold(
      time_trial(),
      [&spec](const double& t) {
        return spec.run(t);
      },
      [&spec](const hpg::Error& err) {
        return spec.skip(err.message());
      });
  std::cout << output << std::endl;
}

/** run gridding trials
 */
void
run_trials(
  const std::vector<unsigned>& gsizes,
  const std::vector<unsigned>& polarizations,
  const std::vector<std::vector<unsigned>>& cfsizes,
  const std::vector<unsigned>& oversamplings,
  const std::vector<unsigned>& visibilities,
  const std::vector<unsigned>& repeats,
  const std::vector<hpg::Device>& devices,
  const std::vector<unsigned>& kernels,
  const std::vector<unsigned>& streams,
  const std::vector<unsigned>& batch,
  bool phase_screen) {

  using rand_pool_type = typename K::Random_XorShift64_Pool<K::OpenMP>;

  std::cout << TrialSpec::id_header() << std::endl;
  for (auto& num_repeats : repeats) {
    for (auto& num_visibilities : visibilities) {
      for (auto& gsize : gsizes) {
        for (auto& num_polarizations : polarizations) {
          for (auto& cfsize : cfsizes) {
            for (auto& oversampling : oversamplings) {
              for (auto& kernel : kernels) {
                const auto input_data =
                  create_input_data(
                    gsize,
                    num_polarizations,
                    cfsize,
                    oversampling,
                    phase_screen,
                    num_visibilities,
                    kernel == 1,
                    rand_pool_type(348842));
                for (auto& device : devices) {
                  for (auto& stream : streams) {
                    for (auto& bsz : batch){
                      TrialSpec spec(
                        device,
                        std::max(stream, 1u),
                        std::max(bsz, 1u),
                        gsize,
                        cfsize,
                        oversampling,
                        phase_screen,
                        num_visibilities,
                        num_repeats
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
                        , {kernel, 0, 0, 0}
#endif
                        );
                      run_hpg_trial(spec, input_data);
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

int
main(int argc, char* argv[]) {
  /* set up the argument parser */
  argparse::ArgumentParser args("minigridder", "0.0.1");
  args.add_description(
    "measure achieved rate of visibilities gridded "s
    + "for various parameter values. "
    + "many command line options can be expressed as comma-separated lists "
    + "to support running sweeps through different gridding trials "
    + "with a single invocation of the program. note that options that can "
    + "take values in a small, enumerable set also accept `*` as a value "
    + "to indicate all the values in the set.");

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
    unsigned dflt = 0;
    args
      .add_argument("-k", "--kernel")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("gridding kernel version ["s + std::to_string(dflt) + "]")
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
    unsigned dflt = default_num_vis;
    args
      .add_argument("-b", "--batch")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("visibility batch size ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    unsigned dflt = 1;
    args
      .add_argument("-p", "--polarizations")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help(
        "number of image polarizations ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  args
    .add_argument("-f", "--phasescreen")
    .default_value(false)
    .implicit_value(true)
    .help("apply phase gradient to CF");

  /* parse the command line arguments */
  try {
    args.parse_args(argc, argv);
  } catch (std::runtime_error& re) {
    std::cerr << re.what() << std::endl;
    return -1;
  }

  /* get the command line arguments */
  auto gsize = args.get<argwrap<std::vector<unsigned>>>("--gsize").val;
  auto polarizations =
    args.get<argwrap<std::vector<unsigned>>>("--polarizations").val;
  auto cfsizes =
    args.get<argwrap<std::vector<std::vector<unsigned>>>>("--cfsizes").val;
  auto oversampling =
    args.get<argwrap<std::vector<unsigned>>>("--oversampling").val;
  auto visibilities =
    args.get<argwrap<std::vector<unsigned>>>("--visibilities").val;
  auto repeats = args.get<argwrap<std::vector<unsigned>>>("--repeats").val;
  auto devices =
    args.get<argwrap<std::vector<hpg::Device>>>("--device").val;
  auto kernels = args.get<argwrap<std::vector<unsigned>>>("--kernel").val;
  auto streams = args.get<argwrap<std::vector<unsigned>>>("--streams").val;
  auto batch = args.get<argwrap<std::vector<unsigned>>>("--batch").val;
  auto phase_screen = args.get<bool>("--phasescreen");

  hpg::ScopeGuard hpg;
  if (hpg::host_devices().count(hpg::Device::OpenMP) > 0)
    run_trials(
      gsize,
      polarizations,
      cfsizes,
      oversampling,
      visibilities,
      repeats,
      devices,
      kernels,
      streams,
      batch,
      phase_screen);
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
