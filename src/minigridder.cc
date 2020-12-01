#include "hpg.hpp"

#include "argparse.hpp"

#include <cassert>
#include <chrono>

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

/** monadic class of functions with domain A and range rval_t<B> */
template <typename A, typename B>
struct RvalM;

/** subclass of RvalM<A> with explicit function type */
template <typename A, typename B, typename F>
struct RvalMF;

template <typename A, typename B>
struct RvalM {

  /** construct an RvalM value as RvalMF<A,F> value with deduced function
   * type */
  template <typename F>
  static RvalMF<A, B, F>
  pure(F&& f) {
    return RvalMF<A, B, F>(std::forward<F>(f));
  }
};

template <typename A, typename B, typename F>
struct RvalMF :
  public RvalM<A, B> {

  static_assert(
    std::is_same_v<
      B,
      typename hpg::rval_value<std::invoke_result_t<F, A>>::type>
    );

  template <typename G>
  using gv_t = typename hpg::rval_value<std::invoke_result_t<G, B>>::type;

  F m_f;

  RvalMF(const F& f)
    : m_f(f) {}

  RvalMF(F&& f)
    : m_f(std::move(f)) {}

  /** apply contained function to a value */
  hpg::rval_t<B>
  operator()(A&& a) const {
    return m_f(std::forward<A>(a));
  }

  /** composition with rval_t-valued function */
  template <typename G>
  auto
  and_then(const G& g) const {

    return
      RvalM<A, gv_t<G>>::pure(
        [g, f=m_f](A&& a) { return hpg::flatmap(f(std::forward<A>(a)), g); });
  }

  /** composition with sequential iterations of rval_t-valued function */
  template <typename G>
  auto
  and_then_loop(unsigned n, const G& g) const {

    return
      RvalM<A, gv_t<G>>::pure(
        [n, g, f=m_f](A&& a) {
          auto result = f(std::forward<A>(a));
          for (unsigned i = 0; i < n; ++i)
            result = hpg::flatmap(std::move(result), g);
          return result;
        });
  }

  /** composition with simple-valued function  */
  template <typename G>
  auto
  map(const G& g) const {

    return
      RvalM<A, std::invoke_result_t<G, B>>::pure(
        [g, f=m_f](A&& a) { return hpg::map(f(std::forward<A>(a)), g); });
  }
};

template <typename B, typename F>
struct RvalMF<void, B, F> :
  public RvalM<void, B> {

  static_assert(
    std::is_same_v<
    B,
    typename hpg::rval_value<std::invoke_result_t<F>>::type>
    );

  template <typename G>
  using gv_t = typename hpg::rval_value<std::invoke_result_t<G, B>>::type;

  F m_f;

  RvalMF(const F& f)
    : m_f(f) {}

  RvalMF(F&& f)
    : m_f(std::move(f)) {}

  /** apply contained function to a value */
  hpg::rval_t<B>
  operator()() const {
    return m_f();
  }

  /** composition with rval_t-valued function */
  template <typename G>
  auto
  and_then(const G& g) const {

    return RvalM<void, gv_t<G>>::pure(
      [g, f=m_f]() { return hpg::flatmap(f(), g); });
  }

  /** composition with sequential iterations of rval_t-valued function */
  template <typename G>
  auto
  and_then_loop(unsigned n, const G& g) const {

    return
      RvalM<void, gv_t<G>>::pure(
        [n, g, f=m_f]() {
          auto result = f();
          for (unsigned i = 0; i < n; ++i)
            result = hpg::flatmap(std::move(result), g);
          return result;
        });
  }

  /** composition with simple-valued function  */
  template <typename G>
  auto
  map(const G& g) const {

    return RvalM<void, std::invoke_result_t<G, B>>::pure(
      [g, f=m_f]() { return hpg::map(f(), g); });
  }
};

/** wrapper class to allow argumentparser to return a vector from a
 * single-string-value option
 */
template <typename T>
struct argwrap {
  argwrap(const T& t)
    : val(t) {}
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

/** trial specification
 *
 * container for performance trial arguments, with some methods for derived
 * values and string representations
 */
struct TrialSpec {
  TrialSpec(
    const hpg::Device& device_,
    const int& streams_,
    const int& gsize_,
    const int& cfsize_,
    const int& oversampling_,
    const int& visibilities_,
    const int& repeats_
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4> versions_
#endif
    )
    : device(device_)
    , streams(streams_)
    , gsize(gsize_)
    , cfsize(cfsize_)
    , oversampling(oversampling_)
    , visibilities(visibilities_)
    , repeats(repeats_)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , versions(versions_)
#endif
 {}

  hpg::Device device;
  int streams;
  int gsize;
  int cfsize;
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
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
   "vsn",
#endif
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
  id() const {
    std::ostringstream oss;
    std::array<char, id_col_width - 1> buff;
    std::snprintf(buff.data(), buff.size(), "%g", double(visibilities));
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    std::ostringstream vsns;
    vsns << versions[0] << "," << versions[1] << ","
         << versions[2] << "," << versions[3];
#endif
    oss << pad_right(device_codes.at(device))
        << pad_right(std::to_string(streams))
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        << pad_right(vsns.str())
#endif
        << pad_right(std::to_string(gsize))
        << pad_right(std::to_string(cfsize))
        << pad_right(std::to_string(oversampling))
        << pad_right(buff.data())
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
  std::vector<std::complex<hpg::cf_fp>> m_values;
  std::array<unsigned, 4> m_extent;

  CFArray() {}

  CFArray(
    const std::array<unsigned, 4>& size,
    unsigned oversampling,
    const std::vector<std::complex<hpg::cf_fp>>& values)
    : m_oversampling(oversampling)
    , m_values(values) {

    m_extent[0] = size[0] * m_oversampling;
    m_extent[1] = size[1] * m_oversampling;
    m_extent[2] = size[2];
    m_extent[3] = size[3];
    assert(
      values.size() == m_extent[0] * m_extent[1] * m_extent[2] * m_extent[3]);
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  extent(unsigned dim) const override {
    return m_extent[dim];
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned stokes, unsigned cube)
    const override {
    return
      m_values[
        ((x * m_extent[1] + y) * m_extent[2] + stokes) * m_extent[3] + cube];
  }
};

/** representation of input data
 */
struct InputData {
  CFArray cf;
  std::array<unsigned, 4> gsize;
  int oversampling;

  std::vector<std::complex<hpg::visibility_fp>> visibilities;
  std::vector<unsigned> grid_cubes;
  std::vector<unsigned> cf_cubes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

};

/** create visibility data
 */
template <typename Generator>
InputData
create_input_data(
  unsigned glen,
  unsigned cflen,
  int oversampling,
  int num_visibilities,
  const Generator& generator) {

  std::array<unsigned, 4> gsize{glen, glen, 1, 1};
  std::array<unsigned, 4> cfsize{cflen, cflen, 1, 1};

  InputData result;
  result.gsize = gsize;
  result.oversampling = oversampling;

  std::vector<std::complex<hpg::cf_fp>> cf_values;
  cf_values.resize(
    cfsize[0] * oversampling * cfsize[1] * oversampling
    * cfsize[2] * cfsize[3]);

  result.visibilities.resize(num_visibilities);
  result.grid_cubes.resize(num_visibilities);
  result.cf_cubes.resize(num_visibilities);
  result.weights.resize(num_visibilities);
  result.frequencies.resize(num_visibilities);
  result.phases.resize(num_visibilities);
  result.coordinates.resize(num_visibilities);

  auto visibilities_p = result.visibilities.data();
  auto grid_cubes_p = result.grid_cubes.data();
  auto cf_cubes_p = result.cf_cubes.data();
  auto weights_p = result.weights.data();
  auto frequencies_p = result.frequencies.data();
  auto phases_p = result.phases.data();
  auto coordinates_p = result.coordinates.data();
  auto cfs_p = cf_values.data();

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  float ulim =
    ((oversampling * (gsize[0] - 2)) / 2 - (oversampling * cfsize[0]) / 2)
    / (default_scale[0] * oversampling * inv_lambda);
  float vlim =
    ((oversampling * (gsize[1] - 2)) / 2 - (oversampling * cfsize[1]) / 2)
    / (default_scale[1] * oversampling * inv_lambda);

  K::parallel_for(
    "init_vis",
    K::RangePolicy<K::OpenMP>(0, num_visibilities),
    KOKKOS_LAMBDA(const int i) {
      auto rstate = generator.get_state();
      *(visibilities_p + i) =
        std::complex<hpg::visibility_fp>(
          rstate.frand(-1, 1),
          rstate.frand(-1, 1));
      *(grid_cubes_p +i) = rstate.urand(0, gsize[3]);
      *(cf_cubes_p + i) = rstate.urand(0, cfsize[3]);
      *(weights_p + i) = rstate.frand(0, 1);
      *(frequencies_p + i) = freq;
      *(phases_p + i) = rstate.frand(-3.14, 3.14);
      *(coordinates_p + i) = {
        rstate.frand(-ulim, ulim),
        rstate.frand(-vlim, vlim),
        0.0};
      generator.free_state(rstate);
    });

  K::parallel_for(
    "init_cf",
    K::RangePolicy<K::OpenMP>(0, cf_values.size()),
    KOKKOS_LAMBDA(int i) {
      auto rstate = generator.get_state();
      *(cfs_p + i) =
        std::complex<hpg::cf_fp>(rstate.frand(-1, 1), rstate.frand(-1, 1));
      generator.free_state(rstate);
    });
  result.cf = CFArray(cfsize, oversampling, cf_values);
  return result;
}

/** gridding with hpg
 */
void
run_hpg_trial(const TrialSpec& spec, const InputData& input_data) {

  auto time_trial =
    RvalM<void, hpg::GridderState>::pure(
      // create the GridderState instance
      [&]() {
        return
          hpg::GridderState::create(
            spec.device,
            spec.streams - 1,
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
          std::make_tuple(
            std::chrono::steady_clock::now(),
            std::move(result));
      })
    .and_then_loop(
      // grid visibilities a number of times, copying the start time into the
      // result tuple after each iteration
      spec.repeats,
      [&](auto&& t_gs) {
        return
          map(
            std::get<1>(std::move(t_gs))
            .grid_visibilities(
              hpg::Device::OpenMP,
              decltype(input_data.visibilities)(input_data.visibilities),
              decltype(input_data.grid_cubes)(input_data.grid_cubes),
              decltype(input_data.cf_cubes)(input_data.cf_cubes),
              decltype(input_data.weights)(input_data.weights),
              decltype(input_data.frequencies)(input_data.frequencies),
              decltype(input_data.phases)(input_data.phases),
              decltype(input_data.coordinates)(input_data.coordinates)),
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
  const std::vector<unsigned>& cfsizes,
  const std::vector<unsigned>& oversamplings,
  const std::vector<unsigned>& visibilities,
  const std::vector<unsigned>& repeats,
  const std::vector<hpg::Device>& devices,
  const std::vector<unsigned>& kernels,
  const std::vector<unsigned>& streams) {

  using rand_pool_type = typename K::Random_XorShift64_Pool<K::OpenMP>;

  std::cout << TrialSpec::id_header() << std::endl;
  for (auto& num_repeats : repeats) {
    for (auto& num_visibilities : visibilities) {
      for (auto& gsize : gsizes) {
        for (auto& cfsize : cfsizes) {
          for (auto& oversampling : oversamplings) {
            const auto input_data =
              create_input_data(
                gsize,
                cfsize,
                oversampling,
                num_visibilities,
                rand_pool_type(348842));
            for (auto& device : devices) {
              for (auto& kernel : kernels) {
                for (auto& stream : streams) {
                  TrialSpec spec(
                    device,
                    stream,
                    gsize,
                    cfsize,
                    oversampling,
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

int
main(int argc, char* argv[]) {
  /* set up the argument parser */
  argparse::ArgumentParser args("minigridder", "0.0.1");
  args.add_description(
    "measure achieved memory bandwidth and rate of visibilities gridded "s
    + "for various parameter values. "
    + "many command line options can be expressed as comma-separated lists "
    + "to support running sweeps through different gridding trials "
    + "with a single invocation of the program. note that options that can "
    + "take values in a small, enumerable set also accept `*` as a value "
    + "to indicate all the values in the set.");

  {
    unsigned dflt = 1000;
    args
      .add_argument("-g", "--gsize")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("grid size ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
  }
  {
    unsigned dflt = 31;
    args
      .add_argument("-c", "--cfsize")
      .default_value(argwrap<std::vector<unsigned>>({dflt}))
      .help("cf size ["s + std::to_string(dflt) + "]")
      .action(parse_unsigned_args);
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
    unsigned dflt = 1000000;
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

  /* parse the command line arguments */
  try {
    args.parse_args(argc, argv);
  } catch (std::runtime_error& re) {
    std::cerr << re.what() << std::endl;
    return -1;
  }

  /* get the command line arguments */
  auto gsize = args.get<argwrap<std::vector<unsigned>>>("--gsize").val;
  auto cfsize = args.get<argwrap<std::vector<unsigned>>>("--cfsize").val;
  auto oversampling =
    args.get<argwrap<std::vector<unsigned>>>("--oversampling").val;
  auto visibilities =
    args.get<argwrap<std::vector<unsigned>>>("--visibilities").val;
  auto repeats = args.get<argwrap<std::vector<unsigned>>>("--repeats").val;
  auto devices =
    args.get<argwrap<std::vector<hpg::Device>>>("--device").val;
  auto kernels = args.get<argwrap<std::vector<unsigned>>>("--kernel").val;
  auto streams = args.get<argwrap<std::vector<unsigned>>>("--streams").val;

  hpg::ScopeGuard hpg;
  if (hpg::host_devices().count(hpg::Device::OpenMP) > 0)
    run_trials(
      gsize,
      cfsize,
      oversampling,
      visibilities,
      repeats,
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
