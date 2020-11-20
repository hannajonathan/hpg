#include "hpg.hpp"

#include "argparse.hpp"

/** wrapper class to allow argumentparser to return a vector from a
 * single-string-value option
 */
template <typename t>
struct argwrap {
  argwrap(const t& t)
    : val(t) {}
  t val;
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
  {hpg::Device::CUda, "cuda"}
};

argwrap<std::vector<hpg::Device>>
parse_devices(const std::string& s) {
  static const std::vector<std::string> all{
    device_codes.at(hpg::Device::Serial),
    device_codes.at(hpg::Device::OpenMP),
    device_codes.at(hpg::Device::Cuda)};

  return parse_enumerated(s, "device", device_codes, all);
}

static const constexpr char *validation_output_stdout = "-";

/** trial specification
 *
 * container for performance trial arguments, with some methods for derived
 * values and string representations
 */
struct TrialSpec {
  TrialSpec(
    const hpg::Device& device_,
    const int& gsize_,
    const int& cfsize_,
    const int& oversampling_,
    const int& visibilities_,
    const int& repeats_)
    : device(device_)
    , gsize(gsize_)
    , cfsize(cfsize_)
    , oversampling(oversampling_)
    , visibilities(visibilities_)
    , repeats(repeats_) {}

  Device device;
  int gsize;
  int cfsize;
  int oversampling;
  int visibilities;
  int repeats;

  // TODO: replace the pad_right() usage with io manipulators

  static const unsigned id_col_width = 10;
  static constexpr const char* id_names[]
  {"status",
   "dev",
   "grid",
   "cf",
   "osmp",
   "nvis",
   "rpt",
   "vis/s",
   "Gf/s",
   "GB/s"};

  static std::string
  pad_right(const std::string& s) {
    auto col = (s.size() + id_col_width - 1) / id_col_width;
    return s + std::string(col * id_col_width - s.size(), ' ');
  }

  static std::string
  id_header() {
    std::ostringstream oss;
    unsigned i = 0;
    for (; i < sizeof(id_names) / sizeof(id_names[0]) - 2; ++i)
      oss << pad_right(id_names[i]);
    oss << id_names[i];
    return oss.str();
  }

  std::string
  id() const {
    std::ostringstream oss;
    std::array<char, id_col_width - 1> buff;
    std::snprintf(buff.data(), buff.size(), "%g", double(visibilities));
    oss << pad_right(device_codes.at(device))
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
    std::array<char, id_col_width - 1> bw;
    std::snprintf(bw.data(), bw.size(), "%-9.2e", gbytes_rw() / seconds);
    std::array<char, id_col_width - 1> gf;
    std::snprintf(gf.data(), gf.size(), "%-9.2e", gflops() / seconds);
    oss << pad_right("RUN")
        << pad_right(id())
        << pad_right(vr.data())
        << pad_right(gf.data())
        << bw.data();
    return oss.str();
  }

  double
  gbytes_rw() const {
    /* Two complex read accesses (cf and grid), one complex write access (grid),
     * and a read of the fields of a Visibility per visibility per point in
     * cf. Visibility read accounted for outside of loop over cf, since that
     * reflects the current implementation. */
    return
      1.0e-9
      * repeats
      * double(visibilities)
      * (cfsize * cfsize * (sizeof(cf_t) + 2 * sizeof(gv_t))
         + sizeof(Visibility));
  }

  double
  total_visibilities() const {
    return double(visibilities) * repeats;
  }

  double
  gflops() const {
    /* One complex multiplication and one complex addition per visibility per cf
     * domain point, which is 4 fp multiplications and 4 fp additions. Note that
     * this is probably a mix of single and double precision ops */
    return
      1.0e-9
      * repeats
      * double(visibilities)
      * (cfsize * cfsize * 8);
  }
};

/** representation of input data read from a file
 */
struct InputData {
  std::unique_ptr<hpg::CFArray> cf;
  int gsize;
  int cfsize;
  int oversampling;
  float u_scale;
  float v_scale;

  std::vector<std::complex<hpg::visibility_fp>> visibilities;
  std::vector<unsigned> grid_cubes;
  std::vector<unsigned> cf_cubes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

};

/** representation of visibility data and metadata from input data file
 */
struct InputVis {
  float u;
  float v;
  float inv_lambda;
  float d_phase;
  float weight;
  cxf_t value;
};

/** representation of convolution function value from input data file
 */
struct InputCF {
  int x;
  int y;
  std::complex<hpg::cf_fp> value;
};

/**
 * send text versions of all trial visibilities, convolution functions and
 * gridded values to an ostream
 *
 * note visibilities and cf are in host memory space, and grid values can be in
 * another memory space
 *
 * @fixme has old interface
 */
template <typename GridL, typename memory_space>
void
output_validation_data(
  const TrialSpec& spec,
  const std::string& run_result,
  std::ostream& data_stream,
  const InputData& input_data,
  const const_grid_view<GridL, memory_space>& grid) {

  data_stream << run_result << std::endl;
  for (int i = 0; i < spec.visibilities; ++i) {
    GridVis<K::Serial> vis(
      input_data.visibilities(i),
      spec.gsize / 2,
      spec.oversampling,
      spec.cfsize / 2,
      input_data.u_scale * spec.oversampling,
      input_data.v_scale * spec.oversampling);
    data_stream << "V "<< vis.coarse[0] * spec.oversampling + vis.fine[0]
                << " " << vis.coarse[1] * spec.oversampling + vis.fine[1]
                << " " << vis.value.real()
                << " " << vis.value.imag()
                << std::endl;
  }
  for (int X = 0; X < spec.cfsize; ++X)
    for (int x = 0; x < spec.oversampling; ++x)
      for (int Y = 0; Y < spec.cfsize; ++Y)
        for (int y = 0; y < spec.oversampling; ++y) {
          auto& cf_val = input_data.cf(X, x, Y, y);
          data_stream << "C " << X * spec.oversampling + x
                      << " " << Y * spec.oversampling + y
                      << " " << cf_val.real()
                      << " " << cf_val.imag()
                      << std::endl;
        }
  {
    auto grid_h = K::create_mirror_view(grid);
    K::deep_copy(grid_h, grid);
    for (int X = 0; X < spec.gsize; ++X)
      for (int Y = 0; Y < spec.gsize; ++Y) {
        auto& gr_val = grid_h(X, Y);
        data_stream << "G " << X
                    << " " << Y
                    << " " << gr_val.real()
                    << " " << gr_val.imag()
                    << std::endl;
      }
  }
}

/**
 * send text versions of all trial gridded values to an ostream
 *
 * Intended for output of gridded values. Note that grid values can be in a
 * non-host memory space.
 *
 * @fixme has old interface
 */
template <typename GridL, typename memory_space>
void
output_grid(
  std::ostream& data_stream,
  const grid_view<GridL, memory_space>& grid) {

  auto grid_h = K::create_mirror_view(grid);
  K::deep_copy(grid_h, grid);
  for (int X = 0; X < grid.extent_int(0); ++X)
    for (int Y = 0; Y < grid.extent_int(1); ++Y) {
      auto& gr_val = grid_h(X, Y);
      data_stream << "G " << X
                  << " " << Y
                  << " " << std::scientific << gr_val.real()
                  << " " << std::scientific << gr_val.imag()
                  << std::endl;
    }
}

/** parse a line from input data file for visibility data
 */
InputVis
parse_input_vis(const std::string& line) {
  auto fields = split_arg(line, ' ');
  assert(false); // FIXME: don't have a format for this yet
  return InputVis{
    std::stof(fields[1]),
    std::stof(fields[2]),
    std::stof(fields[7]),
    std::stof(fields[3]),
    std::stof(fields[3]),
    cxf_t(std::stof(fields[5]), std::stof(fields[6]))};
}

/** parse a line from input data file for convolution function metadata
 */
std::tuple<int, int, int>
parse_cf_params(const std::string& line) {
  auto fields = split_arg(line, ' ');
  // array sz, oversampling, cf radius
  return
    {std::stoi(fields[1]),
     std::rint(std::stof(fields[3])),
     std::stoi(fields[5])};
}

/** parse a line from input data file for UV scale and grid origin offset
 */
std::tuple<float, float, int>
parse_uv_scale_offset(const std::string& line) {
  auto fields = split_arg(line, ' ');
  return {std::stof(fields[1]), std::stof(fields[2]), std::stoi(fields[3])};
}

/** parse a line from input data file for convolution function value
 */
InputCF
parse_cf(const std::string& line) {
  auto fields = split_arg(line, ' ');
  return InputCF{
    std::stoi(fields[1]),
    std::stoi(fields[2]),
    cxf_t(std::stof(fields[3]), std::stof(fields[4]))};
}

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

/** read input data from file
 */
InputData
read_input_file(const std::string& filename) {

  std::ifstream fs;
  fs.open(filename);
  if (!fs.is_open())
    throw std::runtime_error("File '"s + filename + "' could not be opened");
  // U scale, V scale, offset
  std::optional<std::tuple<float, float, int>> uv_scale_offset;
  // array sz, oversampling, cf radius
  std::optional<std::tuple<int, int, int>> cf_params;
  std::vector<InputVis> input_vis;
  std::vector<InputCF> input_cf;
  static const std::string uv_scale_marker("### uScale");
  static const std::string vis_hdr_marker("### U V");
  static const std::string cf_hdr_marker("### Nx");
  enum {no_section, uv_scale_section, vis_hdr_section,
        vis_section, cf_hdr_section, cf_section} current_section = no_section;
  std::string line;
  while (std::getline(fs, line)) {
    if (line.substr(0, 5) == "#### ") {
      switch (current_section) {
      case vis_hdr_section:
        current_section = vis_section;
        // intentional fall-through
      case vis_section:
        input_vis.push_back(parse_input_vis(line));
        break;
      case cf_hdr_section:
        cf_params = parse_cf_params(line);
        current_section = cf_section;
        break;
      case uv_scale_section:
        uv_scale_offset = parse_uv_scale_offset(line);
        current_section = no_section;
        break;
      case cf_section:
        input_cf.push_back(parse_cf(line));
        break;
      default:
        assert(false);
        break;
      }
    } else if (line.substr(0, uv_scale_marker.size()) == uv_scale_marker) {
      current_section = uv_scale_section;
    } else if (line.substr(0, vis_hdr_marker.size()) == vis_hdr_marker) {
      current_section = vis_hdr_section;
    } else if (line.substr(0, cf_hdr_marker.size()) == cf_hdr_marker) {
      current_section = cf_hdr_section;
    } else {
      assert(false);
    }
  }
  fs.close();

  if (!uv_scale_offset
      || !cf_params
      || input_vis.size() == 0
      || input_cf.size() == 0)
    throw std::runtime_error("Missing input parameters and/or data");

  auto& [u_scale, v_scale, uv_offset] = uv_scale_offset.value();
  auto& [cf_array_sz, oversampling, cf_radius] = cf_params.value();
  InputData result;
  result.gsize = 2 * uv_offset;
  result.cfsize = 2 * cf_radius + 1;
  result.oversampling = oversampling;
  result.u_scale = u_scale;
  result.v_scale = v_scale;

  auto grid_ul = result.gsize - result.cfsize;

  // initialize the CF array
  result.cf =
    cf_view<K::HostSpace>(
      K::ViewAllocateWithoutInitializing("cf"),
      cflayout({result.cfsize, oversampling, result.cfsize, oversampling}));
  const int cf_mid = result.cfsize / 2;
  const int cf_array_mid = cf_array_sz / 2;
  for (int X = 0; X < result.cfsize; ++X)
    for (int x = 0; x < oversampling; ++x)
      for (int Y = 0; Y < result.cfsize; ++Y)
        for (int y = 0; y < oversampling; ++y)
          result.cf(X, x, Y, y) =
            lookup_cf(input_cf, cf_array_mid, cf_mid, oversampling, X, x, Y, y);

  const float fine_u_scale = oversampling * u_scale;
  const float fine_v_scale = oversampling * v_scale;
  // filter out visibilities for which the CF footprint does not lie entirely
  // within the grid
  {
    // make value copies of structured bindings used by the following lambda, as
    // nvcc won't compile it otherwise
    int uvo = uv_offset;
    int ovs = oversampling;
    int cfr = cf_radius;
    auto removed =
      std::remove_if(
        input_vis.begin(),
        input_vis.end(),
        [&](auto& iv) {
          int u, v;
          std::tie(u, std::ignore) =
            compute_vis_coord(uvo, ovs, cfr, iv.u, iv.inv_lambda, fine_u_scale);
          std::tie(v, std::ignore) =
            compute_vis_coord(uvo, ovs, cfr, iv.v, iv.inv_lambda, fine_v_scale);
          return u < 0 || grid_ul < u || v < 0 || grid_ul < v;
        });
    input_vis.erase(removed, input_vis.end());
  }

  // initialize the visibilities array
  result.num_visibilities = static_cast<int>(input_vis.size());
  result.visibilities =
    visibilities_view<K::HostSpace>(
      K::ViewAllocateWithoutInitializing("visibilities"),
      result.num_visibilities);
  for (size_t i = 0; i < input_vis.size(); ++i) {
    const auto& i_vis = input_vis[i];
    auto& vis = result.visibilities(i);
    vis.u = i_vis.u;
    vis.v = i_vis.v;
    vis.inv_lambda = i_vis.inv_lambda;
    vis.d_phase = i_vis.d_phase;
    vis.weight = i_vis.weight;
    vis.value = i_vis.value;
  }
  return result;
}

/** create visibility data
 *
 * @fixme has old interface
 */
template <typename Generator>
InputData<K::HostSpace>
create_input_data(
  int gsize,
  int cfsize,
  int oversampling,
  int num_visibilities,
  const std::optional<std::string>& input,
  const Generator& generator) {

  InputData<K::HostSpace> result;
  if (!input) {
    result.gsize = gsize;
    result.cfsize = cfsize;
    result.oversampling = oversampling;
    result.num_visibilities = num_visibilities;
    result.u_scale = default_uv_scale;
    result.v_scale = default_uv_scale;

    result.visibilities =
      visibilities_view<K::HostSpace>(
        K::ViewAllocateWithoutInitializing("visibilities"),
        num_visibilities);
    result.cf =
      cf_view<K::HostSpace>(
        K::ViewAllocateWithoutInitializing("cf"),
        cflayout({cfsize, oversampling, cfsize, oversampling}));

    const float uv_max = ((gsize - cfsize - 1) / 2) / default_uv_scale;
    K::parallel_for(
      "init_vis",
      K::RangePolicy<K::OpenMP>(0, num_visibilities),
      KOKKOS_LAMBDA(const int i) {
        auto rstate = generator.get_state();
        auto& vis = result.visibilities(i);
        vis.u = rstate.frand(-uv_max, uv_max);
        vis.v = rstate.frand(-uv_max, uv_max);
        vis.inv_lambda = 1.0;
        vis.d_phase = 0.0;
        vis.weight = rstate.drand(0, 1);
        vis.value = vis_t(rstate.frand(-1, 1), rstate.frand(-1, 1));
        generator.free_state(rstate);
      });

    K::parallel_for(
      "init_cf",
      K::MDRangePolicy<K::Rank<4>, K::Serial>(
        {0, 0, 0, 0},
        {cfsize, oversampling, cfsize, oversampling}),
      KOKKOS_LAMBDA(int X, int x, int Y, int y) {
        auto rstate = generator.get_state();
        result.cf(X, x, Y, y) = cxf_t(rstate.drand(-1, 1), rstate.drand(-1, 1));
        generator.free_state(rstate);
      });
  } else {
    result = read_input_file(input.value(), cflayout);
  }
  return result;
}

/**
 * gridding with hpg
 */
void
run_hpg_trial(const TrialSpec& spec, const InputData& input_data) {

  switch (spec.implementation) {
  case Implementation::Serial:
#ifdef KOKKOS_ENABLE_SERIAL
    const auto& [elapsed, gridded] =
      run_kokkos_trial<GridL, K::Serial>(spec, input_data);
    auto run_result = spec.run(elapsed);
    std::cout << run_result << std::endl;
    FFT<K::Serial>::forward(gridded);
#else
    std::cout << spec.skip("device not available") << std::endl;
#endif
    break;

  case Implementation::OpenMP:
#ifdef KOKKOS_ENABLE_OPENMP
    const auto& [elapsed, gridded] =
      run_kokkos_trial<GridL, K::OpenMP>(spec, input_data);
    auto run_result = spec.run(elapsed, max_diff);
    std::cout << run_result << std::endl;
    FFT<K::Serial>::forward(gridded);
#else
    std::cout << spec.skip("device not available") << std::endl;
#endif
    break;

  case Implementation::CUDA:
#ifdef KOKKOS_ENABLE_CUDA
    // Copy input data to CUDA device. It would, of course, make more sense to
    // just create the views on the CUDA device initially, but we want to
    // support using the identical input data for multiple algorithms and
    // implementations in this program.

    // first create Views on device
    visibilities_view<K::CudaSpace> visibilities_d(
      K::ViewAllocateWithoutInitializing("visibilities_d"),
      spec.visibilities);
    cf_view<K::CudaSpace> cf_d(
      K::ViewAllocateWithoutInitializing("cf_d"),
      spec.cflayout(
        {spec.cfsize, spec.oversampling, spec.cfsize, spec.oversampling}));
    // create mirror views on host
    auto visibilities_h = K::create_mirror_view(visibilities_d);
    auto cf_h = K::create_mirror_view(cf_d);
    // copy from input views to host mirror views
    K::parallel_for(
      K::RangePolicy<K::OpenMP>(0, spec.visibilities),
      KOKKOS_LAMBDA(const int i) {
        visibilities_h(i) = input_data.visibilities(i);
      });
    K::parallel_for(
      "init_cf",
      K::MDRangePolicy<K::Rank<4>, K::OpenMP>(
        {0, 0, 0, 0},
        {spec.cfsize, spec.oversampling, spec.cfsize, spec.oversampling}),
      KOKKOS_LAMBDA(int X, int x, int Y, int y) {
        cf_h(X, x, Y, y) = input_data.cf(X, x, Y, y);
      });
    // deep_copy to device
    K::deep_copy(visibilities_d, visibilities_h);
    K::deep_copy(cf_d, cf_h);

    InputData<K::CudaSpace> input_data_d{
      visibilities_d,
      cf_d,
      input_data.gsize,
      input_data.cfsize,
      input_data.oversampling,
      input_data.num_visibilities,
      input_data.u_scale,
      input_data.v_scale};

    const auto& [elapsed, gridded] =
      run_kokkos_trial<GridL, K::Cuda>(spec, input_data_d);
    auto run_result = spec.run(elapsed, max_diff);
    std::cout << run_result << std::endl;
    FFT<K::Cuda>::forward(gridded);
#else
    std::cout << spec.skip("device not available") << std::endl;
#endif
    break;
  }
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
  const std::vector<hpg::Device>& devices) {

  using rand_pool_type = typename K::Random_XorShift64_Pool<K::OpenMP>;

  std::cout << TrialSpec::id_header() << std::endl;
  for (auto& num_repeats : repeats) {
    for (auto& num_visibilities : visibilities) {
      for (auto& gsize : gsizes) {
        for (auto& cfsize : cfsizes) {
          for (auto& oversampling : oversamplings) {
            const auto& input_data =
              create_input_data(
                gsize,
                cfsize,
                oversampling,
                num_visibilities,
                rand_pool_type(348842));
            for (auto& device : devices) {
              TrialSpec spec(
                device,
                input_data.gsize,
                input_data.cfsize,
                input_data.oversampling,
                input_data.num_visibilities,
                num_repeats);
              run_hpg_trial(spec, input_data);
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
  argparse::argumentparser args("minigridder", "0.0.1");
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
    hpt::Device dflt = hpg::Device::OpenMP;
    args
      .add_argument("-i", "--impl")
      .default_value(argwrap<std::vector<hpg::Device>>({dflt}))
      .help("device ["s + device_codes.at(dflt) + "]")
      .action(parse_devices);
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
    args.get<argwrap<std::vector<hpg::Device>>>("--impl").val;

  hpg::scopeguard hpg;

  run_trials(gsize, cfsize, oversampling, visibilities, repeats, devices);
  return 0;
}

// local variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// end:
