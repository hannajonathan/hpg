# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
from spack import *


class Hpg(CMakePackage):
    """High performance (de-)gridding kernel implementation library"""

    homepage = "https://gitlab.nrao.edu/mpokorny/hpg"
    git = "https://gitlab.nrao.edu/mpokorny/hpg.git"

    maintainers = ['mpokorny']

    version('main')

    variant('api', default='17', description='C++ standard for API',
            values=('11', '17'), multi=False)
    variant('cuda', default=False, description='Enable CUDA device')
    variant('exptl', default=False,
            description='Enable experimental kernel implementations')
    variant('max_cf_grps', default='1000',
            description='Maximum number of CF groups')
    variant('openmp', default=False, description='Enable OpenMP device')
    variant('serial', default=True, description='Enable serial device')
    variant('shared', default=True, description='Build shared libraries')

    depends_on('cmake@3.14:', type='build')

    depends_on('cuda@11.0.2:', when='+cuda')
    depends_on('fftw@3.3.8: precision=double', when='+serial')
    depends_on('fftw@3.3.8: +openmp precision=double', when='+openmp')
    depends_on('kokkos@3.2.00: std=17')
    depends_on('kokkos+cuda+cuda_lambda', when='+cuda')
    depends_on('kokkos+openmp', when='+openmp')
    depends_on('kokkos+serial', when='+serial')

    def cmake_args(self):
        args = [
            self.define('INSTALL_GTEST', False),
            self.define('BUILD_GMOCK', False),
            self.define('Hpg_BUILD_DOCS', False),
            self.define('Hpg_BUILD_TESTS', self.run_tests),
            self.define_from_variant('BUILD_SHARED_LIBS', 'shared'),
            self.define_from_variant('Hpg_ENABLE_SERIAL', 'serial'),
            self.define_from_variant('Hpg_ENABLE_OPENMP', 'openmp'),
            self.define_from_variant('Hpg_ENABLE_CUDA', 'cuda'),
            self.define_from_variant('Hpg_API', 'api'),
            self.define_from_variant('Hpg_MAX_NUM_CF_GROUPS', 'max_cf_grps'),
            self.define_from_variant(
                'Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS',
                'exptl')]
        if self.spec['kokkos'].satisfies('+wrapper'):
            args.extend([
                self.define(
                    'CMAKE_CXX_COMPILER',
                    self.spec['kokkos'].kokkos_cxx)])
        return args
