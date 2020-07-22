#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' setup.py is the main PETGEM setup script, it is based on python
setup-module.
'''


def name():
    ''' Set name for petgem package.

    :param: None
    :return: petgem name
    :rtype: string
    '''
    return 'petgem'


def description():
    ''' Descripton of PETGEM code
    '''
    petgem_description = ('Parallel python code for electromagnetic '
                          'modeling in geophysics using high-order nédélec finite elements')

    return petgem_description


def long_description():
    ''' Read long description of petgem of DESCRIPTION.rst file.

    :param: None
    :return: petgem description
    :rtype: string
    '''
    with open(os.path.join('DESCRIPTION.rst')) as f:
        return f.read()


def get_ext_modules():
    ''' Get paths of extension modules.

    :param: None
    :return: numpy path include
    '''
    try:
        import numpy
        numpy_includes = [numpy.get_include()]
    except ImportError:
        numpy_includes = []

    return numpy_includes


if __name__ == '__main__':
    from setuptools import setup
    import os
    # PETGEM setup
    setup(name=name(),
          maintainer="Octavio Castillo-Reyes",
          maintainer_email="octavio.castillo@bsc.es",
          version='0.7',
          long_description=long_description(),
          description=description(),
          url="https://www.bsc.es/castillo-octavio",
          download_url="http://petgem.bsc.es",
          license='BSD-3',
          packages=['petgem'],
          include_dirs=get_ext_modules(),
          install_requires=['petsc4py', 'numpy', 'scipy', 'singleton_decorator', 'mpi4py', 'h5py'],
          setup_requires=['sphinx'],
          classifiers=['Development Status :: 3 - Alpha',
                       'License :: Free for non-commercial use',
                       'Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'Programming Language :: Python :: 3.6',
                       'Topic :: Scientific/Engineering',
                       'Topic :: Software Development :: Libraries',
                       'Operating System :: POSIX :: Linux'],
          keywords=['3D CSEM, high-order edge finite elements, HPC, numerical simulation'],
          platforms="Linux",
          include_package_data=True,
          )

else:
    pass
