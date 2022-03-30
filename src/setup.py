# CFLAGS=-stdlib=libc++ python3 setup.py build

from distutils.core import setup, Extension
import numpy as np

cpp_args = ['-I/usr/local/Cellar/boost/1.71.0/include', '-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']
#cpp_args = ['-std=c++11', ]

ext = Extension('sublevel', 
                sources = ['sublevel_v6.cpp'],
                language='c++',
                extra_compile_args = cpp_args,)
setup(name="Sublevel homology calculation in C-Extension", 
      include_dirs = [np.get_include()], #Add Include path of numpy
      ext_modules = [ext]
     )