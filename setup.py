from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = [
    Extension("swt.fastRay",
    ["swt/fastRay.pyx"],
  ),
    Extension("swt.fastConnectedComponents",
      ['swt/fastConnectedComponents.pyx'],
  )
])