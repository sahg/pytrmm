import os
import subprocess

# BEFORE importing distutils, remove MANIFEST. distutils doesn't
# properly update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from setuptools import setup

MAJOR               = 0
MINOR               = 2
MICRO               = 0
ISRELEASED          = True
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

dev_version_py = 'pytrmm/__dev_version.py'

def generate_version_py(filename):
    try:
        if os.path.exists(".git"):
            # should be a Git clone, use revision info from Git
            s = subprocess.Popen(["git", "rev-parse", "HEAD"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out = s.communicate()[0]
            GIT_REVISION = out.strip()
        elif os.path.exists(dev_version_py):
            # should be a source distribution, use existing dev
            # version file
            from pytopkapi.__dev_version import git_revision as GIT_REVISION
        else:
            GIT_REVISION = "Unknown"
    except:
        GIT_REVISION = "Unknown"

    FULL_VERSION = VERSION
    if not ISRELEASED:
        FULL_VERSION += '.dev-'
        FULL_VERSION += GIT_REVISION[:7]

    cnt = """\
# This file was autogenerated
version = '%s'
git_revision = '%s'
"""
    cnt = cnt % (FULL_VERSION, GIT_REVISION)

    f = open(filename, "w")
    try:
        f.write(cnt)
    finally:
        f.close()

    return FULL_VERSION, GIT_REVISION

if __name__ == '__main__':
    full_version, git_rev = generate_version_py(dev_version_py)

    setup(name='pytrmm',
          version=full_version,
          description='Tools for reading TRMM data in Python',
          long_description = """\
PyTRMM - Python tools for reading TRMM data
===========================================

PyTRMM is a BSD licensed Python library containing tools for reading
the data files produced by the Tropical Rainfall Measuring Mission
(TRMM - http://trmm.gsfc.nasa.gov/). At this stage the package has
tools to read the TRMM 3B4XRT data files, but the code should be
useful for reading other data provided by TRMM.

""",
          license='BSD',
          author='Scott Sinclair',
          author_email='scott.sinclair.za@gmail.com',
          url='http://github.com/sahg/pytrmm',
          download_url='http://github.com/sahg/pytrmm',
          packages=['pytrmm'],
          classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: BSD License',
          'Environment :: Console',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          ],
          )

