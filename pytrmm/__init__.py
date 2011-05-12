"""Package tools for reading TRMM data.

"""

try:
    from __dev_version import version as __version__
    from __dev_version import git_revision as __git_revision__
except ImportError:
    from __version import version as __version__
    from __version import git_revision as __git_revision__

from trmm3b4xrt import *
