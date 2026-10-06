"""Read numpy-2 pickles under numpy 1.x.

MolmoMotion-1M track NPZs store per-object dicts as pickled object arrays written with
numpy 2, whose pickles reference `numpy._core.*`. numpy 1.x names that package
`numpy.core`, so `np.load(..., allow_pickle=True)[key]` fails with
"No module named 'numpy._core'". The project venv pins numpy 1.26 (Depth Anything 3
requires numpy<2), so importing this module registers `numpy._core` as an alias of
`numpy.core`; under numpy 2 it does nothing.
"""

import sys

import numpy as np

if int(np.__version__.split(".")[0]) < 2 and "numpy._core" not in sys.modules:
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric
