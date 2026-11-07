"""Kończy działanie, zanim na cokolwiek odpowie."""
# ruff: noqa  (skrypt-zabawka skopiowany z kursu: zwięzły styl jak w oryginale)

import os
import sys

open(sys.argv[1], "w").write(str(os.getpid()))
print("zabawkowy serwer: celowo nie startuje", file=sys.stderr)
sys.exit(3)
