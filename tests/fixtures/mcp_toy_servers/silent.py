"""Działa dalej, czyta stdin, nigdy nie odpowiada na initialize."""
# ruff: noqa  (skrypt-zabawka skopiowany z kursu: zwięzły styl jak w oryginale)

import os
import sys
import time

open(sys.argv[1], "w").write(str(os.getpid()))
for _ in sys.stdin:
    pass
time.sleep(600)
