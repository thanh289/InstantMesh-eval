from __future__ import annotations

import numpy as np

INPUT_VIEW_IDS = [0, 8, 16, 24, 32, 40, 48, 56, 64]
INPUT_CAMERA_PARAMS = [(0.0, 45.0 * i) for i in range(8)] + [(89.99, 0.0)]
EVAL_CAMERA_PARAMS = [(30.0, 45.0 * i) for i in range(8)] + [(60.0, 45.0 * i) for i in range(8)]


