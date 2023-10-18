# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, unused-import
"""transforms compatible with generators Transform framework"""


from opencvlite.transforms import gamma, gamma1, log, sigmoid
from opencvlite.transforms import compute_average2, equalize_adapthist, equalize_hist
from opencvlite.transforms import histeq, histeq_adapt, histeq_color
from opencvlite.transforms import intensity, resize
from opencvlite.transforms import Transforms, Transform

from opencvlite.color import BGR2RGB, RGB2BGR, togreyscale
