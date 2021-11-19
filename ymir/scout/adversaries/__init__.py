from dataclasses import dataclass
from typing import Mapping

import numpy as np
import optax

from functools import partial

from . import backdoor
from . import scaler
from . import onoff
from . import freerider
from . import mouther
from . import labelflipper