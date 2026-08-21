"""Module that stores Clapeyron properties in (total) volume-temperature basis.

The functions stored in the VT0 module do not perform any type of phase stability
checking. The user must be sure to give a physically sensible volume value. For
calculations in volume-temperature basis that check and calculate if there are multiple
phases, use the VT module instead.
"""

from clapeyron_jax._modules import ClapeyronModules
from clapeyron_jax._wrapper import create_jax_wrapper

_module = ClapeyronModules.VT0


# Pressure-based bulk properties
speed_of_sound = create_jax_wrapper("speed_of_sound", "(),(),(n)->()", _module)

# Multiphase properties
bubble_pressure = create_jax_wrapper("bubble_pressure", "(),(n)->(),(),(),(n)", _module)
