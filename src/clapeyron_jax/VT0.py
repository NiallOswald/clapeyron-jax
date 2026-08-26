"""Module that stores Clapeyron properties in (total) volume-temperature basis.

The functions stored in the VT0 module do not perform any type of phase stability
checking. The user must be sure to give a physically sensible volume value. For
calculations in volume-temperature basis that check and calculate if there are multiple
phases, use the VT module instead.
"""

from clapeyron_jax._julia import clapeyron as cl
from clapeyron_jax._wrapper import create_jax_wrapper

mass_density = create_jax_wrapper(cl.VT0.mass_density, "(),(),(n)->()")
speed_of_sound = create_jax_wrapper(cl.VT0.speed_of_sound, "(),(),(n)->()")
