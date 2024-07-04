
# This file contains functions exploring the possibility of geometry-dependent
# multipoles. Essentially, the idea is the same as in EEM, except we also carry
# out an expansion of the energy in terms of dipoles and quadrupoles. This means
# there is a hardness parameter associated with each charge and every component
# of dipoles and quadrupoles. You can imagine doing this the EEM way which is to
# have a linear parameter that sets the main value of the multipole elements and
# then a quadratic parameter that controls the variation with potential/field.
# Or, you can do it as an expansion around an already equilibrated solution.
# In the latter case, you would use a distributed multipole analysis to determine
# the appropriate multipoles at equilibrium, then expand around those values.

