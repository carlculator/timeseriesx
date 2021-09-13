from pint import Unit

from timeseriesx import ureg


def coerce_unit(unit):
    """
    returns the convenient representation of a unit as a `pint.Unit`-object

    :param str/pint.Unit unit: the unit string to parse or a Unit object
    :return: the coerced unit
    :rtype: pint.Unit
    """
    if isinstance(unit, str):
        unit = ureg.parse_units(unit)
    if isinstance(unit, Unit):
        return unit
    else:
        return ValueError(f'{unit} is no valid unit')
