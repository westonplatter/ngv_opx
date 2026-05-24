"""
py_lets_be_rational (transitively imported by py_vollib) does
`from _testcapi import DBL_MIN, DBL_MAX`. CPython >= 3.12 no longer ships
_testcapi in standard installs, so the import fails. Shim it from
sys.float_info before py_vollib is imported anywhere in the test session.
"""
import sys
import types

if "_testcapi" not in sys.modules:
    shim = types.ModuleType("_testcapi")
    shim.DBL_MIN = sys.float_info.min
    shim.DBL_MAX = sys.float_info.max
    sys.modules["_testcapi"] = shim
