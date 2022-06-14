try:
    import veros  # noqa: F401
except ImportError:
    raise RuntimeError(
        "veros-bgc needs Veros to be installed (try `pip install veros`)"
    )

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from veros_bgc.variables import VARIABLES
from veros_bgc.settings import SETTINGS
from veros_bgc.core.npzd import npzd, setupNPZD
from veros_bgc.diagnostics.npzd_monitor import NPZDMonitor


__VEROS_INTERFACE__ = {
    name:"biogeochemistry",
    setup_entrypoint: setupNPZD,
    run_entrypoint: npzd,
    settings: SETTINGS,
    variables: VARIABLES,
    diagnostics: [NPZDMonitor],
}

