from .launch_util import is_installed, run_pip
from .keyframe.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

if not is_installed("rich"):
    run_pip("install rich", desc="Install rich", live=True)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']