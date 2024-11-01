import os
from typing import Any

import tango
from tango import Executor, Workspace
from tango.common.det_hash import CustomDetHash
from tango.format import DillFormat, Format
from tango.settings import TangoGlobalSettings

# https://github.com/allenai/tango/issues/606
from tango.workspaces.local_workspace import LocalWorkspace  # noqa: F401
from tango.workspaces.memory_workspace import MemoryWorkspace, default_workspace  # noqa: F401

__all__ = ["TangoStringHash", "step", "tango_executor", "tango_settings", "tango_workspace"]

tango_settings: TangoGlobalSettings = TangoGlobalSettings(
    workspace={
        **(
            {"type": "from_url", "url": f"local://{os.environ['TANGO_WORKSPACE_DIR']}"}
            if os.environ.get("TANGO_WORKSPACE_DIR")
            else {"type": "memory"}
        ),
    },
    file_friendly_logging=True,
)
tango_workspace: Workspace = tango.cli.prepare_workspace(settings=tango_settings)
tango_executor: Executor = tango.cli.prepare_executor(workspace=tango_workspace, settings=tango_settings)


## Use string class representation for hashing
## E.g. when serialization is not cannonical


class TangoStringHash(CustomDetHash):
    def det_hash_object(self) -> Any:
        self.__class__.__module__ = ""
        return str(self)


# cloudpickle sometimes sets __qualname__ = __name__
# so we should always set this to ensure hashing is consistent
# modifying tango.step.step (https://github.com/allenai/tango/blob/3400aa89d879fd4ea5b6986e4e24cdd852c0f4ad/tango/step.py#L800-L857)
def step(
    name: str | None = None,
    *,
    exist_ok: bool = False,
    bind: bool = False,
    deterministic: bool = True,
    cacheable: bool | None = None,
    version: str | None = None,
    format: Format = DillFormat("gz"),
    skip_id_arguments: set[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    from tango.step import FunctionalStep, Step

    def step_wrapper(step_func):
        @Step.register(name or step_func.__name__, exist_ok=exist_ok)
        class WrapperStep(FunctionalStep):
            DETERMINISTIC = deterministic
            CACHEABLE = cacheable
            VERSION = version
            FORMAT = format
            SKIP_ID_ARGUMENTS = skip_id_arguments or set()
            METADATA = metadata or {}

            WRAPPED_FUNC = step_func
            BIND = bind

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__class__.__module__ = "tango.step"
                self.__class__.__qualname__ = self.__class__.__name__

        return WrapperStep

    return step_wrapper
