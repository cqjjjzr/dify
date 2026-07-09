from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from graphon.file.runtime import peek_workflow_file_runtime, set_workflow_file_runtime

if TYPE_CHECKING:
    from graphon.file.protocols import WorkflowFileRuntimeProtocol


class _LazyDifyWorkflowFileRuntime:
    def __init__(self) -> None:
        self._runtime: WorkflowFileRuntimeProtocol | None = None

    def _load(self) -> WorkflowFileRuntimeProtocol:
        if self._runtime is None:
            from core.app.workflow.file_runtime import bind_dify_workflow_file_runtime

            bind_dify_workflow_file_runtime()
            runtime = peek_workflow_file_runtime()
            assert runtime is not None
            assert runtime is not self
            self._runtime = runtime
        return self._runtime

    def __getattr__(self, name: str) -> object:
        return getattr(self._load(), name)


@pytest.fixture(autouse=True)
def _bind_workflow_file_runtime() -> None:
    runtime = cast("WorkflowFileRuntimeProtocol", _LazyDifyWorkflowFileRuntime())
    set_workflow_file_runtime(runtime)
