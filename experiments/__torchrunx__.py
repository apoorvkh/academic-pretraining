import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torchrunx


def build_logging_handlers(hostnames):
    log_dir = os.environ["TORCHRUNX_LOG_DIR"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    file_paths = [f"{log_dir}/{timestamp}-{hostname}.log" for hostname in hostnames]

    def _handler_builder() -> list[logging.Handler]:
        handlers = []

        stream_handler = torchrunx.stream_handler(hostname=hostnames[0], local_rank=0)
        stream_handler.addFilter(logging.Filter(name="academic-pretraining"))
        handlers.append(stream_handler)

        for hostname, file_path in zip(hostnames, file_paths):
            handlers += [
                torchrunx.file_handler(hostname=hostname, local_rank=None, file_path=file_path),
                torchrunx.file_handler(hostname=hostname, local_rank=0, file_path=file_path),
            ]
        return handlers

    return _handler_builder, file_paths


def distribute(
    func: Callable,
    func_args: tuple[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    hostnames: list[str] | None = None,
    workers_per_host: int | None = None,
) -> Any:
    if hostnames is None:
        hostnames = torchrunx.utils.environment.auto_hosts()
    if workers_per_host is None:
        workers_per_host = torchrunx.utils.environment.auto_workers()

    log_handlers_builder, log_files = build_logging_handlers(hostnames)

    print(f"Logging results of \"{func.__name__}\" to:")
    for file_path in log_files:
        print(f"  - {file_path}")

    return torchrunx.launch(
        func=func,
        func_kwargs=func_kwargs,
        hostnames=hostnames,
        workers_per_host=workers_per_host,
        log_handlers_builder=log_handlers_builder,
    ).rank(0)
