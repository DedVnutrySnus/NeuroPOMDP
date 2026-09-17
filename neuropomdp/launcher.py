"""Windows-friendly launcher for the NeuroPOMDP dashboard."""

from __future__ import annotations

import atexit
import argparse
import contextlib
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator
import webbrowser

from neuropomdp import __version__


LAUNCHER_NAME = "NeuroPOMDP"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_HOST = "127.0.0.1"
GLOBAL_MUTEX_NAME = "Global\\NeuroPOMDP_SingleInstance"
LOCAL_MUTEX_NAME = "Local\\NeuroPOMDP_SingleInstance"
ERROR_ALREADY_EXISTS = 183


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """Configuration for the launcher runtime."""

    port: int
    script_path: Path
    debug: bool = False
    parent_pid: int | None = None
    log_path: Path | None = None


def main(argv: list[str] | None = None) -> int:
    """Launch the packaged dashboard or serve it in child mode."""

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = _parse_args(args)
    _configure_logging(parsed.debug)
    _log_runtime_banner(parsed)

    if parsed.diagnose:
        return _diagnose()

    if parsed.serve:
        return _serve_dashboard(parsed)

    try:
        with _single_instance_mutex():
            return _launch_dashboard(parsed.debug)
    except Exception as exc:  # noqa: BLE001
        _show_error("NeuroPOMDP failed to start.", exc)
        return 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse launcher arguments."""

    parser = argparse.ArgumentParser(prog="neuropomdp.launcher", add_help=True)
    parser.add_argument("--serve", action="store_true", help="Run the internal Streamlit server.")
    parser.add_argument("--port", type=int, default=0, help="Port used by the internal Streamlit server.")
    parser.add_argument("--script", type=str, default="", help="Path to the Streamlit bootstrap script.")
    parser.add_argument("--parent-pid", type=int, default=0, help="PID of the supervising launcher process.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose launcher diagnostics.")
    parser.add_argument("--diagnose", action="store_true", help="Print runtime diagnostics and exit.")
    return parser.parse_args(argv)


def _launch_dashboard(debug: bool) -> int:
    """Start the child Streamlit server and keep it alive."""

    script_path = _write_bootstrap_script()
    log_path = _log_path()
    port = _choose_free_port()
    child = _start_child_process(
        LaunchConfig(port=port, script_path=script_path, debug=debug, log_path=log_path)
    )
    cleanup_state = {"done": False}

    def cleanup() -> None:
        if cleanup_state["done"]:
            return
        cleanup_state["done"] = True
        _terminate_process(child)
        with contextlib.suppress(OSError):
            script_path.unlink(missing_ok=True)

    atexit.register(cleanup)
    _install_signal_handlers(cleanup)

    url = f"http://{DEFAULT_HOST}:{port}/"
    readiness = _wait_for_server(child, url, timeout=DEFAULT_TIMEOUT_SECONDS)
    if not readiness.ready:
        cleanup()
        raise RuntimeError(readiness.message or f"Streamlit did not become ready on {url}.")

    webbrowser.open(url)

    try:
        return_code = child.wait()
        if return_code not in (0, None):
            raise RuntimeError(
                "Streamlit exited with status code "
                f"{return_code}.\n\n{_tail_text(log_path)}"
            )
        return 0
    finally:
        cleanup()


def _serve_dashboard(args: argparse.Namespace) -> int:
    """Run the dashboard inside the child Streamlit server."""

    script_path = Path(args.script) if args.script else _write_bootstrap_script()
    port = int(args.port) if int(args.port or 0) > 0 else _choose_free_port()
    log_path = _log_path()
    config = LaunchConfig(
        port=port,
        script_path=script_path,
        debug=bool(args.debug),
        parent_pid=args.parent_pid or None,
        log_path=log_path,
    )
    _log_info(
        "Entering serve mode "
        f"port={config.port} script={config.script_path} parent_pid={config.parent_pid}"
    )

    try:
        bootstrap = _import_streamlit_bootstrap()
        _configure_streamlit_runtime(config.port)
        _install_parent_watchdog(config.parent_pid)
        _log_streamlit_details(config)
        with contextlib.suppress(Exception):
            import streamlit.cli_util as cli_util

            cli_util.print_to_cli = lambda *args, **kwargs: None  # noqa: ARG005
        with contextlib.suppress(Exception):
            import click

            click.secho = lambda *args, **kwargs: None  # noqa: ARG005
            click.echo = lambda *args, **kwargs: None  # noqa: ARG005
        with contextlib.suppress(Exception):
            bootstrap._print_url = lambda *args, **kwargs: None  # type: ignore[attr-defined]  # noqa: ARG005
        bootstrap.run(str(config.script_path), False, [], _streamlit_flag_options(config.port))
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _show_error("Streamlit server failed to start.", exc)
        return 1


def _import_streamlit_bootstrap():
    """Import Streamlit lazily so diagnostics can report import failures clearly."""

    from streamlit.web import bootstrap

    return bootstrap


def _configure_streamlit_runtime(port: int) -> None:
    """Apply explicit Streamlit runtime settings before bootstrapping."""

    bootstrap = _import_streamlit_bootstrap()
    flag_options = _streamlit_flag_options(port)
    bootstrap.load_config_options(flag_options)

    with contextlib.suppress(Exception):
        import streamlit.config as st_config

        for option_name, value in (
            ("server.address", DEFAULT_HOST),
            ("server.port", int(port)),
            ("server.headless", True),
            ("server.enableCORS", False),
            ("server.enableXsrfProtection", False),
            ("browser.gatherUsageStats", False),
            ("browser.serverAddress", DEFAULT_HOST),
            ("browser.serverPort", int(port)),
            ("global.developmentMode", False),
        ):
            st_config.set_option(option_name, value)

    os.environ["STREAMLIT_SERVER_ADDRESS"] = DEFAULT_HOST
    os.environ["STREAMLIT_SERVER_PORT"] = str(int(port))
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = DEFAULT_HOST
    os.environ["STREAMLIT_BROWSER_SERVER_PORT"] = str(int(port))

    _log_info(
        "Configured Streamlit runtime "
        f"address={DEFAULT_HOST} port={port} headless=true enableCORS=false enableXsrfProtection=false"
    )


def _log_streamlit_details(config: LaunchConfig) -> None:
    """Log package and runtime details for packaged-build troubleshooting."""

    import streamlit

    _log_info(f"Streamlit version: {streamlit.__version__}")
    _log_info(f"Launch script: {config.script_path}")
    _log_info(f"Selected port: {config.port}")


def _choose_free_port() -> int:
    """Choose a free localhost port for Streamlit."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return int(sock.getsockname()[1])


def _write_bootstrap_script() -> Path:
    """Create a tiny Streamlit bootstrap script in the temp directory."""

    temp_dir = Path(tempfile.gettempdir()) / "NeuroPOMDP"
    temp_dir.mkdir(parents=True, exist_ok=True)
    script_path = temp_dir / f"streamlit_bootstrap_{uuid.uuid4().hex}.py"
    script_path.write_text(
        textwrap.dedent(
            """
            from neuropomdp.dashboard.app import main

            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return script_path


def _start_child_process(config: LaunchConfig) -> subprocess.Popen[bytes]:
    """Spawn the hidden Streamlit child process."""

    command = _child_command(config)
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_ADDRESS", DEFAULT_HOST)
    env.setdefault("STREAMLIT_SERVER_PORT", str(config.port))
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
    env.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
    env.setdefault("STREAMLIT_BROWSER_SERVER_ADDRESS", DEFAULT_HOST)
    env.setdefault("STREAMLIT_BROWSER_SERVER_PORT", str(config.port))

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    log_path = config.log_path or _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    try:
        _write_log_header(log_handle, "child-start")
        log_handle.write("COMMAND: " + " ".join(command) + "\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        _log_info(f"Child process started pid={process.pid}")
    finally:
        log_handle.close()

    return process


def _child_command(config: LaunchConfig) -> list[str]:
    """Build the command that launches the child Streamlit server."""

    if getattr(sys, "frozen", False):
        command = [
            sys.executable,
            "--serve",
            "--port",
            str(config.port),
            "--script",
            str(config.script_path),
            "--parent-pid",
            str(os.getpid()),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "neuropomdp.launcher",
            "--serve",
            "--port",
            str(config.port),
            "--script",
            str(config.script_path),
            "--parent-pid",
            str(os.getpid()),
        ]

    if config.debug:
        command.append("--debug")
    return command


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Result of probing the Streamlit server."""

    ready: bool
    message: str = ""
    http_status: int | None = None
    tcp_ready: bool = False


def _wait_for_server(proc: subprocess.Popen[bytes], url: str, timeout: float) -> ReadinessResult:
    """Wait until Streamlit responds on the expected localhost port."""

    deadline = time.monotonic() + timeout
    last_http_status: int | None = None
    tcp_ready = False

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return ReadinessResult(
                ready=False,
                message=(
                    f"Streamlit exited before readiness with code {proc.returncode}.\n\n"
                    f"{_tail_text(_log_path())}"
                ),
            )

        tcp_ready = _tcp_port_is_open(DEFAULT_HOST, int(url.rsplit(":", 1)[-1].rstrip("/")))
        http_status, http_ready = _probe_http(url)
        if http_status is not None:
            last_http_status = http_status
        if tcp_ready and http_ready:
            return ReadinessResult(ready=True, http_status=http_status, tcp_ready=True)
        time.sleep(0.25)

    if tcp_ready and last_http_status == 404:
        return ReadinessResult(
            ready=False,
            message=(
                f"Streamlit responded on {url} but returned HTTP 404.\n"
                "The TCP listener is up, but the expected dashboard route is not mounted."
            ),
            http_status=last_http_status,
            tcp_ready=True,
        )

    if proc.poll() is not None:
        return ReadinessResult(
            ready=False,
            message=(
                f"Streamlit exited before readiness with code {proc.returncode}.\n\n"
                f"{_tail_text(_log_path())}"
            ),
            http_status=last_http_status,
            tcp_ready=tcp_ready,
        )

    return ReadinessResult(
        ready=False,
        message=f"Streamlit did not become ready on {url}.",
        http_status=last_http_status,
        tcp_ready=tcp_ready,
    )


def _tcp_port_is_open(host: str, port: int) -> bool:
    """Return ``True`` when a TCP listener accepts connections."""

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _probe_http(url: str) -> tuple[int | None, bool]:
    """Probe a HTTP endpoint and report the status and readiness."""

    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            status = int(getattr(response, "status", response.getcode()))
            body = response.read(4096)
            content_type = response.headers.get("content-type", "")
            ready = 200 <= status < 400 and (
                "text/html" in content_type.lower() or b"<html" in body.lower() or b"streamlit" in body.lower()
            )
            return status, ready
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            body = exc.read(4096)
        except OSError:
            body = b""
        content_type = exc.headers.get("content-type", "") if exc.headers else ""
        ready = 200 <= status < 400 and (
            "text/html" in content_type.lower() or b"<html" in body.lower() or b"streamlit" in body.lower()
        )
        return status, ready
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, False


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the child process tree."""

    if proc.poll() is not None:
        return
    _log_info(f"Terminating child process pid={proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _log_warning(f"Child pid={proc.pid} did not exit after terminate(); killing.")
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5.0)


def _show_error(message: str, exc: Exception) -> None:
    """Display a startup failure to the user and log the traceback."""

    detail = f"{message}\n\n{exc}"
    _write_error_log(detail, exc)
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(0, detail, f"{LAUNCHER_NAME} {__version__}", 0x10)
        except Exception:
            print(detail, file=sys.stderr)
    else:
        print(detail, file=sys.stderr)


def _write_error_log(message: str, exc: Exception) -> None:
    """Persist startup failures for troubleshooting packaged builds."""

    log_dir = _runtime_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path()
    with log_path.open("a", encoding="utf-8") as handle:
        _write_log_header(handle, "error")
        handle.write(f"{message}\n")
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        handle.write("\n")


def _install_signal_handlers(cleanup: Callable[[], None]) -> None:
    """Install termination handlers for the parent launcher process."""

    def handler(signum: int, frame) -> None:  # noqa: ARG001
        cleanup()
        raise SystemExit(128 + int(signum))

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            continue


def _install_parent_watchdog(parent_pid: int | None) -> None:
    """Terminate the child server if the supervising launcher disappears."""

    if not parent_pid or os.name != "nt":
        return

    def _watch_parent() -> None:
        kernel32 = ctypes.windll.kernel32
        process = kernel32.OpenProcess(0x100000, False, int(parent_pid))
        if not process:
            return
        try:
            kernel32.WaitForSingleObject(process, 0xFFFFFFFF)
        finally:
            with contextlib.suppress(Exception):
                kernel32.CloseHandle(process)
            with contextlib.suppress(SystemExit):
                _log_warning(
                    f"Parent pid={parent_pid} exited; shutting down child server."
                )
                os._exit(0)

    import threading

    thread = threading.Thread(target=_watch_parent, name="NeuroPOMDPParentWatchdog", daemon=True)
    thread.start()


@contextmanager
def _single_instance_mutex() -> Iterator[None]:
    """Prevent multiple launcher instances from starting on Windows."""

    if os.name != "nt":
        yield
        return

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handles: list[int] = []
    try:
        for mutex_name in (GLOBAL_MUTEX_NAME, LOCAL_MUTEX_NAME):
            handle = kernel32.CreateMutexW(None, False, mutex_name)
            if not handle:
                continue
            handles.append(handle)
            if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
                raise RuntimeError("NeuroPOMDP is already running.")
            yield
            return
        raise RuntimeError("NeuroPOMDP is already running.")
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                kernel32.CloseHandle(handle)


def _diagnose() -> int:
    """Print runtime diagnostics and exit."""

    lines: list[str] = []
    lines.append(f"NeuroPOMDP {__version__}")
    lines.append(f"Frozen: {getattr(sys, 'frozen', False)}")
    lines.append(f"Executable: {sys.executable}")
    lines.append(f"MEIPASS: {getattr(sys, '_MEIPASS', '')}")
    lines.append(f"cwd: {Path.cwd()}")
    lines.append(f"launcher.py: {Path(__file__).resolve()}")
    lines.append(f"runtime log: {_log_path()}")
    lines.append(f"bootstrap script dir: {(_runtime_dir())}")

    for label, module_name in (
        ("Streamlit", "streamlit"),
        ("JAX", "jax"),
        ("NumPy", "numpy"),
        ("Matplotlib", "matplotlib"),
    ):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            lines.append(f"{label}: OK ({version})")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{label}: FAIL ({exc})")

    try:
        from neuropomdp.dashboard import app as dashboard_app

        lines.append(f"Dashboard: OK ({Path(dashboard_app.__file__).resolve()})")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Dashboard: FAIL ({exc})")

    port = _choose_free_port()
    lines.append(f"Port binding: OK ({port})")

    try:
        script_path = _write_bootstrap_script()
        lines.append(f"Bootstrap script: OK ({script_path})")
        with contextlib.suppress(OSError):
            script_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Bootstrap script: FAIL ({exc})")

    for line in lines:
        print(line)
        _log_info(line)
    return 0


def _streamlit_flag_options(port: int) -> dict[str, object]:
    """Build Streamlit config overrides using CLI-style option names."""

    return {
        "server_address": DEFAULT_HOST,
        "server_port": int(port),
        "server_headless": True,
        "server_enableCORS": False,
        "server_enableXsrfProtection": False,
        "browser_gatherUsageStats": False,
        "browser_serverAddress": DEFAULT_HOST,
        "browser_serverPort": int(port),
        "global_developmentMode": False,
    }


def _runtime_dir() -> Path:
    """Return a writable directory for launcher runtime files."""

    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "NeuroPOMDP"


def _log_path() -> Path:
    return _runtime_dir() / "launcher.log"


def _log_runtime_banner(parsed: argparse.Namespace) -> None:
    """Log the initial runtime context."""

    _log_info(
        "Startup "
        f"serve={parsed.serve} diagnose={parsed.diagnose} debug={parsed.debug} "
        f"port={parsed.port} script={parsed.script or '<auto>'} parent_pid={parsed.parent_pid} "
        f"frozen={getattr(sys, 'frozen', False)} cwd={Path.cwd()} executable={sys.executable} "
        f"meipass={getattr(sys, '_MEIPASS', '')}"
    )


def _configure_logging(debug: bool) -> None:
    """Configure file-based logging for the launcher."""

    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if debug and sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [pid=%(process)d] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _log_info(message: str) -> None:
    logging.getLogger("neuropomdp.launcher").info(message)


def _log_warning(message: str) -> None:
    logging.getLogger("neuropomdp.launcher").warning(message)


def _write_log_header(handle, section: str) -> None:
    handle.write(
        f"\n--- {section} {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"pid={os.getpid()} ppid={os.getppid()} ---\n"
    )
    handle.write(f"version={__version__}\n")
    handle.write(f"frozen={getattr(sys, 'frozen', False)}\n")
    handle.write(f"executable={sys.executable}\n")
    handle.write(f"meipass={getattr(sys, '_MEIPASS', '')}\n")
    handle.write(f"cwd={Path.cwd()}\n")


def _tail_text(path: Path, max_bytes: int = 6000) -> str:
    """Return the tail of a log file for error reporting."""

    if not path.exists():
        return "<log file missing>"
    data = path.read_bytes()
    tail = data[-max_bytes:]
    return tail.decode("utf-8", errors="replace")


def _probe_configured_runtime() -> None:
    """Used by tests to exercise the config override path without launching Streamlit."""

    _configure_streamlit_runtime(_choose_free_port())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
