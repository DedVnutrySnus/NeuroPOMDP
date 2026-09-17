from __future__ import annotations

from types import SimpleNamespace

from neuropomdp import launcher


def test_launcher_bootstrap_script_and_child_command(monkeypatch) -> None:
    script_path = launcher._write_bootstrap_script()
    try:
        contents = script_path.read_text(encoding="utf-8")
        assert "neuropomdp.dashboard.app" in contents
        assert "main()" in contents

        config = launcher.LaunchConfig(port=12345, script_path=script_path)

        monkeypatch.setattr(launcher.sys, "frozen", False, raising=False)
        dev_command = launcher._child_command(config)
        assert dev_command[0] == launcher.sys.executable
        assert dev_command[1:3] == ["-m", "neuropomdp.launcher"]

        monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
        frozen_command = launcher._child_command(config)
        assert frozen_command[0] == launcher.sys.executable
        assert frozen_command[1] == "--serve"
        assert "--script" in frozen_command

        debug_config = launcher.LaunchConfig(port=12345, script_path=script_path, debug=True)
        debug_command = launcher._child_command(debug_config)
        assert debug_command[-1] == "--debug"
    finally:
        script_path.unlink(missing_ok=True)


def test_choose_free_port_returns_valid_local_port() -> None:
    port = launcher._choose_free_port()
    assert 0 < port < 65536


def test_streamlit_flag_options_are_pinned_to_localhost() -> None:
    options = launcher._streamlit_flag_options(45678)
    assert options["server_address"] == "127.0.0.1"
    assert options["server_port"] == 45678
    assert options["server_headless"] is True
    assert options["server_enableCORS"] is False
    assert options["server_enableXsrfProtection"] is False


def test_normal_launch_opens_browser_after_server_readiness(monkeypatch, tmp_path) -> None:
    script_path = tmp_path / "bootstrap.py"
    events: list[str] = []
    opened_urls: list[str] = []

    class FakeChild:
        def wait(self):
            return 0

    monkeypatch.setattr(launcher, "_write_bootstrap_script", lambda: script_path)
    monkeypatch.setattr(launcher, "_log_path", lambda: tmp_path / "launcher.log")
    monkeypatch.setattr(launcher, "_choose_free_port", lambda: 45678)
    monkeypatch.setattr(launcher, "_start_child_process", lambda config: FakeChild())
    monkeypatch.setattr(
        launcher,
        "_wait_for_server",
        lambda child, url, timeout: (
            events.append("readiness")
            or launcher.ReadinessResult(ready=True, http_status=200, tcp_ready=True)
        ),
    )
    monkeypatch.setattr(launcher, "_install_signal_handlers", lambda cleanup: None)
    monkeypatch.setattr(launcher, "_terminate_process", lambda child: None)
    monkeypatch.setattr(launcher, "_log_info", lambda message: None)
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: (events.append("browser"), opened_urls.append(url)))

    assert launcher._launch_dashboard(debug=False) == 0
    assert opened_urls == ["http://127.0.0.1:45678/"]
    assert events == ["readiness", "browser"]


def test_serve_mode_does_not_open_browser(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []

    class FakeBootstrap:
        def run(self, script_path, command_line, args, flag_options):
            calls.append((script_path, flag_options["server_port"]))

    monkeypatch.setattr(launcher, "_write_bootstrap_script", lambda: tmp_path / "bootstrap.py")
    monkeypatch.setattr(launcher, "_log_path", lambda: tmp_path / "launcher.log")
    monkeypatch.setattr(launcher, "_import_streamlit_bootstrap", lambda: FakeBootstrap())
    monkeypatch.setattr(launcher, "_configure_streamlit_runtime", lambda port: None)
    monkeypatch.setattr(launcher, "_install_parent_watchdog", lambda parent_pid: None)
    monkeypatch.setattr(launcher, "_log_streamlit_details", lambda config: None)
    monkeypatch.setattr(
        launcher.webbrowser,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("browser must not open")),
    )

    args = launcher._parse_args(["--serve", "--port", "45678"])

    assert launcher._serve_dashboard(args) == 0
    assert calls == [(str(tmp_path / "bootstrap.py"), 45678)]


def test_single_instance_mutex_rejects_existing_global_mutex(monkeypatch) -> None:
    calls: list[str] = []

    def create_mutex(_security, _owner, name):
        calls.append(name)
        return 101

    def close_handle(_handle):
        return True

    fake_kernel32 = SimpleNamespace(
        CreateMutexW=create_mutex,
        CloseHandle=close_handle,
        GetLastError=lambda: 183,
    )
    monkeypatch.setattr(launcher.ctypes, "windll", SimpleNamespace(kernel32=fake_kernel32), raising=False)

    try:
        with launcher._single_instance_mutex():
            raise AssertionError("mutex should have rejected the second instance")
    except RuntimeError as exc:
        assert "already running" in str(exc)

    assert calls == [launcher.GLOBAL_MUTEX_NAME]
