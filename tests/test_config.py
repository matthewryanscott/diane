"""Tests for configuration loading and generic value resolution."""

from diane.config import DianeConfig, ServerConfig, load_config, resolve_value


def test_resolve_at_reads_file_contents(tmp_path):
    """@path reads file contents, stripping trailing whitespace."""
    secret = tmp_path / "secret.txt"
    secret.write_text("my-secret-token\n")
    result = resolve_value(f"@{secret}", config_dir=tmp_path)
    assert result == "my-secret-token"


def test_resolve_at_relative_path(tmp_path):
    """@relative/path resolves relative to config_dir."""
    secret = tmp_path / "secret.txt"
    secret.write_text("relative-secret\n")
    result = resolve_value("@secret.txt", config_dir=tmp_path)
    assert result == "relative-secret"


def test_resolve_at_env_key(tmp_path):
    """@path:KEY extracts KEY=value from a dotenv-style file."""
    env = tmp_path / ".env"
    env.write_text("FOO=bar\nAPI_KEY=abc123\nBAZ=qux\n")
    result = resolve_value("@.env:API_KEY", config_dir=tmp_path)
    assert result == "abc123"


def test_resolve_at_env_key_missing(tmp_path):
    """@path:KEY raises if key not found."""
    env = tmp_path / ".env"
    env.write_text("FOO=bar\n")
    import pytest

    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        resolve_value("@.env:MISSING_KEY", config_dir=tmp_path)


def test_resolve_at_file_not_found(tmp_path):
    """@path raises if file doesn't exist."""
    import pytest

    with pytest.raises(RuntimeError, match="no-such-file"):
        resolve_value("@no-such-file", config_dir=tmp_path)


def test_resolve_at_env_quoted_value(tmp_path):
    """@path:KEY strips surrounding quotes from values."""
    env = tmp_path / ".env"
    env.write_text('MY_KEY="quoted-value"\n')
    result = resolve_value("@.env:MY_KEY", config_dir=tmp_path)
    assert result == "quoted-value"


def test_resolve_at_env_single_quoted(tmp_path):
    """@path:KEY strips single quotes too."""
    env = tmp_path / ".env"
    env.write_text("MY_KEY='single-quoted'\n")
    result = resolve_value("@.env:MY_KEY", config_dir=tmp_path)
    assert result == "single-quoted"


def test_resolve_plain_value_unchanged():
    """Plain values pass through regardless of config_dir."""
    assert resolve_value("hello", config_dir=None) == "hello"


def test_resolve_bang_still_works():
    """! prefix still runs shell commands."""
    result = resolve_value("!echo hello", config_dir=None)
    assert result == "hello"


def test_resolve_at_env_key_falls_back_to_env_var_when_file_missing(tmp_path, monkeypatch):
    """@path:KEY falls back to os.environ[KEY] when the file doesn't exist."""
    monkeypatch.setenv("MY_SECRET", "from-env")
    result = resolve_value("@nonexistent.env:MY_SECRET", config_dir=tmp_path)
    assert result == "from-env"


def test_resolve_at_env_key_falls_back_to_env_var_when_key_not_in_file(tmp_path, monkeypatch):
    """@path:KEY falls back to os.environ[KEY] when key is missing from file."""
    env = tmp_path / ".env"
    env.write_text("OTHER=stuff\n")
    monkeypatch.setenv("MY_SECRET", "from-env")
    result = resolve_value("@.env:MY_SECRET", config_dir=tmp_path)
    assert result == "from-env"


def test_resolve_at_env_key_still_raises_when_no_env_var(tmp_path, monkeypatch):
    """@path:KEY raises if file missing AND env var not set."""
    monkeypatch.delenv("NOPE", raising=False)
    import pytest

    with pytest.raises(RuntimeError, match="nonexistent.env"):
        resolve_value("@nonexistent.env:NOPE", config_dir=tmp_path)


def test_load_config_missing_file_returns_empty_config(tmp_path):
    assert load_config(tmp_path / "missing.toml") == DianeConfig(servers=[])


def test_load_config_empty_file_returns_empty_config(tmp_path):
    path = tmp_path / "diane.toml"
    path.write_text("")

    assert load_config(path) == DianeConfig(servers=[])


def test_load_config_servers_preserves_order_and_defaults(tmp_path):
    path = tmp_path / "diane.toml"
    path.write_text(
        '[[servers]]\n'
        'url = "http://localhost:11434"\n'
        'model = "custom-model"\n'
        '\n'
        '[[servers]]\n'
        'url = "http://worker.local:11434"\n'
    )

    assert load_config(path) == DianeConfig(servers=[
        ServerConfig(url="http://localhost:11434", model="custom-model"),
        ServerConfig(url="http://worker.local:11434", model="gemma4:e4b"),
    ])


def test_load_config_resolves_server_values_relative_to_config(tmp_path, monkeypatch):
    (tmp_path / "server-url.txt").write_text("http://localhost:11434\n")
    (tmp_path / "models.env").write_text("MODEL=file-model\n")
    monkeypatch.setenv("DIANE_TEST_MODEL", "environment-model")
    path = tmp_path / "diane.toml"
    path.write_text(
        '[[servers]]\n'
        'url = "@server-url.txt"\n'
        'model = "@models.env:MODEL"\n'
        '\n'
        '[[servers]]\n'
        'url = "!printf http://worker.local:11434"\n'
        'model = "@missing.env:DIANE_TEST_MODEL"\n'
    )

    assert load_config(path) == DianeConfig(servers=[
        ServerConfig(url="http://localhost:11434", model="file-model"),
        ServerConfig(url="http://worker.local:11434", model="environment-model"),
    ])
