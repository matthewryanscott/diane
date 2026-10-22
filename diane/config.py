import os
import random
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from diane.db import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "diane.toml"


def _is_fifo(path: Path) -> bool:
    """Check if a path is a FIFO (named pipe)."""
    try:
        return os.stat(path).st_mode & 0o170000 == 0o010000
    except OSError:
        return False


def _read_file_robust(path: Path, max_retries: int = 5, base_delay: float = 0.05) -> str:
    """Read a file with retries for FIFOs.

    FIFOs (like 1Password's .env pipe) can have timing issues where
    the read happens before data is available. This function retries
    with random backoff if the file appears empty or has issues.
    """
    is_fifo = _is_fifo(path)

    for attempt in range(max_retries):
        try:
            content = path.read_text()
            # For FIFOs, validate we got meaningful content
            if is_fifo and (not content or content.isspace()):
                # FIFO returned empty, retry with delay
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                time.sleep(delay)
                continue
            return content
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to read {path} after {max_retries} attempts: {e}")

    # Fallback - should not reach here
    return path.read_text()


def resolve_value(val: str, *, config_dir: Path | None = None) -> str:
    """Resolve a config value.

    Prefixes:
      !command  — run shell command, return stdout
      @path     — read file contents (relative to config_dir)
      @path:KEY — read dotenv-style file, extract KEY
    """
    if val.startswith("!"):
        cmd = val[1:]
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)  # noqa: S602
        if result.returncode != 0:
            stderr = result.stderr.strip()
            msg = f"Config command failed: {cmd!r}"
            if stderr:
                msg += f"\n{stderr}"
            raise RuntimeError(msg)
        return result.stdout.strip()

    if val.startswith("@"):
        ref = val[1:]
        # Split on first ':' to separate path from optional key
        if ":" in ref:
            file_part, key = ref.split(":", 1)
        else:
            file_part, key = ref, None

        file_path = Path(file_part)
        if not file_path.is_absolute() and config_dir is not None:
            file_path = config_dir / file_path

        if not file_path.exists():
            # When extracting a key and the file is missing, fall back to env vars
            if key is not None:
                env_val = os.environ.get(key)
                if env_val is not None:
                    return env_val
            raise RuntimeError(f"Config file not found: {file_part}")

        if key is None:
            return _read_file_robust(file_path).strip()

        # Parse dotenv-style KEY=value
        content = _read_file_robust(file_path)
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                v = v.strip()
                if len(v) >= 2 and v[0] in ('"', "'") and v[-1] == v[0]:
                    v = v[1:-1]
                return v
        # Key not in file — try env vars before failing
        env_val = os.environ.get(key)
        if env_val is not None:
            return env_val
        raise RuntimeError(f"Key {key!r} not found in {file_part}")

    return val


@dataclass
class ServerConfig:
    url: str
    model: str


@dataclass
class DianeConfig:
    servers: list[ServerConfig] = field(default_factory=list)


def load_config(path: Path | None = None) -> DianeConfig:
    """Load and resolve diane.toml from the startup workspace, or an explicit path."""
    path = path or CONFIG_PATH
    if not path.exists():
        return DianeConfig()

    with path.open("rb") as f:
        raw = tomllib.load(f)

    config_dir = path.parent

    def resolve(val: str) -> str:
        return resolve_value(val, config_dir=config_dir)

    servers = []
    for srv in raw.get("servers", []):
        servers.append(ServerConfig(
            url=resolve(srv["url"]),
            model=resolve(srv.get("model", "gemma4:e4b")),
        ))

    return DianeConfig(servers=servers)


_config: DianeConfig | None = None


def get_config() -> DianeConfig:
    """Get config, loading lazily on first access."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = load_config()
    return _config
