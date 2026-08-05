"""Launch this repository's SigVue application."""

import sys
from pathlib import Path


def main() -> None:
    """Run SigVue with this project's profile unless another is supplied."""
    if "--config" not in sys.argv:
        profile = Path(__file__).resolve().parents[1] / "browser.toml"
        sys.argv.extend(("--config", str(profile)))
    from sigvue.web.application import main as sigvue_main

    sigvue_main()


if __name__ == "__main__":
    main()
