"""Allow ``python -m whales_identify ...`` as an alias for the CLI."""

from whales_identify.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
