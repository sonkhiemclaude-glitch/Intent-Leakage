"""Allow ``python -m intent_leakage`` to invoke the command-line interface."""

from .cli import main

raise SystemExit(main())
