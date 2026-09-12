"""`robotwin_icil.serve` speaking protocol version 0: a model environment left out of date.

Run as a script with the package on PYTHONPATH; takes serve's own arguments.
"""

from robotwin_icil import protocol, serve

if __name__ == "__main__":
    protocol.PROTOCOL_VERSION = 0
    raise SystemExit(serve.main())
