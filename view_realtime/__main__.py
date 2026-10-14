"""Run the viewer with ``python -m view_realtime``."""

from .cli import main


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
