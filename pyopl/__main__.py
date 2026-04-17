import argparse
import logging

from .pyopl_ide_bootstrap import OPLIDE


def main():
    parser = argparse.ArgumentParser(description="pyopl IDE")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,  # ensure DEBUG is applied even if logging was configured earlier
        )

    ide = OPLIDE(debug=args.debug)
    ide.mainloop()


if __name__ == "__main__":
    main()
