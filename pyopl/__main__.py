


from .pyopl_cli import main as cli_main


def main():
    # Delegate to the CLI entrypoint which preserves the IDE-as-default behaviour
    return cli_main()


if __name__ == "__main__":
    main()
