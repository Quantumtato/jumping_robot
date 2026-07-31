import argparse

from start_instance_and_vscode_remote import (
    DEFAULT_INSTANCE_ID,
    DEFAULT_REGION,
    stop_instance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop the jumping robot EC2 instance")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--region", default=DEFAULT_REGION)
    args = parser.parse_args()
    stop_instance(args.instance_id, args.region)


if __name__ == "__main__":
    main()
