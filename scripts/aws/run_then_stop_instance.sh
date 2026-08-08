#!/usr/bin/env bash

set -u

if (( $# == 0 )); then
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

if ! sudo -n true; then
  echo "Passwordless sudo is required for unattended EC2 shutdown." >&2
  exit 1
fi

"$@"
status=$?

if (( status != 0 )); then
  echo "Command exited with status ${status}; leaving the EC2 instance running." >&2
  exit "${status}"
fi

echo "Command completed successfully; stopping the EC2 instance in 10 seconds."
sleep 10
sync
sudo -n shutdown -h now
