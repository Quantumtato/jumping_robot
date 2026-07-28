import argparse
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as exc:  # pragma: no cover
    raise SystemExit("boto3 is required. Install it with: pip install boto3") from exc


DEFAULT_INSTANCE_ID = "i-0a64abeb2c1a63f62"
DEFAULT_REGION = "us-east-1"
DEFAULT_SSH_ALIAS = "jumping-robot-aws"
DEFAULT_SSH_USER = "ubuntu"
DEFAULT_KEY_FILE = os.path.expanduser("~/.ssh/jumping_robot_aws")
DEFAULT_SECURITY_GROUP_ID = "sg-0b53a7e3dbd8387fe"
DEFAULT_REMOTE_PATH = "/home/ubuntu/workspace/jumping_robot"


def describe_instance(ec2_client, instance_id):
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    return response["Reservations"][0]["Instances"][0]


def wait_for_running(ec2_client, instance_id):
    while True:
        instance = describe_instance(ec2_client, instance_id)
        state = instance["State"]["Name"]
        if state == "running":
            return instance
        print(f"Instance {instance_id} is currently {state}; waiting...")
        time.sleep(5)


def start_instance(instance_id, region):
    ec2 = boto3.client("ec2", region_name=region)
    instance = describe_instance(ec2, instance_id)
    state = instance["State"]["Name"]
    if state == "running":
        print(f"Instance {instance_id} is already running.")
    elif state == "pending":
        instance = wait_for_running(ec2, instance_id)
    else:
        if state == "stopping":
            print(f"Waiting for instance {instance_id} to finish stopping...")
            ec2.get_waiter("instance_stopped").wait(InstanceIds=[instance_id])
        elif state != "stopped":
            raise RuntimeError(f"Cannot start instance {instance_id} while it is {state}.")

        print(f"Starting instance {instance_id}...")
        try:
            ec2.start_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "InsufficientInstanceCapacity":
                instance_type = describe_instance(ec2, instance_id)["InstanceType"]
                raise RuntimeError(
                    f"AWS currently has no capacity for {instance_type} in this "
                    "instance's Availability Zone. Retry later or use another instance type."
                ) from exc
            raise
        instance = wait_for_running(ec2, instance_id)

    public_ip = instance.get("PublicIpAddress")
    if not public_ip:
        raise RuntimeError(f"Instance {instance_id} is running but has no public IP address yet.")
    print(f"Instance public IP: {public_ip}")
    return public_ip


def ensure_ssh_config(alias, host, user, key_file):
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(exist_ok=True)
    config_path = ssh_dir / "config"
    entry = (
        f"Host {alias}\n"
        f"    HostName {host}\n"
        f"    User {user}\n"
        f"    IdentityFile {key_file}\n"
        "    StrictHostKeyChecking no\n"
        "    UserKnownHostsFile ~/.ssh/known_hosts\n"
    )

    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    host_block = re.compile(
        rf"(?ms)^Host[ \t]+{re.escape(alias)}[ \t]*\r?\n.*?(?=^Host[ \t]+|\Z)"
    )
    if host_block.search(existing):
        updated = host_block.sub(lambda _: entry.rstrip() + "\n", existing)
    else:
        updated = existing.rstrip() + "\n\n" + entry
    config_path.write_text(updated.lstrip(), encoding="utf-8")


def refresh_ssh_ingress(ec2_client, security_group_id):
    public_ip = (
        urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10)
        .read()
        .decode("ascii")
        .strip()
    )
    current_cidr = f"{public_ip}/32"
    group = ec2_client.describe_security_groups(GroupIds=[security_group_id])[
        "SecurityGroups"
    ][0]

    for permission in group.get("IpPermissions", []):
        if (
            permission.get("IpProtocol") != "tcp"
            or permission.get("FromPort") != 22
            or permission.get("ToPort") != 22
        ):
            continue
        for ip_range in permission.get("IpRanges", []):
            if ip_range.get("Description") != "Current development machine":
                continue
            if ip_range["CidrIp"] == current_cidr:
                print(f"SSH access already allows {current_cidr}.")
                return
            ec2_client.revoke_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [ip_range],
                    }
                ],
            )

    ec2_client.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [
                    {
                        "CidrIp": current_cidr,
                        "Description": "Current development machine",
                    }
                ],
            }
        ],
    )
    print(f"SSH access updated to {current_cidr}.")


def open_vscode_remote(alias, remote_path):
    candidates = ["code", "code.cmd"]
    for name in candidates:
        code_path = shutil.which(name)
        if not code_path:
            continue
        command = [code_path, "--remote", f"ssh-remote+{alias}", remote_path]
        if os.name == "nt" and code_path.lower().endswith((".bat", ".cmd")):
            command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", *command]
        return subprocess.run(command, check=False).returncode == 0

    print("VS Code CLI was not found on PATH.")
    print(f"You can connect manually with: Remote-SSH: Connect to Host -> {alias}")
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Start an EC2 instance and open it in VS Code Remote SSH")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--ssh-alias", default=DEFAULT_SSH_ALIAS)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    parser.add_argument("--security-group-id", default=DEFAULT_SECURITY_GROUP_ID)
    parser.add_argument("--remote-path", default=DEFAULT_REMOTE_PATH)
    parser.add_argument("--skip-ssh-config", action="store_true")
    parser.add_argument("--skip-ingress-update", action="store_true")
    parser.add_argument("--skip-vscode", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.key_file).is_file():
        raise FileNotFoundError(f"SSH private key not found: {args.key_file}")

    if not args.skip_ingress_update:
        ec2 = boto3.client("ec2", region_name=args.region)
        refresh_ssh_ingress(ec2, args.security_group_id)

    public_ip = start_instance(args.instance_id, args.region)

    if not args.skip_ssh_config:
        ensure_ssh_config(args.ssh_alias, public_ip, args.ssh_user, args.key_file)
        print(f"SSH alias '{args.ssh_alias}' configured.")

    print(f"SSH command: ssh {args.ssh_alias}")

    if not args.skip_vscode:
        open_vscode_remote(args.ssh_alias, args.remote_path)


if __name__ == "__main__":
    main()
