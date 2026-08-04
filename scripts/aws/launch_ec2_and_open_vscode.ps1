param(
    [string]$Profile = "default",
    [string]$Region = "us-east-1",
    [Parameter(Mandatory = $false)]
    [string]$ImageId,
    [string]$InstanceType = "g4dn.xlarge",
    [string]$KeyName,
    [string]$InstanceName = "jumping-robot-ml",
    [string]$SshUser = "ubuntu",
    [string]$KeyFilePath,
    [string]$SecurityGroupId,
    [string]$SubnetId,
    [string]$InstanceId,
    [switch]$OpenInVSCode,
    [switch]$SkipSshConfig
)

$ErrorActionPreference = "Stop"

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

function Get-ExistingSshKeyPath {
    param([string]$PreferredPath)

    if ($PreferredPath -and (Test-Path $PreferredPath)) {
        return $PreferredPath
    }

    $candidatePaths = @(
        $PreferredPath,
        "$HOME/.ssh/id_ed25519",
        "$HOME/.ssh/id_rsa",
        "$HOME/.ssh/id_ecdsa"
    ) | Where-Object { $_ }

    foreach ($path in $candidatePaths) {
        if ($path -and (Test-Path $path)) {
            return $path
        }
    }

    return $null
}

Assert-Command -Name "aws"

if (-not $KeyFilePath) {
    $KeyFilePath = Get-ExistingSshKeyPath -PreferredPath $KeyFilePath
}

if (-not $KeyFilePath) {
    throw "No SSH private key file found. Pass -KeyFilePath to an existing private key or place one in ~/.ssh/id_ed25519 or ~/.ssh/id_rsa."
}

if (-not $ImageId -and -not $InstanceId) {
    throw "Pass -ImageId to launch a new instance, or -InstanceId to reuse an existing one. Example: -ImageId ami-0f3a4ca79c857d43f -InstanceType g4dn.xlarge"
}

$env:AWS_PROFILE = $Profile
$env:AWS_DEFAULT_REGION = $Region

if ($InstanceId) {
    Write-Host "Using existing instance $InstanceId"
    $instanceIdToUse = $InstanceId
}
else {
    if (-not $KeyName) {
        throw "Pass -KeyName with the name of an existing EC2 key pair."
    }

    $launchArgs = @(
        "ec2", "run-instances",
        "--image-id", $ImageId,
        "--count", "1",
        "--instance-type", $InstanceType,
        "--key-name", $KeyName,
        "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=$InstanceName}]"
    )

    if ($SecurityGroupId) {
        $launchArgs += @("--security-group-ids", $SecurityGroupId)
    }

    if ($SubnetId) {
        $launchArgs += @("--subnet-id", $SubnetId)
    }

    $launchOutput = aws @launchArgs
    $instanceIdToUse = ($launchOutput | ConvertFrom-Json).Instances[0].InstanceId
    Write-Host "Launched instance $instanceIdToUse"
}

aws ec2 wait instance-running --instance-ids $instanceIdToUse | Out-Null

$describeOutput = aws ec2 describe-instances --instance-ids $instanceIdToUse --output json | ConvertFrom-Json
$publicIp = $describeOutput.Reservations[0].Instances[0].PublicIpAddress

if (-not $publicIp) {
    throw "Instance $instanceIdToUse is running but has no public IP address yet."
}

$sshHostAlias = "jumping-robot-aws"

if (-not $SkipSshConfig) {
    $sshConfigPath = Join-Path $HOME ".ssh/config"
    $sshConfigDir = Split-Path $sshConfigPath -Parent
    if (-not (Test-Path $sshConfigDir)) {
        New-Item -ItemType Directory -Force -Path $sshConfigDir | Out-Null
    }

    $sshConfigEntry = @"
Host $sshHostAlias
    HostName $publicIp
    User $SshUser
    IdentityFile $KeyFilePath
    StrictHostKeyChecking no
    UserKnownHostsFile ~/.ssh/known_hosts
    LocalForward 18081 127.0.0.1:8080
    LocalForward 16006 127.0.0.1:6006
"@

    $existingConfig = ""
    if (Test-Path $sshConfigPath) {
        $existingConfig = Get-Content -Path $sshConfigPath -Raw
    }

    if ($existingConfig -notmatch [regex]::Escape("Host $sshHostAlias")) {
        Add-Content -Path $sshConfigPath -Value "`n$sshConfigEntry"
    }
    else {
        Write-Host "SSH config already contains an entry for $sshHostAlias"
    }
}

Write-Host "Instance public IP: $publicIp"
Write-Host "SSH host alias: $sshHostAlias"
Write-Host "Use: ssh $sshHostAlias"

if ($OpenInVSCode) {
    if (-not (Get-Command code -ErrorAction SilentlyContinue)) {
        throw "VS Code CLI 'code' was not found. Install the VS Code command line tool or omit -OpenInVSCode."
    }

    & code --remote ssh-remote+$sshHostAlias
}
