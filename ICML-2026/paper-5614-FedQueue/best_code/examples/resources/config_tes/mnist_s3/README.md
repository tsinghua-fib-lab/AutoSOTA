# TES MNIST S3 Example

APPFL TES integration with **MNIST dataset and S3 storage** for remote federated learning.

## Prerequisites

```bash
# S3 bucket
aws s3 mb s3://your-appfl-tes-bucket --region us-east-1

# AWS credentials (server only)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Docker image
docker build --no-cache -f examples/tes/Dockerfile -t appfl/client:latest .
```

## Create a Remote TES Endpoint on AWS EC2

If you do not have a remote TES server, you can set one up on an AWS EC2 instance:

a. Launch an EC2 instance (e.g., t3.large) with Amazon Linux 2

b. Install necessary docker and funnel:

```bash
# Install Docker
sudo yum install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
# Install Funnel
sudo yum groupinstall -y "Development Tools"
/bin/bash -c "$(curl -fsSL https://github.com/ohsu-comp-bio/funnel/releases/latest/download/install-funnel.sh)" -- 0.11.3
funnel version
```

c. Configure Funnel with a `funnel.conf` file (example below):

```ini
# funnel.conf  — local compute + local storage
Database: boltdb
Compute: local
EventWriters:
  - boltdb
  - log

Server:
  HostName: 0.0.0.0
  HTTPPort: 8000
  RPCPort: 9090

Worker:
  WorkDir: /tmp/funnel-workspace
  # Optional tweaks:
  # LeaveWorkDir: false
  # LogTailSize: 10000
  # MaxParallelTransfers: 10

# Allow the worker to read/write specific local paths
LocalStorage:
  AllowedDirs:
    - /tmp/funnel-workspace
    - /home/ec2-user   # add any other paths you'll use
```
d. Start the Funnel server:

```bash
mkdir -p /tmp/funnel-workspace
funnel server run --config funnel.conf
```

e. Ensure port 8000 is open in the EC2 security group for TES access. Then you can replace the `comm_configs.tes_endpoint` in the client config with your EC2 public IP, e.g. `http://<ec2-public-ip>:8000`. For `docker_iamge`, you can either use our pre-built image `zilinghan/tes-client:latest` or build your own.

## Configuration

- **Model**: CNN for MNIST (28x28 images, 10 classes)
- **Data**: Non-IID class partitioning between 2 clients
- **Storage**: S3 with presigned URLs for secure transfer

## Usage

1. **Configure**: Update S3 bucket in `server.yaml`
2. **Run**:
```bash
cd examples
python tes/run.py --server_config ./resources/config_tes/mnist_s3/server.yaml \
                  --client_config ./resources/config_tes/mnist_s3/clients.yaml
```

## Benefits

- ✅ Realistic FL with MNIST + CNN
- ✅ Remote TES deployment via S3
- ✅ Non-IID data distribution
- ✅ Secure presigned URL transfers
