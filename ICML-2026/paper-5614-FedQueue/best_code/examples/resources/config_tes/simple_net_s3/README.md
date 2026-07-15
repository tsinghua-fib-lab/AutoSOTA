# TES S3 Storage Example

This configuration demonstrates APPFL TES integration with **S3 storage** for remote TES deployments using secure presigned URLs.

## Prerequisites

### 0. Create a Remote TES Endpoint on AWS EC2

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


### 1. AWS S3 Bucket
```bash
aws s3 mb s3://your-appfl-tes-bucket --region us-east-1
```

### 2. AWS Credentials (Server Only)
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

## Configuration

### Server (S3 Storage)
```yaml
server_configs:
  comm_configs:
    tes_configs:
      file_storage: "s3"
      file_storage_kwargs:
        s3_bucket: "your-appfl-tes-bucket"
        s3_region: "us-east-1"
        presigned_url_expiry: 3600  # 1 hour
```

### Clients (Remote TES Endpoints)
```yaml
comm_configs:
  tes_configs:
    tes_endpoint: # your-remote-tes-endpoint
    auth_token: "${TES_AUTH_TOKEN}" # optional if TES requires auth
```

## Usage

1. **Update Configuration**: Set your S3 bucket in `server.yaml` and TES endpoints in `clients.yaml`

2. **Set Environment**: Export AWS credentials and TES auth tokens

3. **Run Example**:
```bash
cd examples
python tes/run.py --server_config ./resources/config_tes/simple_net_s3/server.yaml \
                  --client_config ./resources/config_tes/simple_net_s3/clients.yaml
```

## How S3 Transfer Works

- **Server → Client**: Server uploads models to S3, provides presigned download URLs to TES
- **Client → Server**: Server generates presigned upload URLs, clients upload results directly to S3
- **Security**: Clients only get time-limited upload permissions, no AWS credentials needed

## Benefits

- ✅ Works with remote TES endpoints across networks
- ✅ Secure transfer using presigned URLs
- ✅ No AWS credentials needed in client containers
- ✅ Automatic S3 cleanup after completion
