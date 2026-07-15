# GA4GH TES (Task Execution Service) Integration

This directory demonstrates APPFL federated learning using GA4GH Task Execution Service (TES) for distributed client execution.

## Setup

### 1. Install and Start Funnel (Local TES Server)
Please refer to [this link](https://ohsu-comp-bio.github.io/funnel/download/) to download the latest funnel. Funnel is a lightweight TES server implementation for local testing. Run the following commands to create a workspace and start the server:

```bash
mkdir -p /tmp/funnel-workspace
funnel server run --LocalStorage.AllowedDirs /tmp/funnel-workspace
```

### 2. Build Base Docker Image
Run the following command from the **APPFL root directory** to build the Docker image used by TES clients.
```bash
docker build -f examples/tes/Dockerfile -t appfl/client:latest .
```

## Running the Example

### Basic Example - Training TinyNet on Synthetic Data

This runs federated learning with:
- 2 TES-based FL clients using TinyNet model on synthetic data\
- Both 2 clients run on the same TES endpoint at `http://localhost:8000`

```bash
cd examples
python tes/run.py --server_config ./resources/config_tes/simple_net/server.yaml \
   --client_config ./resources/config_tes/simple_net/clients.yaml
```

### Other Examples

For more advanced scenarios, see the configurations and setup instructions within different sub-directories of `examples/resources/config_tes/`:

- **`simple_net_embedded/`** - Data embedding approach with client-specific Docker images
- **`simple_net_s3/`** - S3 storage for remote TES deployments (works across networks)

Each subdirectory contains complete configurations and setup scripts for different TES deployment patterns.

## Troubleshooting

- **Funnel not running**: Check that `funnel server run --LocalStorage.AllowedDirs /tmp/funnel-workspace` is active and accessible at `http://localhost:8000`
- **Docker image missing**: Ensure you've built the base image with `docker build -f examples/tes/Dockerfile -t appfl/client:latest .` from APPFL root directory
- **Task failures**: Check Funnel logs and verify the workspace directory `/tmp/funnel-workspace` exists and is writable
- **File access errors**: Ensure Funnel was started with `--LocalStorage.AllowedDirs /tmp/funnel-workspace` to allow access to the shared workspace
