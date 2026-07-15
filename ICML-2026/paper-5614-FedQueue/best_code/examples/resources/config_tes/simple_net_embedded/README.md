# TES Data Embedding Demo

This approach embeds client datasets directly into Docker images during build time, avoiding the need for runtime volume mounting which isn't supported by standard TES/Funnel.

## Approach

Instead of mounting volumes at runtime, we:
1. **Generate datasets** on the host filesystem
2. **Build client-specific images** with data baked in
3. **Use different images** for each client in TES tasks

## Setup

### 1. Generate Test Datasets
```bash
cd examples/resources/config_tes/simple_net_embedded
python generate_test_data.py
```

### 2. Build Client-Specific Images
```bash
chmod +x build_client_images.sh
./build_client_images.sh
```

This creates:
- `appfl/client1:data-embedded` (contains client_0 data)
- `appfl/client2:data-embedded` (contains client_1 data)

### 3. Start Funnel
```bash
mkdir -p /tmp/funnel-workspace
funnel server run --LocalStorage.AllowedDirs /tmp/funnel-workspace
```

## How It Works

### Build Process:
```dockerfile
# Dockerfile.client1
FROM appfl/client:latest
COPY /tmp/tes-data/client_0/ /data/    # Data baked into image
ENV DATA_DIR=/data
```

### Runtime:
- **TESClient1** uses `appfl/client1:data-embedded`
- **TESClient2** uses `appfl/client2:data-embedded`
- Each container has `/data` with client-specific datasets

### Data Flow:
```
Host: /tmp/tes-data/client_0/
  ↓ (COPY during build)
Image: appfl/client1:data-embedded with /data
  ↓ (TES container starts)
Container: /data/client_0_features.csv available
  ↓ (file_dataset.py loads)
FL Client: Different datasets per client
```

## Testing

### Run with Data-Embedded Images
```bash
cd examples
python tes/run.py --server_config ./resources/config_tes/simple_net_embedded/server.yaml \
  --client_config ./resources/config_tes/simple_net_embedded/clients.yaml
```
