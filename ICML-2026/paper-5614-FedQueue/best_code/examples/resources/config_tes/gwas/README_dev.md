
**Build a local docker image for the client and server, and run the example with TES.**

```bash
docker build --no-cache -f examples/resources/config_tes/gwas/Dockerfile -t appfl/client-gwas:latest .
```

**Build and push the docker image to Docker Hub, and run the example with TES**

```bash
docker buildx build --platform linux/amd64  -f examples/resources/config_tes/gwas/Dockerfile -t zilinghan/client-gwas:latest  --push .
```

**Run the example with TES**

```bash
python tes/run.py --server_config ./resources/config_tes/gwas/server.yaml --client_config ./resources/config_tes/gwas/clients.yaml
```
