# APPFL as a Service (APPFLx)

This submodule supports APPFLx — APPFL as a Service. The server-side federated learning job runs inside a Docker container on AWS ECS, downloads its configuration from S3, dispatches training tasks to client Globus Compute endpoints, and uploads results back to S3 when finished.

This README provides instructions for running the entry point script locally (without Docker) and building the Docker image.

---

## Step 1 — Generate and Validate Globus Tokens

The entry point requires two Globus access tokens:

- **`compute_token`** — authorizes calls to the Globus Compute service
- **`openid_token`** — authorizes identity lookup via Globus Auth

### 1a. Generate tokens (interactive browser login)

```bash
python _token_generation.py --login
```

This prints a Globus authorization URL. Open it in a browser, log in, copy the authorization code, and paste it back into the terminal. The script will print the full token response — an example response is shown below:

```bash
OAuthAuthorizationCodeResponse:
  id_token: eyJhbGciOi... (truncated)
  by_resource_server:
    {
      "auth.globus.org": {
        "scope": "openid",
        "access_token": <ACCESS_TOKEN_STRING>,
        "refresh_token": null,
        "token_type": "Bearer",
        "expires_at_seconds": 1772839080,
        "resource_server": "auth.globus.org"
      },
      "funcx_service": {
        "scope": "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all",
        "access_token": <ACCESS_TOKEN_STRING>,
        "refresh_token": null,
        "token_type": "Bearer",
        "expires_at_seconds": 1772839080,
        "resource_server": "funcx_service"
      }
    }
```

- `access_token` under the `funcx_service` (or `compute`) resource server → this is your **`compute_token`**
- `access_token` under the `auth.globus.org` resource server → this is your **`openid_token`**


### 1b. Validate tokens against a Globus Compute endpoint

After obtaining the tokens, verify that they can actually reach a running endpoint by passing `--endpoint_id 4f53131b-e59d-465f-9153-efe9b02d9f3e`:

> **💡 Note: You can use the above endpoint id - contact me if you do not have access.**

```bash
python _token_generation.py \
    --compute_token <compute_token> \
    --openid_token  <openid_token> \
    --endpoint_id   4f53131b-e59d-465f-9153-efe9b02d9f3e
```

A successful test prints `14` (double of 7) followed by `Endpoint Token Test Successful!`. If the endpoint is not reachable or the tokens are expired, an exception is raised instead.

---

## Step 2 — Run the Entry Point Locally

`appflx_entry_point.py` is the main server-side script. It supports two modes.

### Local test mode (no S3 required)

Loads configs directly from `sample_configs/` and skips all S3 uploads. Useful for verifying that the script and its dependencies work before deploying to ECS. Globus tokens still need to be valid — update `sample_configs/appfl_config.yaml` with current tokens before running.

> **💡 Note: The only thing you need to update is the tokens in the `sample_configs/appfl_config.yaml` file.**

```bash
python appflx_entry_point.py --local_test
```

The sample configs use the [FLamby](https://github.com/owkin/FLamby) Fed-Heart-Disease dataset with a simple linear model. The relevant files are:

| File | Purpose |
|------|---------|
| `sample_configs/appfl_config.yaml` | Server config (aggregator, scheduler, tokens, etc.) |
| `sample_configs/client.yaml` | Client config (endpoint ID, logging, etc.) |
| `sample_configs/model.py` | Simple linear baseline model |
| `sample_configs/loss.py` | BCE loss |
| `sample_configs/metric.py` | Binary accuracy metric |
| `sample_configs/dataloader.py` | FLamby Fed-Heart-Disease dataloader |

### Production mode (requires S3 access and valid Globus tokens)

Configs are downloaded from S3 and results are uploaded back after training. The `appfl_config.yaml` stored in S3 must contain valid `compute_token` and `openid_token` under `appflx_configs`.

```bash
python appflx_entry_point.py --base_dir <s3-base-dir>
```

To run an AI Data Readiness (AIDR) inspection instead of training:

```bash
python appflx_entry_point.py --base_dir <s3-base-dir> --run_aidr_only
```

---

## Step 3 — Docker

### Build and push the image

Run from the **repository root** (the Dockerfile copies the entire repo into the image):

> **💡 Note: Make sure you changed the token within src/appfl/service/sample_configs/appfl_config.yaml before building the docker image. If you forget, follow [this section](#mount-updated-sample-configs-at-runtime) to mount updated configuration.**

```bash
cd /path/to/appfl   # repo root
docker build \
    --platform=linux/amd64 \
    --no-cache \
    --progress=plain \
    -f ./src/appfl/service/Dockerfile \
    -t zilinghan/pytest \
    .
docker push zilinghan/pytest
```

### Test the container locally (no S3)

Use `--local_test` to exercise the full code path inside the container without any AWS credentials. Update `sample_configs/appfl_config.yaml` with valid Globus tokens before building, or mount the file at runtime (see below).

```bash
docker run --rm zilinghan/pytest --local_test
```

### Mount updated sample configs at runtime

If you want to test with different tokens/configs without rebuilding the image, mount the local `sample_configs/` directory over the one inside the container:

```bash
docker run --rm \
    -v ./src/appfl/service/sample_configs:/app/src/appfl/service/sample_configs \
    zilinghan/pytest --local_test
```

This is the recommended workflow for iterating on tokens: edit `sample_configs/appfl_config.yaml` locally, then re-run the above command — no rebuild needed.
