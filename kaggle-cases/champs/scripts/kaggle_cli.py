#!/usr/bin/env python3
"""Run pinned Kaggle CLI with all account files in the selected project folder.

Kaggle CLI 2.2.4 honors KAGGLE_CONFIG_DIR for legacy keys, but its SDK's
OAuth credentials default to ~/.kaggle/credentials.json. Redirect that default
without changing HOME, credentials, OAuth scopes, or the authorization flow.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_auth_directory():
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(ROOT / ".kaggle"))).expanduser().resolve()
    config.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config)
    from kagglesdk.kaggle_creds import KaggleCredentials
    credentials = str(config / "credentials.json")
    KaggleCredentials.DEFAULT_CREDENTIALS_FILE = credentials
    # save/delete bind their path at function definition time in SDK 0.1.37.
    # load reads DEFAULT_CREDENTIALS_FILE at call time.
    KaggleCredentials.save.__defaults__ = (credentials,)
    KaggleCredentials.delete.__defaults__ = (credentials,)
    return config


def main():
    configure_auth_directory()
    # SDK 0.1.37 allows logged-out `auth login` only for recognized CLI entry
    # points. This wrapper is a CLI entry point, not a library client.
    sys.argv[0] = str(ROOT / "envs/kaggle-cli/bin/kaggle")
    from kaggle.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
