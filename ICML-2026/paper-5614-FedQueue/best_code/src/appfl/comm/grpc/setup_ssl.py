import os
import pathlib
import re
import subprocess

_SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]*$")


def setup_ssl():
    """
    Command line interface for creating SSL certificate signed by a private
    certificate authority (CA), and storing the certificate and private key
    in the specified directory.
    """
    # Prompt user for the directory to store the SSL certificate and private key
    while True:
        default_ssl_dir = os.path.join(pathlib.Path.home(), ".appfl", "ssl")
        ssl_dir = input(
            f"Enter the absolute path of the directory where the SSL certificate and private key will be stored, press Enter to use the default directory {default_ssl_dir}: "
        )
        if not ssl_dir:
            ssl_dir = default_ssl_dir
        ssl_dir = os.path.abspath(os.path.expanduser(ssl_dir))
        if not _SAFE_PATH_RE.match(ssl_dir):
            print(
                "Invalid directory: only absolute paths containing letters, "
                "digits, '.', '_', '-', and '/' are allowed. Please try again."
            )
            continue
        try:
            if not os.path.exists(ssl_dir):
                pathlib.Path(ssl_dir).mkdir(parents=True, exist_ok=True)
            if not os.path.isdir(ssl_dir):
                raise NotADirectoryError(ssl_dir)
            for f in os.listdir(ssl_dir):
                os.remove(os.path.join(ssl_dir, f))
        except OSError as e:
            print(f"Invalid directory ({e}), please try again")
            continue
        break

    # Default values
    default_C = "US"
    default_ST = "Illinois"
    default_O = "APPFL"
    default_DNS = "localhost"
    default_IP = "127.0.0.1"

    # Prompt user for C, ST, O, CN, DNS, IP, with default values
    C = (
        input(f"Enter Country Code, press Enter to use default '{default_C}': ")
        or default_C
    )
    ST = (
        input(f"Enter State, press Enter to use default '{default_ST}': ") or default_ST
    )
    ORG = (
        input(f"Enter Organization (O), press Enter to use default '{default_O}': ")
        or default_O
    )
    DNS = (
        input(f"Enter DNS (DNS.1), press Enter to use default '{default_DNS}': ")
        or default_DNS
    )
    IP = input(f"Enter IP, press Enter to use default '{default_IP}': ") or default_IP

    # Create the configuration file content
    conf_content = f"""
[req]
default_bits = 4096
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn

[dn]
C = {C}
ST = {ST}
O = {ORG}
CN = {DNS}

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = {DNS}
IP.1 = {IP}
    """

    # Write the configuration file
    conf_file = os.path.join(ssl_dir, "certificate.conf")
    with open(conf_file, "w") as f:
        f.write(conf_content)

    # Certificate generation script
    script_content = f"""#!/bin/bash
set -e
cd "$( cd "$( dirname "{{{{BASH_SOURCE[0]}}}}" )" >/dev/null 2>&1 && pwd )"

CA_PASSWORD=notsafe

CERT_DIR={ssl_dir}

# Generate the root certificate authority key and certificate based on key
openssl genrsa -out $CERT_DIR/ca.key 4096
openssl req \\
    -new \\
    -x509 \\
    -key $CERT_DIR/ca.key \\
    -sha256 \\
    -subj "/C={C}/ST={ST}/O={ORG}" \\
    -days 365 -out $CERT_DIR/ca.crt

# Generate a new private key for the server
openssl genrsa -out $CERT_DIR/server.key 4096

# Create a signing CSR
openssl req \\
    -new \\
    -key $CERT_DIR/server.key \\
    -out $CERT_DIR/server.csr \\
    -config {conf_file}

# Generate a certificate for the server
openssl x509 \\
    -req \\
    -in $CERT_DIR/server.csr \\
    -CA $CERT_DIR/ca.crt \\
    -CAkey $CERT_DIR/ca.key \\
    -CAcreateserial \\
    -out $CERT_DIR/server.crt \\
    -days 365 \\
    -sha256 \\
    -extfile {conf_file} \\
    -extensions req_ext
    """

    script_file = os.path.join(ssl_dir, "generate_ssl.sh")
    with open(script_file, "w") as f:
        f.write(script_content)

    os.chmod(script_file, 0o700)
    try:
        subprocess.run([script_file], check=True)
        print_str = f"Please copy the CA certificate {os.path.join(ssl_dir, 'ca.crt')} to the client machines"
        print("=" * len(print_str))
        print(f"SSL certificate stored in {os.path.join(ssl_dir, 'server.crt')}")
        print(f"SSL private key stored in {os.path.join(ssl_dir, 'server.key')}")
        print(f"CA certificate stored in  {os.path.join(ssl_dir, 'ca.crt')}")
        print(print_str)
        print("=" * len(print_str))
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
