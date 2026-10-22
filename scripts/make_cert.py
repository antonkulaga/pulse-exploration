"""Make a self-signed TLS certificate so phones can open the tier "capture" page over HTTPS.

Chrome only gives web pages the camera (getUserMedia) in a secure context, and a LAN address over plain
HTTP is not one. This writes data/certs/key.pem and data/certs/cert.pem (data/ is gitignored) with a
subjectAltName for the laptop's LAN IP and localhost, using the openssl CLI, then prints the uvicorn
command that serves the same app over HTTPS on port 8766, next to the plain-HTTP server on 8765.

    uv run python scripts/make_cert.py            # keeps an existing certificate
    uv run python scripts/make_cert.py --force    # makes a new one
    uv run python scripts/make_cert.py --ip 192.168.1.50 --ip 10.0.0.2

Phones show a certificate warning once per browser profile; tap Advanced, then Proceed.
"""

import argparse
import ipaddress
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "certs"
DEFAULT_IPS = ["192.168.1.177"]
HTTP_PORT = 8765
HTTPS_PORT = 8766
DAYS = 397  # stay under the 398-day limit browsers apply to server certificates

CONFIG = """\
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no

[dn]
CN = pulse-exploration local prototype
O = pulse-exploration self-signed LAN only

[v3]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectKeyIdentifier = hash
subjectAltName = @alt

[alt]
{alt}
"""


def alt_names(ips: list[str], dns: list[str]) -> str:
    lines = [f"IP.{i} = {ip}" for i, ip in enumerate(ips, start=1)]
    lines += [f"DNS.{i} = {name}" for i, name in enumerate(dns, start=1)]
    return "\n".join(lines)


def run(cmd: list[str]) -> str:
    done = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return done.stdout


def local_ips() -> set[str]:
    """IPv4 addresses of this machine, best effort (empty if `hostname -I` is unavailable)."""
    try:
        return set(run(["hostname", "-I"]).split())
    except (OSError, subprocess.CalledProcessError):
        return set()


def make_cert(out: Path, ips: list[str], force: bool) -> tuple[Path, Path, bool]:
    key = out / "key.pem"
    cert = out / "cert.pem"
    if key.exists() and cert.exists() and not force:
        return key, cert, False
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cnf = Path(tmp) / "openssl.cnf"
        cnf.write_text(CONFIG.format(alt=alt_names(ips, ["localhost"])))
        run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", str(DAYS),
            "-keyout", str(key), "-out", str(cert), "-config", str(cnf),
        ])
    key.chmod(0o600)
    return key, cert, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ip", action="append", help=f"LAN IP for the certificate (repeatable; default {DEFAULT_IPS[0]})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--force", action="store_true", help="replace an existing certificate")
    args = parser.parse_args(argv)

    if shutil.which("openssl") is None:
        print("openssl not found on PATH", file=sys.stderr)
        return 1
    ips = args.ip or list(DEFAULT_IPS)
    for ip in ips:
        ipaddress.ip_address(ip)  # raises ValueError on a typo
    if "127.0.0.1" not in ips:
        ips.append("127.0.0.1")

    key, cert, made = make_cert(args.out.resolve(), ips, args.force)
    print(("Wrote" if made else "Kept existing (use --force to replace)") + f": {cert}\n           and {key}")
    print(run(["openssl", "x509", "-in", str(cert), "-noout", "-subject", "-enddate", "-ext", "subjectAltName"]).rstrip())
    print(run(["openssl", "x509", "-in", str(cert), "-noout", "-fingerprint", "-sha256"]).rstrip())

    mine = local_ips()
    lan = [ip for ip in ips if ip != "127.0.0.1"]
    if mine and not any(ip in mine for ip in lan):
        print(f"\nNote: this machine's addresses are {' '.join(sorted(mine))}; none is in the certificate."
              " Re-run with --ip <address> --force if the laptop's IP changed.")

    lan_ip = lan[0] if lan else "127.0.0.1"
    print(f"""
Serve HTTPS on port {HTTPS_PORT} next to the plain-HTTP server on {HTTP_PORT} (run in a second terminal):

  cd {ROOT} && uv run uvicorn pulse_exploration.server:app --host 0.0.0.0 --port {HTTPS_PORT} --ssl-keyfile {key} --ssl-certfile {cert}

On the phone (same Wi-Fi):
  tier video   http://{lan_ip}:{HTTP_PORT}/upload
  tier capture https://{lan_ip}:{HTTPS_PORT}/capture   (Chrome warns once: Advanced, then Proceed)""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
