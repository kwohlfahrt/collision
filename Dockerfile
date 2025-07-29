FROM ubuntu:24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/sharing=locked <<EOF
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y ca-certificates curl
ARCH=$(dpkg --print-architecture)
if [ "$ARCH" = "arm64" ]; then
    ARCH=sbsa
fi
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/$ARCH/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
EOF

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/sharing=locked <<EOF
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y libnvidia-compute-570 clinfo python3-venv
EOF

ENV PATH=/app/venv/bin:${PATH}

WORKDIR /app

RUN <<EOF
python3 -m venv ./venv
mkdir collision
touch collision/__init__.py
EOF

COPY requirements.txt setup.py MANIFEST.in ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements.txt && pip install --no-deps -e .[test]

COPY collision ./collision
COPY tests ./tests
