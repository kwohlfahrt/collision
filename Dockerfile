FROM ubuntu:24.04 AS build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/sharing=locked <<EOF
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
EOF

ENV PATH=/app/venv/bin:${PATH}

WORKDIR /app

RUN <<EOF
python3 -m venv ./venv
mkdir collision
touch collision/__init__.py
EOF

COPY requirements.txt pyproject.toml MANIFEST.in ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements.txt && pip install --no-deps -e .[test]

COPY collision ./collision
COPY tests ./tests

FROM ubuntu:24.04 AS nvidia

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
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 libnvidia-compute-570 clinfo
EOF

COPY --link --from=build /app /app

WORKDIR /app

ENV PATH=/app/venv/bin:${PATH}

FROM ubuntu:24.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/sharing=locked <<EOF
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 pocl-opencl-icd clinfo
EOF

COPY --link --from=build /app /app

WORKDIR /app

ENV PATH=/app/venv/bin:${PATH}
