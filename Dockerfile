# use snap latest/stable version for typst and draw.io
ARG UV_VERSION=0.13.7
ARG TYPST_VERSION=0.15.1
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM ghcr.io/typst/typst:${TYPST_VERSION} AS typst

FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=typst /bin/typst /usr/local/bin/

ARG DRAWIO_VERSION=32.6.10
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget \
 && wget -O /tmp/drawio.deb \
    "https://github.com/jgraph/drawio-desktop/releases/download/v${DRAWIO_VERSION}/drawio-amd64-${DRAWIO_VERSION}.deb" \
 && apt-get install -y --no-install-recommends \
    git \
    python3 \
    make \
    fonts-stix \
    poppler-utils \
    /tmp/drawio.deb xvfb \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/* /tmp/drawio.deb \
 && uv cache clean
