ARG IMAGE_NAME=python
ARG IMAGE_TAG=3.9

FROM ${IMAGE_NAME}:${IMAGE_TAG} as tlt-dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    make \
    pandoc \
    python3.9-dev
