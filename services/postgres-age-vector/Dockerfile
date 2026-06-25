# PostgreSQL 18 with pgvector + Apache AGE.
#
# Build deps are installed, used to compile AGE, then purged in the same
# layer so the final image only carries the runtime artefacts (~250 MB
# saved vs. an unpurged build).

ARG PG_MAJOR=18
ARG PGVECTOR_TAG=0.8.3-pg18-trixie
ARG AGE_TAG=PG18/v1.7.0-rc0

FROM pgvector/pgvector:${PGVECTOR_TAG}

ARG PG_MAJOR
ARG AGE_TAG

RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        postgresql-server-dev-${PG_MAJOR} \
        libreadline-dev \
        zlib1g-dev \
        flex \
        bison \
    && git clone --depth 1 --branch "${AGE_TAG}" https://github.com/apache/age.git /tmp/age \
    && make -C /tmp/age \
    && make -C /tmp/age install \
    && rm -rf /tmp/age \
    && apt-get purge -y --auto-remove \
        build-essential \
        git \
        postgresql-server-dev-${PG_MAJOR} \
        libreadline-dev \
        zlib1g-dev \
        flex \
        bison \
    && rm -rf /var/lib/apt/lists/*

# Initialise extensions on first cluster start.
RUN printf '%s\n' \
        'CREATE EXTENSION IF NOT EXISTS vector;' \
        'CREATE EXTENSION IF NOT EXISTS age;' \
        "SET search_path = ag_catalog, public;" \
    > /docker-entrypoint-initdb.d/01-extensions.sql
