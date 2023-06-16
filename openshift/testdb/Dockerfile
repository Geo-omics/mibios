FROM image-registry.openshift-image-registry.svc:5000/openshift/debian:bookworm-slim
# FROM debian:bookworm-slim

ARG POSTGRES_BASE=/var/local/postgresql
ARG DEBIAN_FRONTEND=noninteractive 

RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime

# locale setup
RUN apt-get update && apt-get upgrade --assume-yes --quiet && apt-get install --assume-yes --no-install-recommends --quiet \
    locales \
    adduser
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen && update-locale
ENV LANG=en_US.utf8

# get us a postgres user
RUN addgroup --system --gid $APPLICATION_USER_ID postgres
RUN adduser --system --home=$POSTGRES_BASE --no-create-home --uid $APPLICATION_USER_ID --ingroup postgres postgres
RUN adduser postgres root

RUN apt-get update && apt-get install --assume-yes --no-install-recommends --quiet \
    acl \
    iproute2 \
    nano \
    nfs4-acl-tools \
    dnsutils \
    pipebench \
    postgresql-15 \
    procps

# replace data directory, normally that is /var/lib/postgres/15/main, which
# already has a initialized DB from package installation, we leave that in
# place and just substitute a new data directory, but leave the rest of the
# configuration alone.  The new place needs to have something mounted via
# docker-compose's volumes directive or docker-run's --volume option that
# contains a ready-to-use database or have initdb run manually.
ENV PGDATA=/tmp/pgdata
COPY start-testdb /
COPY pg_hba.conf /etc/postgresql/15/main/pg_hba.conf
RUN echo "data_directory = '$PGDATA'" >> /etc/postgresql/15/main/conf.d/local.conf
# network: database is our hostname via docker-compose, so we'll be listening at our IP address:
RUN echo "listen_addresses = '0.0.0.0'" >> /etc/postgresql/15/main/conf.d/local.conf
RUN echo "client_encoding = 'UTF8'" >> /etc/postgresql/15/main/conf.d/local.conf
RUN echo "max_wal_size = 10GB" >> /etc/postgresql/15/main/conf.d/local.conf
# RUN echo "shared_buffers = 10GB" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "work_mem = 100MB" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "maintenance_work_mem = 10GB" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "effective_io_concurrency = 50" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "max_worker_processes = 20" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "max_parallel_maintenance_workers = 10" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "max_parallel_workers_per_gather = 10" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "max_parallel_workers = 30" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "ssl_cert_file = '$POSTGRES_BASE/server.crt'" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "ssl_key_file = '$POSTGRES_BASE/server.key'" >> /etc/postgresql/15/main/conf.d/local.conf
# allow DB access from webapp between containers
##RUN echo "host glamr glamr_django glamr.gdick-web-app.svc.cluster.local trust" >> /etc/postgresql/15/main/pg_hba.conf
RUN echo "host glamr-public-test glamr_django localhost trust" >> /etc/postgresql/15/main/pg_hba.conf
RUN echo "host all all localhost trust" >> /etc/postgresql/15/main/pg_hba.conf
# allow SSL+password for development:
#RUN echo "hostssl all all all scram-sha-256" >> /etc/postgresql/15/main/pg_hba.conf

USER postgres
RUN mkdir -v -p -m 700 $PGDATA

EXPOSE 5432/tcp
HEALTHCHECK CMD pg_isready
STOPSIGNAL SIGINT
CMD ["/start-testdb"]
