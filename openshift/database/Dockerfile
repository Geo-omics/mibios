FROM image-registry.openshift-image-registry.svc:5000/openshift/debian:bookworm-slim

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
    less \
    nano \
    nfs4-acl-tools \
    dnsutils \
    pipebench \
    postgresql-15 \
    procps \
    screen \
    time

COPY start-database stop-database $POSTGRES_BASE/
COPY pg_hba.conf restore_mode.conf /etc/postgresql/15/main/
COPY local.conf /etc/postgresql/15/main/conf.d/
RUN sed -i 's/^data_directory = .*//' /etc/postgresql/15/main/postgresql.conf
RUN cp /etc/postgresql/15/main/postgresql.conf /etc/postgresql/15/main/postgresql_restore.conf
RUN [ -n $DATABASE_RESTORE ] && echo "include = '/etc/postgresql/15/main/restore_mode.conf'" >> /etc/postgresql/15/main/postgresql.conf
ENV PGDATA_ROOT=$POSTGRES_BASE/pgdata_mnt/data
RUN env

USER postgres
WORKDIR $POSTGRES_BASE
EXPOSE 5432/tcp
HEALTHCHECK CMD pg_isready
STOPSIGNAL SIGINT
CMD ["./start-database"]
