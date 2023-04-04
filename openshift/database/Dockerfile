FROM debian:bookworm-slim

ARG POSTGRES_BASE=/var/local/postgresql
ARG POSTGRES_GID=130
ARG DEBIAN_FRONTEND=noninteractive 

RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime

# locale setup
RUN apt-get update && apt-get upgrade --assume-yes --quiet && apt-get install --assume-yes --no-install-recommends --quiet \
    locales \
    adduser
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen && update-locale
ENV LANG=en_US.utf8

# get us a postgres user
RUN addgroup --system --gid $POSTGRES_GID postgres
RUN adduser --system --home=$POSTGRES_BASE --no-create-home --uid $APPLICATION_USER_ID --ingroup postgres postgres
RUN adduser postgres root

RUN apt-get update && apt-get install --assume-yes --no-install-recommends --quiet \
    dnsutils \
    postgresql-15

# replace data directory, normally that is /var/lib/postgres/15/main, which
# already has a initialized DB from package installation, we leave that in
# place and just substitute a new data directory, but leave the rest of the
# configuration alone.  The new place needs to have something mounted via
# docker-compose's volumes directive or docker-run's --volume option that
# contains a ready-to-use database or have initdb run manually.
ENV PGDATA=$POSTGRES_BASE/data
RUN mkdir $POSTGRES_BASE && chown postgres:root $POSTGRES_BASE
USER postgres
# RUN mkdir --mode 00700 $PGDATA
RUN ls -laR $POSTGRES_BASE
RUN echo "data_directory = '$PGDATA'" >> /etc/postgresql/15/main/conf.d/local.conf
# network: database is our hostname via docker-compose, so we'll be listening at our IP address:
RUN echo "listen_addresses = '0.0.0.0'" >> /etc/postgresql/15/main/conf.d/local.conf
RUN echo "client_encoding = 'UTF8'" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "max_wal_size = 10GB" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "ssl_cert_file = '$POSTGRES_BASE/server.crt'" >> /etc/postgresql/15/main/conf.d/local.conf
#RUN echo "ssl_key_file = '$POSTGRES_BASE/server.key'" >> /etc/postgresql/15/main/conf.d/local.conf
# allow DB access from webapp between containers
RUN echo "host glamr glamr_django glamr.gdick-web-app.svc.cluster.local trust" >> /etc/postgresql/15/main/pg_hba.conf
RUN echo "host glamr glamr_django all trust" >> /etc/postgresql/15/main/pg_hba.conf
# allow SSL+password for development:
#RUN echo "hostssl all all all scram-sha-256" >> /etc/postgresql/15/main/pg_hba.conf

EXPOSE 5432/tcp
HEALTHCHECK CMD pg_isready
STOPSIGNAL SIGINT
CMD ["/usr/lib/postgresql/15/bin/postgres", "--config-file=/etc/postgresql/15/main/postgresql.conf"]
