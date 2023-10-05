FROM debian:bookworm-slim
ENV APP_BASE_DIR=/glamr-webapp
ARG APP_USER=webapp
ARG APP_GROUP=webapp

USER root

ARG DEBIAN_FRONTEND=noninteractive 
RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime
RUN apt-get update && apt-get upgrade --assume-yes --quiet && apt-get install --assume-yes --no-install-recommends --quiet \
    adduser \
    curl \
    bind9-dnsutils \
    git \
    iproute2 \
    iputils-ping \
    postgresql-client \
    procps \
    python3-biopython \
    python3-defusedxml \
    python3-django \
    python3-django-crispy-forms \
    python3-django-extensions \
    python3-djangorestframework \
    python3-django-filters \
    python3-django-tables2 \
    python3-matplotlib \
    python3-pandas \
    python3-pip \
    python3-psycopg2 \
    python3-pygraphviz \
    python3-xlrd \
    python3-zipstream \
    uwsgi \
    uwsgi-plugin-python3

RUN echo "increment for mibios upgrade: 60"
# this is Debian's pip so the below cmd line with --b-s-p will install mibios
# (and nothing else) under /usr/local
# Also, currently, just "pip install" would install for python3.10, hence
# invoke via "python3.11 -m pip"  // unsure if the --python option does
# anything.
RUN python3.11 -m pip install --no-index --no-deps --break-system-packages --python /usr/bin/python3.11 git+https://github.com/Geo-omics/mibios.git@testing

COPY uwsgi.ini /etc/uwsgi.ini

# openshift must supply APPLICATION_USER_ID
RUN adduser --system --home=$APP_BASE_DIR --group $APP_USER
RUN adduser $APP_USER root

WORKDIR ${APP_BASE_DIR}
COPY settings.py start-webapp ./
RUN chmod ugo+rwx+t $APP_BASE_DIR
USER $APP_USER:$APP_GROUP
RUN mkdir static_var

RUN manage_mibios collectstatic --no-input

EXPOSE 3032/tcp
HEALTHCHECK CMD echo | curl --silent telnet://127.0.0.1:9191 | grep -q accepting
CMD sh -c "exec $APP_BASE_DIR/start-webapp"
