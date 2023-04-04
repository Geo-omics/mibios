FROM debian:bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive 
RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime
RUN apt-get update && apt-get upgrade --assume-yes && apt-get install --assume-yes --no-install-recommends \
    bind9-dnsutils \
    git \
    iputils-ping \
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
    python3-xlrd \
    python3-zipstream \
    uwsgi \
    uwsgi-plugin-python3

# this is Debian's pip so the below cmd line with --b-s-p will install mibios
# (and nothing else) under /usr/local
# Also, currently, just "pip install" would install for python3.10, hence
# invoke via "python3.11 -m pip"  // unsure if the --python option does
# anything.
RUN echo "increment for mibios upgrade: 7"
RUN python3.11 -m pip install --no-index --no-deps --break-system-packages --python /usr/bin/python3.11 git+https://github.com/Geo-omics/mibios.git

COPY uwsgi.ini /etc/uwsgi.ini
RUN mkdir /glamr && chown www-data:www-data /glamr

USER www-data:www-data
WORKDIR /glamr
RUN mkdir static_var && chown www-data:www-data static_var
COPY settings.py start-webapp ./
RUN    manage_mibios collectstatic --no-input

EXPOSE 3032/tcp
CMD ["/glamr/start-webapp"]
