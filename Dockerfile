FROM python:3.9-slim

WORKDIR /

RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends \
    git \
    ffmpeg \
    python3-pip \
    build-essential \
    gcc \
    libpq-dev \
    make && \
    apt-get clean autoclean && \
    apt-get autoremove --yes

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "gunicorn", "-w1", "-b 0.0.0.0:8000", "-t 600", "app:app"]
