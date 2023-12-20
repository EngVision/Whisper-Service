FROM python:3.11-slim

WORKDIR /

COPY . .

RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install \
 -y --no-install-recommends git ffmpeg python3-pip && \
 apt-get clean autoclean && apt-get autoremove --yes

RUN pip3 install -r requirements.txt

RUN python3 load_model.py

EXPOSE 8000

CMD [ "gunicorn", "-w1", "-b 0.0.0.0:8000", "-t 600", "app:app"]
