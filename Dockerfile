FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1

ENV PYTHONUNBUFFERED=1 

RUN apt-get update && \
    apt-get install -y wget build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libffi-dev libncurses5-dev libncursesw5-dev \
    liblzma-dev libgdbm-dev libnss3-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY . . 

EXPOSE 8000

CMD ["uvicorn", "app.FastApi:app", "--host", "0.0.0.0", "--port", "8000"]