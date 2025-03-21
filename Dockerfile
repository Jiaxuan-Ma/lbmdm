# app/Dockerfile

FROM python:3.9

WORKDIR /MLMD

COPY . .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/&& pip install --upgrade pip&& pip3 install -r requirements.txt

CMD ["streamlit", "run", "--server.port", "8500", "dmmd.py"]
