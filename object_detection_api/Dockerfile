FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir --upgrade -i https://pypi.org/simple/ -r /app/requirements.txt

COPY ./app /app/app