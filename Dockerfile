FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Render, PORT ortam degiskenini calisma zamaninda atar; bu yuzden komutu
# shell formunda yazip $PORT'un genisletilmesini sagliyoruz. Sabit port
# yazarsak Render'in yonlendirdigi port ile uyusmayabilir.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
