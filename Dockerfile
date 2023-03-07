
FROM python:3.7
WORKDIR /app
# Intalling Python dependencies
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
# Copies all files in current directory to Image
COPY . /app
EXPOSE 6000
#CMD ["python3","app.py"]
CMD flask run --host 0.0.0.0 --port 6000


##  flask app > git > deploy on local server>  docker > aws 