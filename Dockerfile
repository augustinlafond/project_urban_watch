####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########

FROM python:3.10.6-buster

WORKDIR /prod

# install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy project package
COPY urban_watch urban_watch

EXPOSE 8000

# launch API
CMD ["uvicorn", "urban_watch.api.fast_api:app", "--host", "0.0.0.0", "--port", "8000"]