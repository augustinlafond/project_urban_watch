# Build the Docker image
build:
	docker build \
  --platform linux/amd64 \
  -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/urbanwatch/$GAR_IMAGE:prod .

# Run the container
run:
	docker run -it --env-file .env -p 8000:8000 urbanwatch-api

# push the container
push:
	docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/urbanwatch/$GAR_IMAGE:prod

# Stop all running containers of this image
stop:
	docker stop $$(docker ps -q --filter ancestor=urbanwatch-api) || true
