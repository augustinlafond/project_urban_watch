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

# Redeploy the Docker deploy and GC
deploy:
	docker build --platform linux/amd64 -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/urbanwatch/${GAR_IMAGE}:prod .
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/urbanwatch/${GAR_IMAGE}:prod
	gcloud run deploy --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/urbanwatch/${GAR_IMAGE}:prod --memory ${GAR_MEMORY} --region ${GCP_REGION} --env-vars-file .env.yaml
