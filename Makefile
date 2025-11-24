# Build the Docker image
build:
	docker build -t urbanwatch-api .

# Run the container
run:
	docker run -it --env-file .env -p 8000:8000 urbanwatch-api

# Stop all running containers of this image
stop:
	docker stop $$(docker ps -q --filter ancestor=urbanwatch-api) || true
