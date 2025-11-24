x🚀 UrbanWatch – Docker Usage Guide

This README explains how any team member can build and run the UrbanWatch API using Docker — no prior setup needed.

✅ 1. Prerequisites

You must have Docker Desktop installed and running.

macOS (Intel & M1/M2)

Download Docker Desktop:

https://www.docker.com/products/docker-desktop

Install & launch it

You should see the 🐳 whale in the top menu bar

Confirm it's running:

docker info

If logs display → ✅ Docker is ready.

📦 2. Files Required

Make sure these exist at the project root:

project_urban_watch/
├── Dockerfile
├── requirements.txt
├── .env   ✅ not committed, provided separately
└── urban_watch/
.env must contain:
SH_CLIENT_ID=your_client_id_here
SH_CLIENT_SECRET=your_secret_here

⚠️ Never push .env to GitHub.

🛠️ 3. Build the Docker Image

Run this from the project root:

docker build -t urbanwatch-api .

Verify the image exists:

docker images

You should see:

ur​banwatch-api   latest
▶️ 4. Run the API in Docker

Use this command:

docker run -it \
  --env-file .env \
  -p 8000:8000 \
  urbanwatch-api
What this does:

--env-file .env → loads SentinelHub credentials

-p 8000:8000 → exposes API on localhost

urbanwatch-api → runs the built image


🔍 5. Test the API

Open a browser:

http://localhost:8000

Expected response:

{"status": "UrbanWatch API running ✅"}

Prediction example:
http://localhost:8000/predict?x_min=5&y_min=43&x_max=5.1&y_max=43.1&date=2021-06-15

⚠️ Requires valid SentinelHub credentials.

🛑 6. Stop the Container

Press:

CTRL + C

Or from another terminal:

docker ps
docker stop <container_id>
🧹 7. Clean Up (optional)

Remove the image:

docker rmi urbanwatch-api
❗️ Troubleshooting
🔴 Error: Cannot connect to Docker daemon

Open Docker Desktop

Wait 10–20 seconds

Retry:

docker info
🔴 Error: invalid_client

Means:

.env missing or incorrect

Ask for valid SentinelHub keys

🔴 API not reachable

Check container logs:

docker ps

Then:

docker logs <container_id>
🔁 8. Restart Docker (if needed)
✅ Restart Docker Desktop on macOS

Click the 🐳 whale icon in the top menu bar

Select:

Quit Docker Desktop

Re-open Docker Desktop from Applications

Wait until the whale icon appears again

Verify:

docker info
✅ Restart a container (not full Docker)

List running containers:

docker ps

Stop a container:

docker stop <container_id>

Restart it:

docker start <container_id>
✅ You’re Ready!

Anyone can now: ✔ build the Docker image ✔ run the API locally ✔ test predictions ✔ restart Docker if needed ✔ without needing a Python environment 🎉

If code changes:

docker build -t urbanwatch-api .

(rebuild required)
