import time
import random
import requests
import os


class SentinelHubRateLimitedClient:
    def __init__(
        self,
        client_id=None,
        client_secret=None,
        auth_url="https://services.sentinel-hub.com/oauth/token",
        min_interval=7,
        max_jitter=2,
        max_retries=5
    ):
        # Load keys automatically from OS environment if not passed
        self.client_id = client_id or os.getenv("SENTINEL_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SENTINEL_CLIENT_SECRET")
        self.auth_url = auth_url

        if not self.client_id or not self.client_secret:
            raise ValueError("⚠️ You must set SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET.")

        # Rate limit
        self.min_interval = min_interval
        self.max_jitter = max_jitter
        self.max_retries = max_retries
        self.last_call = 0

        # Auth
        self.token = None
        self.token_expiry = 0  # unix timestamp
        self.get_token()
 
# TOKEN MANAGEMENT

    def get_token(self):
        """Fetch OAuth2 token from Sentinel Hub."""
        print("🔑 Fetching Sentinel Hub token…")

        response = requests.post(
            self.auth_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )

        if response.status_code != 200:
            raise Exception(f"❌ Auth error: {response.status_code} {response.text}")

        data = response.json()
        self.token = data["access_token"]
        self.token_expiry = time.time() + data["expires_in"] - 60  # renew 1 min early

        print("✅ Token acquired.")

    def ensure_token(self):
        """Refresh token automatically if expired."""
        if time.time() >= self.token_expiry:
            self.get_token()

# RATE LIMIT

    def wait_turn(self):
        elapsed = time.time() - self.last_call
        required = self.min_interval + random.uniform(0, self.max_jitter)

        if elapsed < required:
            time.sleep(required - elapsed)


# REQUEST WRAPPER

    def request(self, method, url, **kwargs):
        self.ensure_token()

        if "headers" not in kwargs:
            kwargs["headers"] = {}

        kwargs["headers"]["Authorization"] = f"Bearer {self.token}"

        retries = 0

        while True:
            self.wait_turn()
            response = requests.request(method, url, **kwargs)
            self.last_call = time.time()

            if response.status_code == 429:
                if retries >= self.max_retries:
                    raise Exception("❌ Too many retries (429).")
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"⏳ 429 Too Many Requests → waiting {retry_after} sec…")
                time.sleep(retry_after)
                retries += 1
                continue

            return response
