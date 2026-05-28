APP_NAME = ai-intent-service
PORT = 7000
NETWORK_NAME = tunnel

# The main command to deploy whenever there is a code change
deploy: build run

build:
	@echo "==> Building image..."
	docker build -t $(APP_NAME) .

run:
	@echo "==> Cleaning up old container & running new one..."
	docker rm -f $(APP_NAME) || true
	docker run -d \
		--name $(APP_NAME) \
		--network $(NETWORK_NAME) \
		-p $(PORT):$(PORT) \
		-v "$$(pwd)/model:/app/model" \
		$(APP_NAME)
	@echo "==> Success! Application is running on port $(PORT)"

logs:
	docker logs -f $(APP_NAME)
