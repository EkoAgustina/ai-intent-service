# Variabel untuk memudahkan perubahan nama
APP_NAME = ai-intent-service
PORT = 7000

# Default command (will run if you just type 'make')
all: build run

# Building a Docker image
build:
	docker build -t $(APP_NAME) .

# Running containers
run:
	docker run -d \
		--name $(APP_NAME) \
		-p $(PORT):$(PORT) \
		-v "$$(pwd)/model:/app/model" \
		$(APP_NAME)
