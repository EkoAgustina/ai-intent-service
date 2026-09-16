# ============================================================
# Model Configuration
# ============================================================

# DistilBERT
BASE_MODEL_DISTILBERT = distilbert-base-uncased
MODEL_DIR_DISTILBERT = model/distilbert-banking77
APP_NAME_DISTILBERT = ai-intent-service-distilbert
PORT_DISTILBERT = 7000

# ALBERT
BASE_MODEL_ALBERT = albert-base-v2
MODEL_DIR_ALBERT = model/albert-banking77
APP_NAME_ALBERT = ai-intent-service-albert
PORT_ALBERT = 7001

# ELECTRA-small
BASE_MODEL_ELECTRA = google/electra-small-discriminator
MODEL_DIR_ELECTRA = model/electra-small-banking77
APP_NAME_ELECTRA = ai-intent-service-electra-small
PORT_ELECTRA = 7002



# ============================================================
# Docker Configuration
# ============================================================

IMAGE_NAME = ai-intent-service
IMAGE_TAG = latest

NETWORK_NAME = tunnel

CPU_LIMIT = 2.0
MEMORY_LIMIT = 3g


# ============================================================
# Model Training
# ============================================================

build-distilbert:
	@echo "==> Fine-tuning DistilBERT..."
	python3 src/train_model.py \
		--base-model $(BASE_MODEL_DISTILBERT) \
		--model-dir $(MODEL_DIR_DISTILBERT)


build-albert:
	@echo "==> Fine-tuning ALBERT..."
	python3 src/train_model.py \
		--base-model $(BASE_MODEL_ALBERT) \
		--model-dir $(MODEL_DIR_ALBERT)


build-electra:
	@echo "==> Fine-tuning ELECTRA-small..."
	python3 src/train_model.py \
		--base-model $(BASE_MODEL_ELECTRA) \
		--model-dir $(MODEL_DIR_ELECTRA)


# ============================================================
# Docker Image Build
# ============================================================

docker-build:
	@echo "==> Building Docker image..."
	docker build \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		.

	@echo "==> Docker image built successfully: $(IMAGE_NAME):$(IMAGE_TAG)"


# ============================================================
# Docker Container - DistilBERT
# ============================================================

run-distilbert:
	@echo "==> Cleaning up old DistilBERT container..."
	docker rm -f $(APP_NAME_DISTILBERT) 2>/dev/null || true

	@echo "==> Starting DistilBERT container..."
	docker run -d \
		--name $(APP_NAME_DISTILBERT) \
		--network $(NETWORK_NAME) \
		-p $(PORT_DISTILBERT):$(PORT_DISTILBERT) \
		-e TZ=Asia/Jakarta \
		-e OMP_NUM_THREADS=2 \
        -e MKL_NUM_THREADS=2 \
		-e MODEL_DIR=$(MODEL_DIR_DISTILBERT) \
		-e APP_PORT=$(PORT_DISTILBERT) \
		-v "$$(pwd)/model:/app/model" \
		--cpus="$(CPU_LIMIT)" \
		--memory="$(MEMORY_LIMIT)" \
		--memory-swap="$(MEMORY_LIMIT)" \
		$(IMAGE_NAME):$(IMAGE_TAG)

	@echo "==> DistilBERT is running on port $(PORT_DISTILBERT)"


# ============================================================
# Docker Container - ALBERT
# ============================================================

run-albert:
	@echo "==> Cleaning up old ALBERT container..."
	docker rm -f $(APP_NAME_ALBERT) 2>/dev/null || true

	@echo "==> Starting ALBERT container..."
	docker run -d \
		--name $(APP_NAME_ALBERT) \
		--network $(NETWORK_NAME) \
		-p $(PORT_ALBERT):$(PORT_ALBERT) \
		-e TZ=Asia/Jakarta \
		-e OMP_NUM_THREADS=2 \
        -e MKL_NUM_THREADS=2 \
		-e MODEL_DIR=$(MODEL_DIR_ALBERT) \
		-e APP_PORT=$(PORT_ALBERT) \
		-v "$$(pwd)/model:/app/model" \
		--cpus="$(CPU_LIMIT)" \
		--memory="$(MEMORY_LIMIT)" \
		--memory-swap="$(MEMORY_LIMIT)" \
		$(IMAGE_NAME):$(IMAGE_TAG)

	@echo "==> ALBERT is running on port $(PORT_ALBERT)"

#Docker Container - ELECTRA-small
run-electra:
	@echo "==> Cleaning up old ELECTRA-small container..."
	docker rm -f $(APP_NAME_ELECTRA) 2>/dev/null || true

	@echo "==> Starting ELECTRA-small container..."
	docker run -d \
		--name $(APP_NAME_ELECTRA) \
		--network $(NETWORK_NAME) \
		-p $(PORT_ELECTRA):$(PORT_ELECTRA) \
		-e TZ=Asia/Jakarta \
		-e OMP_NUM_THREADS=2 \
        -e MKL_NUM_THREADS=2 \
		-e MODEL_DIR=$(MODEL_DIR_ELECTRA) \
		-e APP_PORT=$(PORT_ELECTRA) \
		-v "$$(pwd)/model:/app/model" \
		--cpus="$(CPU_LIMIT)" \
		--memory="$(MEMORY_LIMIT)" \
		--memory-swap="$(MEMORY_LIMIT)" \
		$(IMAGE_NAME):$(IMAGE_TAG)

	@echo "==> ELECTRA-small is running on port $(PORT_ELECTRA)"

# ============================================================
# Full Deployment
# ============================================================

deploy: docker-build run-distilbert run-albert run-electra
	@echo "==> Full deployment completed successfully."


# ============================================================
# Convenience Targets
# ============================================================

build: docker-build

run: run-distilbert run-albert

deploy-distilbert: docker-build run-distilbert
	@echo "==> DistilBERT deployment completed successfully."

deploy-albert: docker-build run-albert
	@echo "==> ALBERT deployment completed successfully."
