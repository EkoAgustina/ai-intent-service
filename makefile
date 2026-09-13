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

# Docker network
NETWORK_NAME = tunnel


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


# ============================================================
# Docker Image Build
# ============================================================

docker-build-distilbert:
	@echo "==> Building DistilBERT Docker image..."
	docker build \
		-t $(APP_NAME_DISTILBERT) \
		.

docker-build-albert:
	@echo "==> Building ALBERT Docker image..."
	docker build \
		-t $(APP_NAME_ALBERT) \
		.


# ============================================================
# Docker Container Run
# ============================================================

run-distilbert:
	@echo "==> Cleaning up old DistilBERT container..."
	docker rm -f $(APP_NAME_DISTILBERT) 2>/dev/null || true

	@echo "==> Starting DistilBERT container..."
	docker run -d \
		--name $(APP_NAME_DISTILBERT) \
		--network $(NETWORK_NAME) \
		-p $(PORT_DISTILBERT):$(PORT_DISTILBERT) \
		-v "$$(pwd)/model:/app/model" \
		--cpus="2.0" \
		--memory="2g" \
		--memory-swap="2g" \
		$(APP_NAME_DISTILBERT)

	@echo "==> DistilBERT is running on port $(PORT_DISTILBERT)"


run-albert:
	@echo "==> Cleaning up old ALBERT container..."
	docker rm -f $(APP_NAME_ALBERT) 2>/dev/null || true

	@echo "==> Starting ALBERT container..."
	docker run -d \
		--name $(APP_NAME_ALBERT) \
		--network $(NETWORK_NAME) \
		-p $(PORT_ALBERT):$(PORT_ALBERT) \
		-v "$$(pwd)/model:/app/model" \
		--cpus="2.0" \
		--memory="2g" \
		--memory-swap="2g" \
		$(APP_NAME_ALBERT)

	@echo "==> ALBERT is running on port $(PORT_ALBERT)"


# ============================================================
# Full Deployment
# ============================================================

deploy-distilbert: docker-build-distilbert run-distilbert
	@echo "==> DistilBERT deployment completed successfully."


deploy-albert: docker-build-albert run-albert
	@echo "==> ALBERT deployment completed successfully."


# ============================================================
# Convenience Targets
# ============================================================

deploy: deploy-distilbert deploy-albert

docker-build: docker-build-distilbert docker-build-albert

run: run-distilbert run-albert
