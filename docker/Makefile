# ============================================================
#  LLM Preference Unlearning — Unified Build & Deployment Makefile
#  Author: Jacob Ramey (rameyjm7)
#  Target environment: Docker + Apptainer (GPU)
# ============================================================

# ---------- Variables ----------
DOCKER_USER = rameyjm7
IMAGE_NAME  = llm-preference-unlearning
TAG         = latest
PORT        = 8888
TOKEN       = tml
SIF_NAME    = llm_preference_unlearning.sif

# ---------- Build the Docker image ----------
build:
	@echo "\nBuilding Docker image..."
	docker build -t $(DOCKER_USER)/$(IMAGE_NAME):$(TAG) .

# ---------- Push image to Docker Hub ----------
push:
	@echo "\nPushing image to Docker Hub..."
	docker push $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)

# ---------- Run locally with GPU + Jupyter ----------
run:
	@echo "\nLaunching Jupyter Lab on http://localhost:$(PORT)"
	docker compose up

# ---------- Stop local container ----------
stop:
	@echo "\nStopping Docker Compose..."
	docker compose down

# ---------- Build Apptainer image from Docker Hub ----------
sif:
	@echo "\nBuilding Apptainer .sif from Docker Hub..."
	apptainer build $(SIF_NAME) docker://$(DOCKER_USER)/$(IMAGE_NAME):$(TAG)

# ---------- Run analysis script inside Apptainer ----------
apptainer-run:
	@echo "\nRunning activation unlearning inside Apptainer..."
	apptainer exec --nv $(SIF_NAME) python3 activation_unlearning.py

# ---------- Launch Jupyter inside Apptainer ----------
apptainer-lab:
	@echo "\nLaunching Jupyter Lab inside Apptainer..."
	apptainer exec --nv $(SIF_NAME) \
		jupyter lab --ip=0.0.0.0 --port=$(PORT) --no-browser --allow-root --NotebookApp.token=$(TOKEN)

# ---------- Clean local Docker artifacts ----------
clean:
	@echo "\nCleaning up local Docker images..."
	docker image rm $(DOCKER_USER)/$(IMAGE_NAME):$(TAG) -f || true

# ---------- Display help ----------
help:
	@echo "\nLLM Preference Unlearning Makefile Commands:"
	@echo "  make build           Build Docker image"
	@echo "  make push            Push image to Docker Hub"
	@echo "  make run             Launch Jupyter Lab via Docker Compose"
	@echo "  make stop            Stop Docker Compose session"
	@echo "  make sif             Build Apptainer image from Docker Hub"
	@echo "  make apptainer-run   Run activation_unlearning.py inside Apptainer"
	@echo "  make apptainer-lab   Run Jupyter Lab inside Apptainer"
	@echo "  make clean           Remove local Docker image"
