
.PHONY: help
help:
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } ' $(MAKEFILE_LIST)

.PHONY: docker-start
docker-up: ## Start Docker containers
	docker compose --env-file config.env up -d --build

.PHONY: docker-stop
docker-down: ## Stop Docker containers
	docker compose down

.PHONY: docker-clean
docker-clean: ## Remove Docker containers, networks, and volumes
	docker compose down --volumes --remove-orphans
	rm -rf .docker/*

.PHONY: install
install: ## Install Python dependencies
	uv sync --dev