.PHONY: help install test sim server lint clean

help:
	@echo "Oyster Phone Training Protocol - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make sim        - Run simulation with 10 clients, 3 rounds"
	@echo "  make server     - Start Flower server on port 8080"
	@echo "  make lint       - Run code quality checks (black + ruff)"
	@echo "  make clean      - Remove Python cache and build artifacts"

install:
	pip install -q flwr torch transformers peft numpy pyyaml msgpack-python3 pytest
	@echo "✓ Dependencies installed"

test:
	@echo "Running all tests..."
	PYTHONPATH=. pytest tests/ -v --tb=short
	@echo "✓ Tests passed"

sim:
	@echo "Running simulation with 10 clients, 3 rounds..."
	PYTHONPATH=. python3 simulation/sim_orchestrator.py --clients 10 --rounds 3

server:
	@echo "Starting Flower server on port 8080..."
	PYTHONPATH=. python3 server/flower_server.py --port 8080 --rounds 100

lint:
	@echo "Running code quality checks..."
	@if command -v black >/dev/null 2>&1; then \
		black --check .; \
	else \
		echo "black not found, skipping..."; \
	fi
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
	else \
		echo "ruff not found, skipping..."; \
	fi
	@echo "✓ Linting complete"

clean:
	@echo "Cleaning Python cache and artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache 2>/dev/null || true
	@echo "✓ Clean complete"
