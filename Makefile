PYTHON ?= python
ORT_ROOT ?= $(CURDIR)/third_party/onnxruntime-linux-x64-1.19.2
.PHONY: setup smoke data train evaluate export test benchmark demo tensorrt cpp real lint
setup:
	$(PYTHON) -m pip install -e ".[dev]"
smoke:
	$(PYTHON) -m aerosurface.cli smoke
data:
	$(PYTHON) -m aerosurface.cli data
train:
	$(PYTHON) -m aerosurface.cli train
evaluate:
	$(PYTHON) -m aerosurface.cli evaluate
export:
	$(PYTHON) -m aerosurface.cli export
	$(PYTHON) -m aerosurface.cli parity
test:
	$(PYTHON) -m pytest -q
benchmark:
	$(PYTHON) -m aerosurface.cli benchmark
demo:
	$(PYTHON) -m aerosurface.cli demo
tensorrt:
	$(PYTHON) -m aerosurface.cli tensorrt
cpp:
	cmake -S . -B build -DONNXRUNTIME_ROOT=$(ORT_ROOT)
	cmake --build build --config Release
	ctest --test-dir build -C Release --output-on-failure
real:
	$(PYTHON) -m training.real_domain
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
