FROM python:3.10-slim
WORKDIR /workspace
COPY pyproject.toml ./
COPY aerosurface ./aerosurface
COPY simulation ./simulation
COPY training ./training
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e '.[dev]'
COPY . .
CMD ["python", "-m", "aerosurface.cli", "smoke", "--device", "cpu"]
