# The builder image, used to build the virtual environment
FROM python:3.10-bookworm as builder

RUN pip install poetry==1.5.1

# - Poetry runs without asking for user interaction
# - Virtual environment located within web_service/ directory (fully self-contained)
# - Create virtual environment inside Docker container
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

WORKDIR /app

# Necessary files for Poetry
COPY pyproject.toml poetry.lock ./
RUN touch README.md

# - Avoid installing development dependencies (linters, tests,...)
# - Avoid installing the current project into the virtual environment (VE)
# - Disables Poetry source caches (we only install packages once)
RUN poetry install --without dev --no-root --no-cache

# The runtime image, used to just run the code provided its VE
FROM python:3.10-slim-bookworm as runtime

# Install libgomp1
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# - Path to VE (Python convention) set to VE directory within the Docker container
# - Prepend '/app/.venv/bin' to the PATH. Now when running 'python', it looks in the VE's bin directory first
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copies the contents of the VE created in the builder to the same location in the runtime
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app

COPY predict.py sample_kickstarter_project.json ./