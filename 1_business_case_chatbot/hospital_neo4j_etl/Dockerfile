# Use python distribution
FROM python:3.11-slim

WORKDIR /app

# Copy contents from source folder into the /app directory within the container
COPY ./src/ /app

# Install dependencies
COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

# Run entrypoint
CMD ["sh", "entrypoint.sh"]