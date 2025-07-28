# Use an official base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .

# Run commands
RUN pip install -r requirements.txt
RUN pip uninstall pymupdf -y
RUN pip install pymupdf

# Create a non-root user with the same UID as your host user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser
RUN chown -R appuser:appuser /app

# Copy application code
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Define startup command
CMD ["python", "main.py"]