# Use Python 3.10 base image (compatible with DeepFace/TensorFlow)
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install FROM python:3.10-slim

# ===============================
# SYSTEM DEPENDENCIES (CRITICAL)
# ===============================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# WORKDIR
# ===============================
WORKDIR /app

# ===============================
# PYTHON DEPENDENCIES
# ===============================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ===============================
# COPY APP FILES
# ===============================
COPY . .

# ===============================
# EXPOSE PORT (RENDER USES 10000)
# ===============================
EXPOSE 10000

# ===============================
# START SERVER
# ===============================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Expose port for Render
EXPOSE 10000

# Command to run your FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
