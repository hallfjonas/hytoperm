FROM python:3

# Set the working directory
WORKDIR /app

# Copy the setup file
COPY setup.py .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Build the image
# docker build -t hytoperm .

# Run the container
# docker run -it hytoperm
