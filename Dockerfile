# Use a more recent Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements, app, and data
COPY requirements.txt .
COPY streamlit_eda.py .
COPY data/ ./data/
COPY docs/ ./docs/


# Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "--server.address", "0.0.0.0", "streamlit_eda.py"]