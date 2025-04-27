# Use an official lightweight Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy only the important files
COPY app.py .
COPY ideal_model.pkl .
COPY trained_column.pkl .
COPY preprocessing.py .

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses
EXPOSE 8000

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]