FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    nginx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Setup Nginx
RUN rm /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/nginx.conf
RUN chmod +x start.sh

# Required secrets for LangSmith (set in HF Space settings):
# LANGCHAIN_API_KEY
# LANGCHAIN_PROJECT

EXPOSE 7860

CMD ["./start.sh"]
