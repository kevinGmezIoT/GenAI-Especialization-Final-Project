#!/bin/bash

# Start FastAPI in the background
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Start Streamlit in the background
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true &

# Start Nginx in the foreground
nginx -g 'daemon off;'
