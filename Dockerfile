# ----------------------------
# Stage 1 - React Dev Frontend
# ----------------------------
FROM node:22.14.0-slim as frontend

WORKDIR /app/frontend
COPY UI/package*.json ./
RUN npm install
COPY UI/ ./
CMD ["npm", "start"]
    
# ----------------------------
# Stage 2 - Flask Backend with venv
# ----------------------------
FROM python:3.13.3-slim as backend

WORKDIR /app/backend
    
ENV PYTHONPATH=/app/backend
ENV PATH="/app/venv/bin:$PATH"
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
    
RUN python -m venv /app/venv
    
COPY Backend/requirements.txt ./
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt
    
COPY Backend/ ./
    
CMD ["flask", "run", "--host=0.0.0.0"]