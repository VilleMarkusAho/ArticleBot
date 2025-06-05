# ----------------------------
# Stage 1 - React Dev Frontend
# ----------------------------
FROM node:22.14.0-slim AS frontend

WORKDIR /app/frontend

# Pass build-time environment variables
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL

COPY UI/package*.json ./
RUN npm install
COPY UI/ ./
CMD ["npm", "start"]
    
# ----------------------------
# Stage 2 - Flask Backend
# ----------------------------
FROM python:3.13.3 AS backend

WORKDIR /app/backend

ENV PYTHONPATH=/app/backend
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY Backend/requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install --prefer-binary -r requirements.txt

COPY Backend/ ./

CMD ["flask", "run", "--host=0.0.0.0"]