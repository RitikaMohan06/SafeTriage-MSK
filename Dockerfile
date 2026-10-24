FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy pandas pillow matplotlib fastapi "uvicorn[standard]" python-multipart
COPY src ./src
COPY app ./app
COPY frontend ./frontend
COPY tools ./tools
COPY outputs ./outputs
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
