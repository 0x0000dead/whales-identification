# Deployment guide

Three deployment profiles, ranked by effort:

1. **Local demo** — `docker compose up` (5 min, zero config).
2. **Single VPS** — systemd + nginx reverse proxy (1 hour).
3. **Production cluster** — Kubernetes with HPA, persistent model cache, Prometheus scraping (1 day).

---

## 1. Local demo (laptop / single-node Docker)

Pre-requisites:
- Docker Desktop or any Docker-compatible runtime (Lima, Colima, Podman).
- ~3 GB free disk (image + model weights download on first boot).

```bash
git clone https://github.com/0x0000dead/whales-identification
cd whales-identification
docker compose up --build
```

Open:
- **Web UI** → http://localhost:8080
- **Swagger UI** → http://localhost:8000/docs
- **Metrics** → http://localhost:8000/metrics

Stop with `docker compose down`.

### LAN access from another device

No configuration needed. With the defaults:

- `VITE_BACKEND` is empty, so the frontend resolves the backend at runtime as
  `http://<host the UI is opened from>:8000` — the same image works from
  `127.0.0.1`, a LAN IP or a hostname without rebuilds;
- the dev compose sets `ALLOWED_ORIGINS=*`, so the API accepts requests from
  any origin (the API uses no cookies, so wildcard mode is safe; narrow the
  list for production).

Just run `docker compose up --build` and open `http://<machine-IP>:8080` from
any device on the network. Set `VITE_BACKEND` only for reverse-proxy setups or
a non-standard backend port (this is a build-time variable — re-build after
changing it).

### GPU acceleration (docker compose)

Prerequisite — [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# verify the host setup:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Then start the stack with the GPU overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The backend image already contains CUDA-enabled torch wheels (cu121); once
the device is passed through, inference automatically runs on `cuda:0`.
Verify:

```bash
curl http://localhost:8000/health
# {"status": "ok", "device": "cuda:0"}
```

### Environment variables (local)

| Variable             | Default                                           | Purpose                                      |
|----------------------|---------------------------------------------------|----------------------------------------------|
| `HF_REPO`            | `0x0000dead/ecomarineai-cetacean-effb4`          | Model source for docker-entrypoint.sh        |
| `MODEL_DOWNLOAD_URL` | unset                                             | Legacy: direct URL override for efficientnet_b4_512_fold0.ckpt |
| `ALLOWED_ORIGINS`    | `*` (dev compose)                                 | CORS: `*` or comma-separated origin whitelist |
| `VITE_BACKEND`       | empty (runtime same-host fallback)                | Frontend → backend URL override (build-time) |

---

## 2. Single-VPS install (Ubuntu 22.04)

### 2.1 System packages

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    nginx certbot python3-certbot-nginx
```

### 2.2 Clone + install

```bash
cd /opt
sudo git clone https://github.com/0x0000dead/whales-identification
sudo chown -R $USER /opt/whales-identification
cd whales-identification
python3.11 -m venv .venv
source .venv/bin/activate
cd whales_be_service && pip install poetry && poetry install && cd ..
```

### 2.3 Download weights

```bash
bash scripts/download_models.sh
```

### 2.4 systemd unit

`/etc/systemd/system/ecomarine.service`:

```ini
[Unit]
Description=EcoMarineAI inference API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/whales-identification/whales_be_service
Environment="PYTHONPATH=/opt/whales-identification/whales_be_service/src"
Environment="ALLOWED_ORIGINS=https://ecomarine.example.com"
ExecStart=/opt/whales-identification/.venv/bin/python -m uvicorn whales_be_service.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ecomarine
sudo systemctl status ecomarine
```

### 2.5 Nginx reverse proxy

`/etc/nginx/sites-available/ecomarine`:

```nginx
server {
    listen 80;
    server_name ecomarine.example.com;

    client_max_body_size 64m;  # batch ZIPs can be large

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/ecomarine /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d ecomarine.example.com
```

### 2.6 Rotate logs

```bash
sudo journalctl -u ecomarine --vacuum-time=14d
```

---

## 3. Kubernetes

Minimum components:

- **Deployment** — ≥ 2 replicas of the backend container.
- **PVC** — persistent volume that caches the HF model weights so restarting a pod doesn't re-download 400 MB.
- **Service** → **Ingress** → ClusterIP over HTTPS.
- **HorizontalPodAutoscaler** — scale on CPU or on `predictions_total` increase rate.
- **ServiceMonitor** — Prometheus scrapes `/metrics` every 15 s.

### Example manifests

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecomarine-backend
spec:
  replicas: 2
  selector: {matchLabels: {app: ecomarine}}
  template:
    metadata: {labels: {app: ecomarine}}
    spec:
      containers:
        - name: backend
          image: ghcr.io/0x0000dead/ecomarine-backend:latest
          ports: [{containerPort: 8000}]
          env:
            - {name: HF_REPO,     value: "0x0000dead/ecomarineai-cetacean-effb4"}
            - {name: ALLOWED_ORIGINS, value: "https://ecomarine.example.com"}
          resources:
            requests: {cpu: "1000m", memory: "2Gi"}
            limits:   {cpu: "4000m", memory: "4Gi"}
          readinessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 60
          livenessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 300
          volumeMounts:
            - {name: hf-cache, mountPath: /home/appuser/.cache/huggingface}
      volumes:
        - name: hf-cache
          persistentVolumeClaim: {claimName: ecomarine-hf-cache}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: ecomarine-hf-cache}
spec:
  accessModes: [ReadWriteMany]
  resources: {requests: {storage: 2Gi}}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: ecomarine-backend}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ecomarine-backend
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}
```

### Startup time tuning

Lazy loading means the first request after a cold boot takes ~5 s. Configure a `preStop` hook + 60 s grace period so the HPA doesn't kill pods that are still warming up:

```yaml
lifecycle:
  preStop:
    exec: {command: ["/bin/sh", "-c", "sleep 10"]}
```

---

## Monitoring stack (Prometheus + Grafana)

The service already exposes Prometheus metrics. Minimal `prometheus.yml` scrape config:

```yaml
scrape_configs:
  - job_name: ecomarine
    static_configs: [{targets: ["ecomarine-backend:8000"]}]
    metrics_path: /metrics
```

Recommended alerts:

- `availability_percent < 95` for 10 min → page on-call.
- `rejections_total` rate > 50% of `requests_total` for 15 min → model drift.
- `cetacean_score_avg` drops > 0.1 from baseline → drift.
- `latency_avg_ms > 2000` for 5 min → capacity.

## Backups

Nothing in the pod is stateful, but `models/registry.json` and `reports/metrics_baseline.json` should be committed to git after each retraining. The weights themselves live on HuggingFace + Kaggle mirrors; no VPS backup needed.
