# EcoMarineAI — Kubernetes deployment

Production deployment manifests for the EcoMarineAI inference backend. These
manifests target any vanilla Kubernetes 1.27+ cluster with the
`ingress-nginx` controller and the `metrics-server` installed.

## Files

| File             | Purpose                                                              |
|------------------|----------------------------------------------------------------------|
| `configmap.yaml` | Non-secret runtime configuration (CORS origins, HF repo, log level). |
| `deployment.yaml`| Backend Deployment: 3 replicas, resource limits, readiness/liveness. |
| `service.yaml`   | ClusterIP Service exposing port 80 → pod :8000.                      |
| `hpa.yaml`       | HorizontalPodAutoscaler, min 3 / max 10, target CPU 70 %.            |
| `ingress.yaml`   | nginx Ingress with rate limit 60 req/min per IP + TLS hint.          |

## Prerequisites

- Kubernetes 1.27+ (older clusters miss the `autoscaling/v2` HPA schema).
- `metrics-server` installed (HPA requires pod CPU metrics).
- `ingress-nginx` controller installed.
- (Optional) `cert-manager` + a `ClusterIssuer` for TLS certificates.
- A container registry that serves the `ecomarine-backend:latest` image.
  The Dockerfile lives in `whales_be_service/Dockerfile`.

## One-time setup

```bash
# Create the namespace — manifests assume 'ecomarineai' exists.
kubectl create namespace ecomarineai

# (Optional) secret for Hugging Face Hub authentication if you use a
# private model repo:
kubectl -n ecomarineai create secret generic hf-credentials \
    --from-literal=HF_TOKEN=hf_xxx
```

## Apply the stack

```bash
kubectl apply -f k8s/
```

Verify the rollout:

```bash
kubectl -n ecomarineai rollout status deploy/ecomarine-backend
kubectl -n ecomarineai get pods,svc,hpa,ingress
```

## Smoke test

```bash
# Port-forward the ClusterIP Service for a quick check:
kubectl -n ecomarineai port-forward svc/ecomarine-backend 8000:80
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/metrics | head
```

For a full end-to-end check run `scripts/smoke_test.sh` against the
forwarded port.

## Observability

- `/metrics` exposes Prometheus-compatible counters (requests, latency,
  availability). The Deployment already carries `prometheus.io/scrape`
  annotations so kube-prometheus picks the pods up automatically.
- Log shipping: mount a Fluent Bit DaemonSet or any log-forwarding agent
  of your choice — the backend writes structured logs to stdout.

## Scaling behaviour

The HPA is configured for:

- min 3 replicas (rolling upgrade headroom)
- max 10 replicas
- target 70 % CPU utilisation
- 30 s scale-up stabilisation, 5 min scale-down stabilisation

These thresholds were validated against the load test in
`reports/LOAD_TEST.md` to keep the system within the ТЗ Параметр 3
latency budget (p95 < 8 s) at the 50 RPS throughput goal.

## Rolling upgrades

```bash
kubectl -n ecomarineai set image deploy/ecomarine-backend \
    backend=ghcr.io/your-org/ecomarine-backend:<new-tag>
kubectl -n ecomarineai rollout status deploy/ecomarine-backend
```

Roll back:

```bash
kubectl -n ecomarineai rollout undo deploy/ecomarine-backend
```

## Tear down

```bash
kubectl delete -f k8s/
kubectl delete namespace ecomarineai   # if you used the dedicated namespace
```
