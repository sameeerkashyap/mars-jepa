# Kubernetes deployment

Two-stage pipeline that runs on any cluster with a GPU node pool.

## 1. Build & push the image

```bash
docker build -f k8s/Dockerfile -t your-registry/marsworld:latest .
docker push your-registry/marsworld:latest
```

Replace `your-registry/marsworld:latest` in the Job manifests with the pushed tag.

## 2. Provision shared storage

```bash
kubectl apply -f k8s/pvc.yaml
```

## 3. Generate data (CPU Job)

```bash
kubectl apply -f k8s/generate-data-job.yaml
kubectl logs -f job/marsworld-generate-data
```

## 4. Train (GPU Job)

```bash
kubectl apply -f k8s/train-job.yaml
kubectl logs -f job/marsworld-train
```

Checkpoints land on the `marsworld-data` PVC under `/app/checkpoints`.

## Notes
- The training Job uses `--auto-generate`, so step 3 is optional for a quick run.
- For CPU-only clusters, change the base image in `k8s/Dockerfile` and drop the
  `nvidia.com/gpu` resource requests/tolerations in `train-job.yaml`.
- To run data-efficiency sweeps, mount a ConfigMap with an alternate YAML (e.g.
  `data.data_fraction: 0.25`) and point `--config` at it.
