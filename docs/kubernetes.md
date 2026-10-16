# Kubernetes

MarsWorld runs as a two-stage batch pipeline backed by a shared PVC.

```text
  [generate-data Job (CPU)] --writes--> PVC (data/) --reads--> [train Job (GPU)]
                                           ^                         |
                                           +------ checkpoints/ <----+
```

## Manifests

| File | Resource |
| --- | --- |
| `k8s/pvc.yaml` | `PersistentVolumeClaim` for data + checkpoints |
| `k8s/generate-data-job.yaml` | CPU Job: synthesise trajectories |
| `k8s/train-job.yaml` | GPU Job: train PI-JEPA |
| `k8s/Dockerfile` | Training/data image (PyTorch CUDA base) |

## Build & deploy

```bash
docker build -f k8s/Dockerfile -t your-registry/marsworld:latest .
docker push your-registry/marsworld:latest

kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/generate-data-job.yaml   # optional if using --auto-generate
kubectl apply -f k8s/train-job.yaml
kubectl logs -f job/marsworld-train
```

## GPU scheduling

The training Job requests `nvidia.com/gpu: "1"` and tolerates the standard GPU
taint. Set a `nodeSelector` matching your cluster's accelerator label (an example
is commented in `k8s/train-job.yaml`).

## CPU-only clusters

Swap the Dockerfile base image for `python:3.11-slim`, install the CPU torch
wheel, and remove the `nvidia.com/gpu` requests/limits and GPU toleration from
`train-job.yaml`.

## Sweeping configs

Mount an alternate YAML via a `ConfigMap` and point `--config` at it, e.g. to run
the `data.data_fraction=0.25` data-efficiency point without rebuilding the image.
