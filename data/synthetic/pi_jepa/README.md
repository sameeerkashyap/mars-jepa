# Spatial PI-JEPA data

Generate `temperature.npz` here with:

```bash
python -m scripts.generate_data --config configs/pi_jepa/default.yaml
```

These two-dimensional diffusion fields train the spatial prototype. The heating
pilot uses a separate dataset in `data/synthetic/pilot/`.
