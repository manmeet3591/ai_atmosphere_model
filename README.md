# ai_atmosphere_model

pip install -U "xarray[complete]" zarr fsspec gcsfs numcodecs dask[complete]

python get_data_surface_var.py --var top_net_thermal_radiation

1000, 950, 825, 500, 200/250

For weather regime and MJO you need above levels.

## Neural Architecture Search (NAS) Results

An autonomous Neural Architecture Search loop was run to optimize the diffusion UNet 3D model for S2S prediction. The best performing model configuration (minimizing validation loss within a 15-minute training budget) was extracted into `train_best.py`.

The optimal architecture:
- `BLOCK_OUT_CHANNELS = (128, 256, 512, 512)`
- `LAYERS_PER_BLOCK = 1`
- `ATTENTION_HEAD_DIM = 32`
- Cross-attention applied **only at the bottleneck level** (`level 3` / 8x8), leaving levels 0-2 as pure convolutional blocks for maximum computational efficiency.
