# ai_atmosphere_model

pip install -U "xarray[complete]" zarr fsspec gcsfs numcodecs dask[complete]

python get_data_surface_var.py --var top_net_thermal_radiation

1000, 950, 825, 500, 200/250

For weather regime and MJO you need above levels.

apptainer exec --nv --env PYTHONNOUSERSITE=1 --bind /media/airlab/ROCSTOR:/media/airlab/ROCSTOR /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif python3 /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/train.py --godas-dir /media/airlab/ROCSTOR/earthmind_s2s/godas_pentad --static-dir /media/airlab/ROCSTOR/earthmind_s2s/static_fields --hf-repo sluitel/ai-atmosphere-s2s --start-date 2018-01-06 --end-date 2022-01-01 --epochs-per-day 10 2>&1 | tee train.log

apptainer exec --nv --env PYTHONNOUSERSITE=1 --bind /media/airlab/ROCSTOR:/media/airlab/ROCSTOR /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif python3 /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/train_multi_days.py --godas-dir /media/airlab/ROCSTOR/earthmind_s2s/godas_pentad --static-dir /media/airlab/ROCSTOR/earthmind_s2s/static_fields --hf-repo sluitel/ai-atmosphere-s2s --batch-days 48 --epochs-per-batch 10 --start-date 2018-01-06 --end-date 2022-01-01  2>&1 | tee train.lo


CUDA_VISIBLE_DEVICES=1 apptainer exec --nv --env PYTHONNOUSERSITE=1 --bind /media/airlab/ROCSTOR:/media/airlab/ROCSTOR /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/ai_atmosphere.sif python3 /media/airlab/ROCSTOR/earthmind_s2s/ai_atmosphere_model/train_multi_days.py --godas-dir /media/airlab/ROCSTOR/earthmind_s2s/godas_pentad --static-dir /media/airlab/ROCSTOR/earthmind_s2s/static_fields --hf-repo dsocairlab/Earthmind-S2S --batch-days 8 --epochs-per-batch 8 --start-date 2018-01-06 --end-date 2022-01-01 2>&1 | tee train.log
