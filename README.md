# AtomicGS
 
```shell
conda env create --file environment.yml
```

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

```shell
python export.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to GS model> -i <load iteration>
```