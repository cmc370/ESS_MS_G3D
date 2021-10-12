# ESS_MS_G3D
Skeleton-Based Assessment of Severe Mental Retardation. Our code is based on 

[MS-G3D]: https://github.com/kenziyuliu/MS-G3D.git

# Dependencies
* Python == 3.6
* Pytorch == 1.7.0
* Matplotlib == 3.3.4
* Sklearn 
* PyYAML, Tqdm, Numpy
# Quick Start
- ### Training

```python
cd ./
python main.py 
--config ./kinetics-skeleton/train_joint.yaml 
--work-dir ./work_dir/ 
--num-worker 4
```

- ### Testing

```python
cd ./
python main.py 
--config ./kinetics-skeleton/test_joint.yaml 
--work-dir ./work_dir/ 
--num-worker 4
```



