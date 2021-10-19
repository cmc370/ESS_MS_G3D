# ESS MS-G3D
Skeleton-Based Assessment of Severe Mental Retardation. Our code is based on [MS-G3D](https://github.com/kenziyuliu/MS-G3D.git).

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

# Data Processing

We used [AlphaPose](https://github.com/WildflowerSchools/AlphaPose) to extract 2D coordinates and key point scores of 18 human joints in each frame of MR videos.

- ### Directory Structure

```c++
-------------------------------------------------joint-----------------------------------------------------
- ./data/
	- train_data_joint.npy
    - train_label.pkl
    - val_data_joint.npy
    - val_label.pkl
    - test_data_joint.npy
    - test_label.pkl
-------------------------------------------------bone-----------------------------------------------------
- ./data/
	- train_data_bone.npy
    - train_label.pkl
    - val_data_bone.npy
    - val_label.pkl
    - test_data_bone.npy
    - test_label.pkl
---------------------------------------------joint moiton-------------------------------------------------
- ./data/
	- train_data_joint_motion.npy
    - train_label.pkl
    - val_data_joint_motion.npy
    - val_label.pkl
    - test_data_joint_motion.npy
    - test_label.pkl
----------------------------------------------bone moiton-------------------------------------------------
- ./data/
	- train_data_bone_motion.npy
    - train_label.pkl
    - val_data_bone_motion.npy
    - val_label.pkl
    - test_data_bone_motion.npy
    - test_label.pkl
```

