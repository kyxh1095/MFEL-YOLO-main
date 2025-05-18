### 1. install

Pip install the ultralytics package including all requirements.txt in a 3.10>=Python>=3.7 environment, including PyTorch>=1.7.

```
pip install ultralytics
```

### 2.Usage

#### 2.1 train

```
python train.py 
```

#### 2.2val

```
python val.py 
```

### 3.Key Directory Structure  

#### 3.1 Configuration Files (ultralytics/cfg)

Model Architecture: Located in the MFEL-YOLO subdirectory, containing detailed configuration files for network structures.

Dataset Settings: Stored in the datasets subdirectory, defining paths, classes, and related parameters.

Training/Testing Setup: Managed via default.yaml, including hyperparameters, optimizer settings, and other training configurations.

#### 3.2 Module Implementations (ultralytics/nn/modules)
