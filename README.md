# IMBD 2020

## Environment Setup

### Step 1: Build virtual environment

```bash
python3 -m venv ./venv
```

### Step 2: Activate venv

```bash
source ./venv/bin/activate
```

### Step 3: Install requirements

```bash
pip install -r requirements.txt
```

### Step 3.5: Train model (optional)

```bash
python train.py --file_path=train/data/path
```

### Step 4: Test model

```bash
python test.py --file_path=test/data/path --model_path=model/path
```
