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
python test.py --file_path=test/data/path --model_path=model/path --preprocessor_path=preprocessor/path
```

## Another Running Method

### Step 1: Build docker image

```bash
docker build -t imbd2020 \
             --no-cache \
             --build-arg UID=$(id -u) \
             --build-arg GID=$(id -g) \
             --build-arg USER=$(whoami) \
             --build-arg GROUP=$(whoami) \
             --no-cache \
             .
```

### Step 2: Run training process with docker container

```bash
docker run -v $(pwd)/data:/home/$(whoami)/imbd2020/data \
           -v $(pwd)/models:/home/$(whoami)/imbd2020/models \
           -v $(pwd)/results:/home/$(whoami)/imbd2020/results \
           imbd2020 \
           train.py
```

### Step 3: Run testing process with docker container

```bash
docker run -v $(pwd)/data:/home/$(whoami)/imbd2020/data \
           -v $(pwd)/models:/home/$(whoami)/imbd2020/models \
           -v $(pwd)/results:/home/$(whoami)/imbd2020/results \
           imbd2020 \
           test.py
```
