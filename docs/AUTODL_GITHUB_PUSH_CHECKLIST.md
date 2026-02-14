# GitHub -> AutoDL Push Checklist

## Should upload to GitHub

1. Source code folders:
   - `bridge/`
   - `config/`
   - `data/`
   - `llm/`
   - `model/`
   - `scripts/`
   - `vision/`
2. Entry scripts:
   - `train.py`
   - `train_vlm.py`
   - `inference.py`
   - `setup_autodl_a800.sh`
   - `run_auto_curriculum_a800.sh`
3. Docs + requirements:
   - `README.md`
   - `requirements.txt`
   - `docs/`

## Do NOT upload to GitHub

1. Private/large datasets:
   - NIfTI data
   - nnU-Net raw/preprocessed/results
2. Model weights/checkpoints:
   - `models/`
   - `outputs/`
   - `*.pt`, `*.pth`, `*.safetensors`
3. Local env/cache:
   - `.venv*`, `__pycache__/`

## Recommended push commands (run locally)

```bash
git init
git add .
git commit -m "feat: autodl a800 runnable baseline"
git branch -M main
git remote add origin https://github.com/HYHH-OPS/mambaA800.git
git push -u origin main
```

If remote already exists:

```bash
git remote set-url origin https://github.com/HYHH-OPS/mambaA800.git
git add .
git commit -m "chore: sync full code for autodl"
git push
```

## AutoDL quick run

```bash
cd /root/autodl-tmp
git clone https://github.com/HYHH-OPS/mambaA800.git mambaA800
cd mambaA800
bash scripts/autodl_preflight.sh
bash setup_autodl_a800.sh
conda activate mamba_a800
bash run_auto_curriculum_a800.sh
```
