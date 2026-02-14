# AutoDL A800 Full Workflow

This guide is for running this project on AutoDL with `A800-80GB` (Linux).

## 1. Rent machine on AutoDL

1. Open AutoDL marketplace.
2. Choose zone and `A800-80GB`.
3. Recommended:
   - GPU count: `1`
   - Disk: `>= 100GB` (model + outputs need space)
   - Public network: enabled (for git/huggingface download)
4. Create instance and open terminal/Jupyter.

## 2. Upload project to instance

Choose one of the following:

1. `git clone`:
   ```bash
   cd /root/autodl-tmp
   git clone <your_repo_url> mamba
   cd mamba
   ```
2. Or upload ZIP from local and unzip to:
   - `/root/autodl-tmp/mamba`

## 3. Prepare environment

In repo root:

```bash
cd /root/autodl-tmp/mamba
bash setup_autodl_a800.sh
conda activate mamba_a800
```

## 4. Prepare data and model paths

### 4.1 Required files

1. Training CSV:
   - `outputs/excel_caption/caption_train_template_shortq.csv`
2. Validation CSV:
   - `outputs/excel_caption/caption_val_template_shortq.csv`
3. Mamba model directory:
   - `models/mamba-2.8b-hf` (contains `config.json`, `model.safetensors`, tokenizer files)
4. nnU-Net encoder checkpoint and NIfTI data (as required by your CSV/image paths).

### 4.2 Path strategy

You have two options:

1. Keep CSV paths as Windows `d:/...` and place data under mounted `/mnt/d/...` (WSL-style).
2. Use Linux paths (recommended on AutoDL), then edit CSV and `config/paths.yaml` accordingly.

Minimum keys in `config/paths.yaml` to check:

1. `caption_csv_train`
2. `caption_csv_val`
3. `caption_prompt_json`
4. `nnunet_encoder_checkpoint`

## 5. Run auto curriculum training

In repo root:

```bash
cd /root/autodl-tmp/mamba
conda activate mamba_a800
bash run_auto_curriculum_a800.sh
```

Default pipeline:

1. Stage A: `max_visual_tokens=36`, `max_text_len=256`
2. Stage B: `max_visual_tokens=64`, `max_text_len=512`
3. Strict validation + evaluation

Output:

1. Stage A checkpoint:
   - `outputs/auto_curriculum_a800/stageA_36_256/vision_bridge_vlm_final.pt`
2. Stage B checkpoint:
   - `outputs/auto_curriculum_a800/stageB_64_512/vision_bridge_vlm_final.pt`
3. Validation runs:
   - `mamba-res/run_strict_*/`

## 6. Common runtime variants

1. Faster training (A800 usually can handle larger batch):
   ```bash
   BATCH_SIZE=2 GRAD_ACC=2 bash run_auto_curriculum_a800.sh
   ```
2. Skip validation:
   ```bash
   SKIP_VALIDATE=1 bash run_auto_curriculum_a800.sh
   ```
3. Custom model or CSV path:
   ```bash
   TRAIN_CSV=/path/to/train.csv MAMBA_MODEL=/path/to/mamba-2.8b-hf bash run_auto_curriculum_a800.sh
   ```

## 7. Resume behavior

The script auto-resumes when it finds latest step checkpoint:

1. `vision_bridge_vlm_step*.pt` in stage output dir
2. Or explicit stage handoff from Stage A final checkpoint into Stage B

If final checkpoint already exists, stage is skipped.

## 8. Download results back to local

Download these folders:

1. `outputs/auto_curriculum_a800/`
2. `mamba-res/`

Use AutoDL file manager, `scp`, or sync tools.

## 9. Troubleshooting

1. CUDA unavailable:
   - Check instance really has GPU attached.
   - Run:
     ```bash
     nvidia-smi
     python -c "import torch; print(torch.cuda.is_available())"
     ```
2. `mamba-ssm` install failed:
   - This is optional in setup script.
   - Training still runs with fallback path (slower).
3. OOM:
   - Reduce `BATCH_SIZE` to `1`.
   - Increase `GRAD_ACC` to keep effective batch size.
4. Model loading errors:
   - Ensure `models/mamba-2.8b-hf/config.json` exists.
5. Missing CSV/image:
   - Verify CSV path exists and image paths are reachable on Linux.
