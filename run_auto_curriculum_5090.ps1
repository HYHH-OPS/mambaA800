param(
  [ValidateSet("win5090", "autodl_a800")]
  [string]$Profile = "win5090",
  [int]$EpochA = 2,
  [int]$EpochB = 20,
  [string]$PythonExe = "",
  [string]$OutRoot = "",
  [string]$TrainCsv = "",
  [string]$MambaModel = "",
  [string]$ResultDir = "",
  [int]$BatchSize = 1,
  [int]$GradAccumulationSteps = 4,
  [switch]$SkipValidate
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }
Set-Location $RepoRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:MAMBA_FORCE_CUDA) { $env:MAMBA_FORCE_CUDA = "0" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }

if ($Profile -eq "autodl_a800") {
  if (-not $PythonExe) { $PythonExe = "python" }
  if (-not $OutRoot) { $OutRoot = Join-Path $RepoRoot "outputs/auto_curriculum_a800" }
  if (-not $TrainCsv) { $TrainCsv = Join-Path $RepoRoot "outputs/excel_caption/caption_train_template_shortq.csv" }
  if (-not $MambaModel) { $MambaModel = Join-Path $RepoRoot "models/mamba-2.8b-hf" }
  if (-not $ResultDir) { $ResultDir = Join-Path $RepoRoot "mamba-res" }
} else {
  if (-not $PythonExe) { $PythonExe = "D:/anaconda/envs/mamba5090/python.exe" }
  if (-not $OutRoot) { $OutRoot = Join-Path $RepoRoot "outputs/auto_curriculum" }
  if (-not $TrainCsv) { $TrainCsv = Join-Path $RepoRoot "outputs/excel_caption/caption_train_template_shortq.csv" }
  if (-not $MambaModel) { $MambaModel = Join-Path $RepoRoot "models/mamba-2.8b-hf" }
  if (-not $ResultDir) { $ResultDir = Join-Path $RepoRoot "mamba-res" }
}

$StageAOut = Join-Path $OutRoot "stageA_36_256"
$StageBOut = Join-Path $OutRoot "stageB_64_512"

New-Item -ItemType Directory -Force -Path $StageAOut | Out-Null
New-Item -ItemType Directory -Force -Path $StageBOut | Out-Null
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

Write-Host "[profile] $Profile"
Write-Host "[repo] $RepoRoot"
Write-Host "[python] $PythonExe"
Write-Host "[train_csv] $TrainCsv"
Write-Host "[mamba_model] $MambaModel"
Write-Host "[out_root] $OutRoot"
Write-Host "[result_dir] $ResultDir"

function Invoke-TrainStage {
  param(
    [string]$Name,
    [string]$OutputDir,
    [int]$Epochs,
    [int]$MaxVisualTokens,
    [int]$MaxTextLen,
    [string]$VisionCheckpoint = ""
  )
  $finalCkpt = "$OutputDir/vision_bridge_vlm_final.pt"
  if (Test-Path $finalCkpt) {
    Write-Host "[skip] $Name already complete: $finalCkpt"
    return
  }

  $resumeCkpt = ""
  if ($VisionCheckpoint -and (Test-Path $VisionCheckpoint)) {
    $resumeCkpt = $VisionCheckpoint
  } else {
    $latestStep = Get-ChildItem $OutputDir -File -Filter "vision_bridge_vlm_step*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestStep) {
      $resumeCkpt = $latestStep.FullName
      Write-Host "[resume] $Name from latest step checkpoint: $resumeCkpt"
    }
  }

  $args = @(
    "train_vlm.py",
    "--csv", $TrainCsv,
    "--mamba_model", $MambaModel,
    "--epochs", "$Epochs",
    "--batch_size", "$BatchSize",
    "--lr", "1e-5",
    "--max_visual_tokens", "$MaxVisualTokens",
    "--max_text_len", "$MaxTextLen",
    "--gradient_accumulation_steps", "$GradAccumulationSteps",
    "--gradient_checkpointing",
    "--align_vocab",
    "--num_workers", "0",
    "--log_every_steps", "1",
    "--save_every_steps", "100",
    "--output_dir", $OutputDir
  )
  if ($resumeCkpt) {
    $args += @("--vision_checkpoint", $resumeCkpt)
  }

  Write-Host "[run] $Name => max_visual_tokens=$MaxVisualTokens max_text_len=$MaxTextLen epochs=$Epochs"
  & $PythonExe @args
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
  if (-not (Test-Path $finalCkpt)) {
    throw "$Name finished but checkpoint not found: $finalCkpt"
  }
}

Invoke-TrainStage -Name "Stage A (survival)" -OutputDir $StageAOut -Epochs $EpochA -MaxVisualTokens 36 -MaxTextLen 256
$StageACkpt = "$StageAOut/vision_bridge_vlm_final.pt"
Invoke-TrainStage -Name "Stage B (quality)" -OutputDir $StageBOut -Epochs $EpochB -MaxVisualTokens 64 -MaxTextLen 512 -VisionCheckpoint $StageACkpt

if (-not $SkipValidate) {
  $FinalCkpt = "$StageBOut/vision_bridge_vlm_final.pt"
  Write-Host "[run] strict validation => min_chars=180 max_new_tokens=640 constrained_decode"
  & $PythonExe "scripts/infer_template_strict.py" `
    --checkpoint $FinalCkpt `
    --mamba_model $MambaModel `
    --val_sample `
    --num_val 20 `
    --max_new_tokens 640 `
    --max_visual_tokens 64 `
    --min_chars 180 `
    --max_retries 2 `
    --num_beams 4 `
    --constrained_decode `
    --length_penalty 1.2 `
    --repetition_penalty 1.15 `
    --out_dir $ResultDir
  if ($LASTEXITCODE -ne 0) {
    throw "strict validation failed with exit code $LASTEXITCODE"
  }

  $latestRun = Get-ChildItem $ResultDir -Directory -Filter "run_strict_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latestRun) {
    & $PythonExe "scripts/evaluate_generation_run.py" --run_dir $latestRun.FullName
    if ($LASTEXITCODE -ne 0) {
      throw "evaluation failed with exit code $LASTEXITCODE"
    }
  }
}

Write-Host "[done] Auto curriculum pipeline completed."
