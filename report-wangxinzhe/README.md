# Reproducing the Submitted MP1 Test BPB Without Training

**Author:** WangXinzhe  
**Student ID:** u3036801008  
**Course:** DASE7506 MP1 Small Language Model Challenge  
**Protocol:** `7506-mp1-wt2-v2`

This document reproduces the submitted full-test result directly from the frozen checkpoint. Run all commands from the extracted package's `code` directory. No training, teacher checkpoint, CUDA device, or call to `train.py` is required.

## Submitted Result

The course website submission value is the `bpb` field produced on the complete `test` split:

| Field | Recorded value |
|---|---:|
| **Test BPB to submit** | **`1.4615442689258094`** |
| Token perplexity | `21.226944720949717` |
| Scored targets | `428405` |
| UTF-8 bytes | `1292013` |
| CPU FP32 scoring time on the reproduction machine | `43.158605300001 s` |
| CPU threads | `4` |

BPB is the ranking metric and lower is better. Do not submit token perplexity or the validation BPB. Runtime varies by machine, but the BPB and target count should reproduce up to small floating-point differences.

The validation result used during development and ablation was `1.4440572213535412` BPB. It is not the score submitted to the course website.

## Required Files

Keep the supplied relative layout:

```text
code/
  common.py
  evaluate.py
  student.py
  requirements.txt
  data/
  tests/
  runs/
    student-cache-distill-s17/
      checkpoint.pt
```

Only `runs/student-cache-distill-s17/checkpoint.pt` is needed from the trained artifacts. The training-only teacher is not loaded by the evaluator and is not needed for reproduction.

Frozen artifact identity:

| Artifact | SHA256 | Size |
|---|---|---:|
| Final checkpoint | `9e7ae095e6a3da3589cf3e6857b058e08d7bac00a7022d9e82d50bb00e38a44c` | `26281973` bytes |
| `student.py` | `68a9fdb1c678ec5ad0306b02115b8accb15fd0577f4dad9ad527d65cf77acec1` | `11076` bytes |
| `evaluate.py` | `128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d` | `3437` bytes |
| `data/tokenizer.json` | `020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e` | `121943` bytes |

The checkpoint records `implementation=student`, seed `17`, selected step `19200`, and `157286400` direct student training targets.

## Windows CMD Reproduction

### 1. Enter the extracted code directory

```cmd
cd /d C:\path\to\the\extracted-package\code
```

Replace the example path with the actual extraction location.

### 2. Create the Python environment

Use Python 3.12:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The recorded software versions are:

```text
Python 3.12
torch 2.7.1
numpy 2.5.3
tokenizers 0.21.4
```

After installation, evaluation is offline and needs neither a GPU nor an external dataset.

### 3. Verify the frozen artifacts

```cmd
certutil -hashfile runs\student-cache-distill-s17\checkpoint.pt SHA256
certutil -hashfile student.py SHA256
certutil -hashfile evaluate.py SHA256
certutil -hashfile data\tokenizer.json SHA256
```

Compare all four outputs with the table above. A hash mismatch means the result no longer identifies the submitted predictor. The data loader also checks the supplied tokenizer and dataset against `data\manifest.json` during evaluation.

### 4. Run the course contract tests

```cmd
python -m unittest discover -s tests -v
```

All tests must pass. They cover causality, normalized probabilities, independence between windows and examples, finite gradients, and complete target counting.

### 5. Reproduce the full-test BPB

```cmd
python evaluate.py --checkpoint runs\student-cache-distill-s17\checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced_test_cpu_fp32.json
```

The important fields in `reproduced_test_cpu_fp32.json` should be:

```json
{
  "bpb": 1.4615442689258094,
  "token_ppl": 21.226944720949717,
  "targets": 428405,
  "utf8_bytes": 1292013,
  "protocol": "7506-mp1-wt2-v2",
  "split": "test",
  "precision": "fp32",
  "checkpoint_sha256": "9e7ae095e6a3da3589cf3e6857b058e08d7bac00a7022d9e82d50bb00e38a44c"
}
```

The evaluator also creates `reproduced_test_cpu_fp32.window-nll.npy`. The number entered on the course website is exactly the JSON `bpb` value, `1.4615442689258094`. It is not `token_ppl` and it is not the validation BPB.

## Optional Validation Diagnostic

Validation is useful for checking consistency with the development record, but it is not the submitted test score:

```cmd
python evaluate.py --checkpoint runs\student-cache-distill-s17\checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced_validation_cpu_fp32.json
```

Expected diagnostic values:

```json
{
  "bpb": 1.4440572213535412,
  "targets": 376599,
  "utf8_bytes": 1148007,
  "protocol": "7506-mp1-wt2-v2",
  "split": "validation",
  "precision": "fp32"
}
```

## Evaluation Resource Checks

### CPU time

The course timing ratio must compare the baseline and final predictor on the same machine, software stack, thread count, precision, and split. The recorded same-split validation measurements are:

```text
Supplied baseline validation:  8.6860655 s
Final predictor validation:   38.9519618 s
Normalized validation ratio:   4.4844195x
Course limit:                  5x
```

The separately recorded full-test time is `43.158605300001 s`. It is not divided by the validation baseline time because the test split contains a different number of targets.

### Peak CPU RAM on Windows

This optional CMD command evaluates the full test split while sampling the evaluator process's working set and saves the peak to a text file:

```cmd
powershell -NoProfile -Command "$p=Start-Process python -ArgumentList 'evaluate.py --checkpoint runs\student-cache-distill-s17\checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output ram_test_cpu_fp32.json' -NoNewWindow -PassThru; $peak=0; while(-not $p.HasExited){$p.Refresh(); if($p.WorkingSet64 -gt $peak){$peak=$p.WorkingSet64}; Start-Sleep -Milliseconds 50}; ('peak_ram_gib={0:F3}' -f ($peak/1GB)) | Set-Content ram_test_cpu_fp32.txt; Get-Content ram_test_cpu_fp32.txt"
```

The recorded validation peak was `1.790 GiB`; the course limit is `4 GiB`. Windows working-set sampling can vary slightly between runs.

### Inference asset size

```text
checkpoint.pt  26281973 bytes
student.py        11076 bytes
combined       26293049 bytes = 25.075 MiB
course limit   64 MiB
```

The teacher checkpoint is a training-only ancestor, so it is not an inference asset.

## Expected Evaluation Behavior

- Evaluation runs on CPU in FP32 and is reproducible without CUDA.
- The model receives only input prefixes and returns normalized natural-log probabilities.
- The neural cache is strictly causal and local to the current 256-token window.
- Cache state is recreated on every call and never crosses windows or examples.
- The model does not access the network during evaluation.
- The supplied data, tokenizer, context length, evaluator, and scoring rule are unchanged.

## Troubleshooting

**`CUDA is unavailable`**  
Use the command exactly as shown with `--device cpu`. CUDA is not needed.

**`ModuleNotFoundError: tokenizers`**  
Activate `.venv` and run `python -m pip install -r requirements.txt`.

**Checkpoint hash mismatch**  
Use the frozen `checkpoint.pt` whose SHA256 is listed above. Do not substitute an intermediate checkpoint or the teacher checkpoint.

**Different CPU time but the same BPB**  
CPU time depends on the processor, operating system, thread scheduling, and PyTorch build. For compliance, time the final model and baseline under identical conditions. Use BPB, target count, split, and hashes as the primary correctness checks.

## Method Summary

The submitted predictor is a 6-layer, width-288 causal Transformer with six attention heads, RoPE, RMSNorm, SwiGLU, depth-scaled residual initialization, training-time residual dropout, and a learned causal within-window neural cache. It has `6565826` parameters. Training used equal weighting between hard-label cross-entropy and KL distillation from a self-trained no-cache teacher. The teacher contributes no weights or computation during evaluation.

For the full method, controlled ablations, training costs, and critical analysis, see `report-wangxinzhe\MP1_Report_WangXinzhe_u3036801008.pdf`.
