# Reproducing the MP1 Result Without Training

**Author:** WangXinzhe  
**Student ID:** u3036801008  
**Course:** DASE7506 MP1 Small Language Model Challenge  
**Protocol:** `7506-mp1-wt2-v2`

This document reproduces the frozen model directly from its checkpoint. Training, the teacher checkpoint, and the training script are not required for evaluation. All commands must be run from the repository's `code` directory.

## Recorded Result

The frozen checkpoint currently has the following independently recorded CPU FP32 validation result:

| Field | Value |
|---|---:|
| Validation BPB | `1.4440572213535412` |
| Token perplexity | `21.141375598872724` |
| Scored targets | `376599` |
| UTF-8 bytes | `1148007` |
| CPU scoring time on the recorded machine | `38.9519618 s` |
| CPU threads | `4` |
| Peak CPU evaluation RAM | `1.790 GiB` |

BPB is the ranking metric and lower is better. Runtime varies by machine; the probability score should reproduce up to small floating-point differences.

No full-test result JSON is present in the current workspace. The test command is provided below so the frozen predictor can be evaluated once without retraining. Do not use the resulting test score to change the method or select another checkpoint.

## Required Files

The code package must contain these files at their original relative paths:

```text
code/
  common.py
  evaluate.py
  student.py
  requirements.txt
  data/
  runs/
    student-cache-distill-s17/
      checkpoint.pt
```

Only `runs/student-cache-distill-s17/checkpoint.pt` is required from the trained artifacts. The teacher checkpoint recorded in its ancestry is not loaded during evaluation.

If the checkpoint is distributed separately from the immutable code repository, download the matching checkpoint bundle from the link supplied with the coursework submission and place `checkpoint.pt` at the relative path shown above before running any command.

Frozen artifact identity:

| Artifact | SHA256 | Size |
|---|---|---:|
| Final checkpoint | `9e7ae095e6a3da3589cf3e6857b058e08d7bac00a7022d9e82d50bb00e38a44c` | `26281973` bytes |
| `student.py` | `68a9fdb1c678ec5ad0306b02115b8accb15fd0577f4dad9ad527d65cf77acec1` | `11076` bytes |
| `evaluate.py` | `128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d` | `3437` bytes |
| `data/tokenizer.json` | `020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e` | `121943` bytes |

The checkpoint stores `implementation=student`, seed `17`, selected step `19200`, and `157286400` direct student training targets.

## Windows CMD Setup

Open CMD and enter the code directory:

```cmd
cd /d C:\path\to\the\extracted-package\code
```

Replace the example path with the actual location of the extracted package.

Create and activate a Python 3.12 virtual environment:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

Install the CPU build of PyTorch and the remaining fixed dependencies:

```cmd
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The required versions are:

```text
Python 3.12
torch 2.7.1
numpy 2.5.3
tokenizers 0.21.4
```

After installation, evaluation works offline and does not require a GPU, an API key, or any external dataset.

On Linux, the equivalent environment commands are:

```bash
cd /path/to/the/extracted-package/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Verify the Frozen Files

Run these commands in Windows CMD:

```cmd
certutil -hashfile runs\student-cache-distill-s17\checkpoint.pt SHA256
certutil -hashfile student.py SHA256
certutil -hashfile evaluate.py SHA256
certutil -hashfile data\tokenizer.json SHA256
```

Compare the hashes with the table above. Stop if the checkpoint or implementation hash differs, because the result would no longer refer to the submitted predictor.

The data loader also verifies the supplied tokenizer and dataset files against `data/manifest.json` before scoring.

## Run the Course Contract Tests

```cmd
python -m unittest discover -s tests -v
```

All tests must pass. They check causality, normalized probabilities, independence between examples and windows, finite gradients, and complete target counting.

## Reproduce the Validation Result

```cmd
python evaluate.py --checkpoint runs\student-cache-distill-s17\checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output runs\student-cache-distill-s17\reproduced_validation_cpu_fp32.json
```

The important output fields should be:

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

The evaluator also creates `reproduced_validation_cpu_fp32.window-nll.npy`. The score must come from the complete JSON result rather than an intermediate training log or token perplexity.

## Reproduce the Full Test Score

After confirming the checkpoint hash and freezing the method, run:

```cmd
python evaluate.py --checkpoint runs\student-cache-distill-s17\checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs\student-cache-distill-s17\reproduced_test_cpu_fp32.json
```

Report the `bpb` field from `reproduced_test_cpu_fp32.json`. The evaluator scores every test target once, including the final short window. Do not report validation BPB as the test result.

## Check the Evaluation Resource Limits

### CPU scoring time

The recorded four-thread validation measurements are:

```text
Supplied baseline: 8.6860655 s
Frozen predictor:  38.9519618 s
Normalized ratio:  4.4844195x
Course limit:      5x
```

For a valid timing comparison, evaluate the supplied baseline and final checkpoint on the same machine, split, thread count, PyTorch version, and FP32 precision. Timing from different machines is not comparable.

### Peak CPU RAM on Windows

The following CMD command evaluates validation while sampling the evaluator process's working set and records the peak in a text file:

```cmd
powershell -NoProfile -Command "$p=Start-Process python -ArgumentList 'evaluate.py --checkpoint runs\student-cache-distill-s17\checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output runs\student-cache-distill-s17\ram_reproduction.json' -NoNewWindow -PassThru; $peak=0; while(-not $p.HasExited){$p.Refresh(); if($p.WorkingSet64 -gt $peak){$peak=$p.WorkingSet64}; Start-Sleep -Milliseconds 50}; ('peak_ram_gib={0:F3}' -f ($peak/1GB)) | Set-Content runs\student-cache-distill-s17\ram_reproduction.txt; Get-Content runs\student-cache-distill-s17\ram_reproduction.txt"
```

The recorded result is `1.790 GiB`; the course limit is `4 GiB`. Windows working-set sampling and timing can vary slightly between runs.

### Inference asset size

The final checkpoint plus `student.py` occupies approximately `25.08 MiB` uncompressed:

```text
checkpoint.pt  26281973 bytes
student.py        11076 bytes
combined       26293049 bytes = 25.075 MiB
course limit   64 MiB
```

The teacher checkpoint is a training-only ancestor and is not an inference asset.

## Expected Evaluation Behavior

- Evaluation runs in CPU FP32 and is reproducible without CUDA.
- The model receives only input prefixes and returns normalized natural-log probabilities.
- The neural cache is strictly causal and local to the current 256-token window.
- Cache state is recreated on every call and never crosses windows or examples.
- The model does not access the network during evaluation.
- The supplied data, tokenizer, context length, evaluator, and scoring rule are unchanged.

## Troubleshooting

**`CUDA is unavailable`**  
Use the commands exactly as shown with `--device cpu`. CUDA is not needed for reproduction.

**`ModuleNotFoundError: tokenizers`**  
Activate `.venv` and run `python -m pip install -r requirements.txt`.

**Checkpoint hash mismatch**  
Obtain the matching frozen checkpoint bundle. Do not substitute `checkpoint-step18400.pt`, `checkpoint-step19200.pt`, the teacher checkpoint, or another experimental checkpoint.

**A different CPU time but the same BPB**  
Scoring speed depends on the processor, operating system, thread scheduling, and PyTorch build. Compare the final model with the baseline on the same machine. The BPB and scored-target counts are the primary correctness checks.

## Method Summary

The submitted predictor is a 6-layer, width-288 causal Transformer with six attention heads, RoPE, RMSNorm, SwiGLU, depth-scaled residual initialization, residual dropout during training, and a learned causal within-window neural cache. It has `6565826` parameters. The student was trained with equal weighting between hard-label cross-entropy and KL distillation from a self-trained no-cache teacher. Distillation does not add any teacher weights or computation to evaluation.

For the full experimental method, ablations, training costs, and critical analysis, see `report/MP1_Report_WangXinzhe_u3036801008.pdf`.
