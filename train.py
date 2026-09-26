"""Train a model and select checkpoints using validation data only."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def optimizer_groups(model, weight_decay):
    """Apply decay to matrix weights, not biases, norms, gates or temperatures."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def learning_rate_at(step, steps, peak_lr, warmup_steps, min_lr_ratio):
    """Linear warmup followed by cosine decay to a non-zero learning rate."""
    update = step + 1
    if warmup_steps > 0 and update <= warmup_steps:
        return peak_lr * update / warmup_steps
    decay_steps = max(1, steps - warmup_steps)
    progress = min(1.0, max(0.0, (update - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def cpu_state_dict(model):
    """Clone a checkpoint candidate without moving the training model off-device."""
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()}


def distillation_kl(student_scores, teacher_scores, temperature):
    """Mean token-level KL(teacher || student), with standard T^2 scaling."""
    if student_scores.shape != teacher_scores.shape:
        raise ValueError(
            'Teacher and student scores must have the same shape; got '
            f'{tuple(teacher_scores.shape)} and {tuple(student_scores.shape)}.'
        )
    student_logp = F.log_softmax(student_scores.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_scores.float() / temperature, dim=-1)
    teacher_p = teacher_logp.exp()
    token_kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
    return token_kl.mean() * temperature ** 2


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/student_final.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/student-final-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--min-lr-ratio', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--teacher-checkpoint', type=Path,
                   help='Optional self-trained checkpoint for knowledge distillation.')
    p.add_argument('--distill-alpha', type=float, default=0.5,
                   help='Weight on teacher KL; the hard-label weight is 1-alpha.')
    p.add_argument('--distill-temperature', type=float, default=1.0,
                   help='Softmax temperature used for teacher and student distributions.')
    p.add_argument('--select-best', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='With periodic validation, save the lowest-validation-BPB model.')
    p.add_argument('--save-steps', default='',
                   help='Comma-separated update numbers for averaging snapshots.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.eval_every < 0 or args.warmup_steps < 0:
        p.error('Evaluation interval and warmup steps cannot be negative.')
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        p.error('Learning rate and gradient clip must be positive; decay cannot be negative.')
    if not 0 <= args.min_lr_ratio <= 1:
        p.error('Minimum learning-rate ratio must be between 0 and 1.')
    if not 0 <= args.distill_alpha <= 1:
        p.error('Distillation alpha must be between 0 and 1.')
    if args.distill_temperature <= 0:
        p.error('Distillation temperature must be positive.')
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        p.error('AdamW beta values must be in [0, 1).')
    try:
        save_steps = {int(value) for value in args.save_steps.split(',') if value}
    except ValueError:
        p.error('Save steps must be comma-separated integers.')
    if any(step < 1 or step > args.steps for step in save_steps):
        p.error('Every save step must be between 1 and --steps.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    effective_warmup_steps = min(args.warmup_steps, args.steps)
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    teacher = None
    teacher_metadata = None
    if args.teacher_checkpoint is not None:
        teacher_checkpoint = torch.load(
            args.teacher_checkpoint, map_location='cpu', weights_only=True
        )
        if teacher_checkpoint['protocol'] != PROTOCOL:
            raise ValueError('Teacher checkpoint belongs to a different course protocol.')
        teacher, teacher_implementation_sha = make_model(
            teacher_checkpoint['implementation'], teacher_checkpoint['config'], device
        )
        teacher.load_state_dict(teacher_checkpoint['model'])
        teacher.eval()
        teacher.requires_grad_(False)
        teacher_metadata = {
            'checkpoint': str(args.teacher_checkpoint),
            'checkpoint_sha256': sha(args.teacher_checkpoint),
            'implementation': teacher_checkpoint['implementation'],
            'implementation_sha256': teacher_implementation_sha,
            'config': teacher_checkpoint['config'],
            'seed': teacher_checkpoint.get('seed'),
            'train_tokens': teacher_checkpoint.get('train_tokens'),
            'selected_step': teacher_checkpoint.get('selected_step'),
            'training_config': teacher_checkpoint.get('training_config'),
            'parent_teacher': teacher_checkpoint.get('teacher'),
        }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_state = None
    best_validation = None
    best_step = None
    training_config = {
        'steps': args.steps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'warmup_steps': effective_warmup_steps,
        'requested_warmup_steps': args.warmup_steps,
        'min_lr_ratio': args.min_lr_ratio,
        'weight_decay': args.weight_decay,
        'betas': [args.beta1, args.beta2],
        'grad_clip': args.grad_clip,
        'distillation': teacher_metadata is not None,
        'distill_alpha': args.distill_alpha if teacher is not None else 0.0,
        'distill_temperature': (
            args.distill_temperature if teacher is not None else None
        ),
        'teacher': teacher_metadata,
        'eval_every': args.eval_every,
        'select_best': args.select_best,
        'save_steps': sorted(save_steps),
    }
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-256, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = learning_rate_at(
            step, args.steps, args.learning_rate, effective_warmup_steps,
            args.min_lr_ratio
        )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        teacher_scores = None
        if teacher is not None:
            with torch.no_grad(), autocast(device, precision):
                teacher_scores = teacher(inputs)
        with autocast(device, precision):
            student_scores = model(inputs)
            hard_loss = F.cross_entropy(
                student_scores.flatten(0, 1).float(), targets.flatten()
            )
            if teacher_scores is None:
                soft_loss = None
                loss = hard_loss
            else:
                soft_loss = distillation_kl(
                    student_scores, teacher_scores, args.distill_temperature
                )
                loss = (
                    (1.0 - args.distill_alpha) * hard_loss
                    + args.distill_alpha * soft_loss
                )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        current_step = step + 1
        if current_step in save_steps:
            snapshot = args.run_dir/f'checkpoint-step{current_step:05d}.pt'
            torch.save({
                'protocol': PROTOCOL,
                'implementation': args.implementation,
                'config': config,
                'model': cpu_state_dict(model),
                'seed': args.seed,
                'train_tokens': current_step*args.batch_size*256,
                'selected_step': current_step,
                'training_config': training_config,
                'teacher': teacher_metadata,
            }, snapshot)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1, 'loss':loss.item(),
                   'learning_rate':learning_rate, 'grad_norm':grad_norm.item(),
                   'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            if soft_loss is not None:
                row.update(hard_loss=hard_loss.item(),
                           distill_kl=soft_loss.item())
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if (best_validation is None or
                    intermediate['bpb'] < best_validation['bpb']):
                best_validation = dict(intermediate)
                best_step = step + 1
                if args.select_best:
                    best_state = cpu_state_dict(model)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    if not validation_history or validation_history[-1]['step'] != args.steps:
        final_validation = score(model,*data['validation'],device,'fp32')
        final_validation.pop('window_nll_nats')
        if (best_validation is None or
                final_validation['bpb'] < best_validation['bpb']):
            best_validation = dict(final_validation)
            best_step = args.steps
            if args.select_best:
                best_state = cpu_state_dict(model)
    else:
        final_validation = dict(validation_history[-1])
        final_validation.pop('step')

    if args.select_best and best_state is not None:
        model.load_state_dict(best_state)
        validation = best_validation
        selected_step = best_step
    else:
        validation = final_validation
        selected_step = args.steps

    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256,
                'selected_step':selected_step,
                'training_config':training_config,
                'teacher':teacher_metadata},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,
              'final_validation':final_validation,'selected_step':selected_step,
              'training_config':training_config,'teacher':teacher_metadata,
              'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'run_dir': str(args.run_dir),
        'implementation': args.implementation,
        'seed': args.seed,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'train_tokens': result['train_tokens'],
        'parameters': result['parameters'],
        'precision': result['precision'],
        'threads': result['threads'],
        'preparation_seconds': result['preparation_seconds'],
        'train_seconds': result['train_seconds'],
        'process_seconds': result['process_seconds'],
        'validation': result['validation'],
        'final_validation': result['final_validation'],
        'selected_step': result['selected_step'],
        'training_config': result['training_config'],
        'teacher': result['teacher'],
        'config': result['config'],
        'checkpoint_sha256': result['checkpoint_sha256'],
        'implementation_sha256': result['implementation_sha256'],
    }
    (args.run_dir/'experiment_summary.txt').write_text(
        json.dumps(summary, indent=2)+'\n'
    )
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
