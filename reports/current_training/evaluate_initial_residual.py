"""Reconstruct seed-42 initialization and measure its additive perturbation, with no training."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys, time

ROOT = Path.cwd().resolve()
assert str(ROOT) in ('D:\\floGen', 'D:/floGen', '/home/cvlab/Projects/floGen')
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import main as m
import train_3d_train_test as t


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def measure(model, data, rows, batch_size, training, seed):
    model.train(training)
    values, selected_rows = [], []
    for batch in data.iter_batches(rows, batch_size, 'cuda', shuffle=False, seed=seed):
        prior = batch['prior']
        baseline = torch.cat((prior, torch.zeros_like(prior[:, :1])), dim=1)
        full = t.cached_forward(model, batch)
        delta = (full - baseline).double().flatten(2)
        base = baseline.double().flatten(2)
        target = batch['target'].double().flatten(2)
        values.append(torch.stack((delta.abs().mean(2), base.abs().mean(2), target.abs().mean(2),
            target.amax(2) - target.amin(2), (base - target).abs().mean(2),
            (full.double().flatten(2) - target).abs().mean(2)), dim=-1).cpu().numpy())
        selected_rows.extend(batch['row'].cpu().tolist())
    values = np.concatenate(values)
    assert len(values) == len(rows) and selected_rows == list(map(int, rows))
    result = {'mode': 'train' if training else 'eval', 'sample_count': len(rows), 'batch_size': batch_size,
              'source_rows': selected_rows, 'channels': {}}
    for c, (name, multiplier, _) in enumerate(t.CHANNELS):
        v = values[:, c]
        result['channels'][name] = {
            'mean_absolute_perturbation_physical': float(v[:, 0].mean() / multiplier),
            'perturbation_relative_to_prior_percent': t._summary(t._percent(v[:, 0], v[:, 1]), np.array(rows)),
            'perturbation_relative_to_truth_amplitude_percent': t._summary(t._percent(v[:, 0], v[:, 2]), np.array(rows)),
            'perturbation_relative_to_truth_range_percent': t._summary(t._percent(v[:, 0], v[:, 3]), np.array(rows)),
            'prior_nmae_mean_percent': float(t._percent(v[:, 4], v[:, 3]).mean()),
            'initial_combined_nmae_mean_percent': float(t._percent(v[:, 5], v[:, 3]).mean())}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--cache-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--expected-sha', required=True)
    args = p.parse_args()
    checkpoint, cache, output = [t.project_path(x) for x in (args.checkpoint, args.cache_root, args.output)]
    assert t.digest_file(checkpoint) == args.expected_sha
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    info = saved['training']; assert info['epochs_completed'] == 200
    data = t.PreparedWingData(cache)
    assert data.manifest_sha256 == info['cache_manifest_sha256'] and data.split_signature == info['split_signature']
    # Reproduce the original construction order, including the random numbers
    # consumed by the 2-D constructor before its pretrained state is loaded.
    # The final checkpoint embeds that unchanged frozen state, so the standalone
    # old remote 2-D filename is not needed for this read-only reconstruction.
    m.seed_everything(info['seed'])
    torch.set_num_threads(4)
    spec = m.ModelSpec(**saved['spec']); spec.validate()
    airfoil = m.AirfoilPredictor(spec)
    airfoil.load_state_dict({k[len('airfoil.'):]: v for k, v in saved['state_dict'].items()
                            if k.startswith('airfoil.')}, strict=True)
    assert t.state_digest(airfoil) == info['frozen_airfoil_state_sha256']
    model = m.PhysicsEmbeddedWingPredictor(spec, airfoil=airfoil).to('cuda')
    initial_parameters = parameter_digest(model)
    frozen_airfoil = t.state_digest(model.airfoil)
    copied = []
    for name in ('normalizer', 'input_normalizer', 'output_normalizer'):
        module = getattr(model.residual, name)
        assert isinstance(module, m.FeatureNorm)
        for key in ('minimum', 'scale'):
            state_key = 'residual.' + name + '.' + key
            getattr(module, key).copy_(saved['state_dict'][state_key].to('cuda'))
            copied.append(state_key)
    assert parameter_digest(model) == initial_parameters
    assert all(bool((module.running_mean == 0).all()) and bool((module.running_var == 1).all())
               for module in model.residual.modules() if isinstance(module, torch.nn.BatchNorm2d))
    initial_state = t.state_digest(model)
    started = time.monotonic()
    print(json.dumps({'stage': 'initial_eval_all_test', 'test_samples': len(data.test)}), flush=True)
    eval_result = measure(model, data, data.test, 4, False, info['seed'])
    assert t.state_digest(model) == initial_state
    chosen = data.train.copy(); np.random.default_rng(info['seed']).shuffle(chosen)
    first_rows = chosen[:info['batch_size']]
    # Training-mode BatchNorm uses this batch's statistics, like the first actual
    # forward pass. Only disposable in-memory running statistics are updated.
    first = measure(model, data, first_rows, info['batch_size'], True, info['seed'])
    more = measure(model, data, chosen[:20 * info['batch_size']], info['batch_size'], True, info['seed'])
    assert parameter_digest(model) == initial_parameters
    assert t.state_digest(model.airfoil) == frozen_airfoil
    assert t.digest_file(checkpoint) == args.expected_sha
    def concise(value):
        return {c: {'physical_mean_abs_change': x['mean_absolute_perturbation_physical'],
                    'mean_change_over_prior_percent': x['perturbation_relative_to_prior_percent']['mean'],
                    'mean_change_over_truth_range_percent': x['perturbation_relative_to_truth_range_percent']['mean'],
                    'prior_nmae_percent': x['prior_nmae_mean_percent'],
                    'initial_combined_nmae_percent': x['initial_combined_nmae_mean_percent']}
                for c, x in value['channels'].items()}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': info['seed'],
        'checkpoint_sha256': args.expected_sha, 'epoch_of_measured_residual': 0,
        'loaded_trained_3d_parameters': False, 'normalization_buffers_only_copied': copied,
        'normalization_rationale': 'Training-only FeatureNorm bounds are fitted before epoch 1 and verified unchanged throughout training',
        'initial_parameter_sha256': initial_parameters, 'model_parameters_unchanged': True,
        'frozen_2d_unchanged': True, 'checkpoint_files_unchanged': True, 'optimizer_steps_performed': 0,
        'eval_initial_all_test': eval_result, 'first_actual_training_batch': first,
        'first_20_training_batches_fixed_initial_weights': more,
        'train_mode_note': 'No gradient/optimizer updates; BatchNorm training statistics in disposable model only',
        'summary': {'initial_eval_all_test': concise(eval_result), 'first_training_batch': concise(first),
                    'first_20_training_batches': concise(more)}, 'elapsed_seconds': time.monotonic() - started}
    t.atomic_json(output, result); data.close()
    print(json.dumps({'status': 'complete', 'summary': result['summary'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__': main()
