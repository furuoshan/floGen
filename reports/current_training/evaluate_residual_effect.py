"""Compare a frozen final wing predictor with its prior-only ablation on all test cases."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json, sys, time

ROOT = Path.cwd().resolve()
assert str(ROOT) in ('D:\\floGen', 'D:/floGen', '/home/cvlab/Projects/floGen')
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import main as m
import train_3d_train_test as t


class ZeroResidual(nn.Module):
    """A zero correction with an independent copy of the trained target normalization."""
    def __init__(self, output_normalizer):
        super().__init__()
        self.output_normalizer = copy.deepcopy(output_normalizer)

    def forward(self, geometry_and_prior, conditions):
        return geometry_and_prior.new_zeros((len(geometry_and_prior), 3, 128, 256))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--cache-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--expected-sha', required=True)
    args = p.parse_args()
    checkpoint, cache, output = [t.project_path(x) for x in (args.checkpoint, args.cache_root, args.output)]
    torch.set_num_threads(4)
    before_file = t.digest_file(checkpoint)
    assert before_file == args.expected_sha
    model, spec, payload = m.load_checkpoint(checkpoint, 'wing')
    info = payload['training']
    assert info['epochs_completed'] == 200 and info['sld_source'] == 'pytornado_vlm'
    assert info['sld_network_used'] is False and info['airfoil_frozen'] is True
    data = t.PreparedWingData(cache)
    assert data.manifest_sha256 == info['cache_manifest_sha256']
    assert data.split_signature == info['split_signature']
    assert np.array_equal(data.test, np.asarray(info['test_rows'], dtype=np.int64))
    readonly_paths = [checkpoint, data.manifest_path, data.target_path]
    readonly_paths += [cache / (name + '.npy') for name in
                      ('geometry', 'prior', 'conditions', 'geometry_indices', 'groups', 'sld', 'reynolds')]
    source_stats = {str(x): (x.stat().st_size, x.stat().st_mtime_ns) for x in readonly_paths}
    before_model = t.state_digest(model)
    model.to('cuda').eval()
    ablation = nn.Module()
    ablation.residual = ZeroResidual(model.residual.output_normalizer)
    ablation.to('cuda').eval()
    started = time.monotonic()
    print(json.dumps({'stage': 'evaluating_prior_only', 'test_samples': len(data.test)}), flush=True)
    baseline = t.evaluate_epoch(ablation, data, data.test, 4, 'cuda', 'none')
    print(json.dumps({'stage': 'evaluating_final_network'}), flush=True)
    full = t.evaluate_epoch(model, data, data.test, 4, 'cuda', 'none')
    saved = payload['epoch_history'][-1]['test']
    assert abs(full['normalized_mse'] - saved['normalized_mse']) <= max(1e-8, 1e-5 * abs(saved['normalized_mse']))
    differences = {c: abs(full[c]['nmae_range_percent']['mean'] - saved[c]['nmae_range_percent']['mean'])
                   for c in ('cp', 'cf_tau', 'cf_z')}
    assert max(differences.values()) < 1e-4
    assert t.state_digest(model) == before_model
    assert t.digest_file(checkpoint) == before_file
    assert all((Path(name).stat().st_size, Path(name).stat().st_mtime_ns) == value
               for name, value in source_stats.items())
    result = {'checkpoint': str(checkpoint), 'checkpoint_sha256': before_file, 'epoch': 200,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'test_samples': len(data.test),
        'baseline_definition': 'Sweep-corrected frozen 2-D prior for Cp/Cf_tau, zero Cf_z; identical geometry-only VLM SLD',
        'full_definition': 'Same prior plus the trained 3-D residual; no retraining or changed normalization',
        'prior_only': baseline, 'with_trained_residual': full,
        'checkpoint_reload_mean_percent_abs_difference': differences,
        'model_state_unchanged': True, 'checkpoint_file_unchanged': True,
        'source_arrays_opened_readonly': True, 'source_file_sizes_and_mtimes_unchanged': True,
        'optimizer_steps_performed': 0, 'normalizers_fitted_here': False,
        'elapsed_seconds': time.monotonic() - started,
        'summary': {c: {'prior_nmae_mean_percent': baseline[c]['nmae_range_percent']['mean'],
                        'final_nmae_mean_percent': full[c]['nmae_range_percent']['mean'],
                        'nmae_mean_reduction_percent': 100 * (1 - full[c]['nmae_range_percent']['mean'] / baseline[c]['nmae_range_percent']['mean']),
                        'prior_relative_l1_mean_percent': baseline[c]['relative_l1_percent']['mean'],
                        'final_relative_l1_mean_percent': full[c]['relative_l1_percent']['mean']}
                    for c in ('cp', 'cf_tau', 'cf_z')}}
    t.atomic_json(output, result)
    data.close()
    print(json.dumps({'status': 'complete', 'epoch': 200, 'test_samples': result['test_samples'],
                      'elapsed_seconds': result['elapsed_seconds'], 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    main()
