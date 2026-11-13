"""Read-only local comparison of the latest downloaded wing checkpoint and CFD.

The two examples are a fixed random held-out sample and the checkpoint's worst
held-out complete-surface Cp NMAE sample. No model or production data are updated.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ['MPLCONFIGDIR'] = str(OUT / 'mplconfig')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import main as m
from foilpara import _state_digest
from low_fidelity_sld import estimate_spanwise_lift


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / 'build/superwing_200_bs8_config.json').read_text())
    job = json.loads(Path(config['local_job_path']).read_text())
    entries = job['checkpoints_downloaded']
    if args.checkpoint:
        weight = args.checkpoint.resolve(strict=True)
        identity = next(v for v in entries.values() if Path(v['path']).resolve() == weight)
    else:
        identity = entries[str(max(map(int, entries)))]; weight = Path(identity['path']).resolve(strict=True)
    assert weight.is_relative_to(ROOT)
    before_weight = file_sha(weight)
    assert before_weight == identity['sha256'] and weight.stat().st_size == identity['size_bytes']
    torch.set_num_threads(2)
    model, spec, payload = m.load_checkpoint(weight, 'wing')
    model.eval()
    epoch = int(payload['training']['epochs_completed'])
    history_row = payload['epoch_history'][-1]
    assert history_row['epoch'] == epoch
    aggregate = history_row['test']
    test_rows = np.asarray(payload['training']['test_rows'], dtype=int)
    random_row = int(np.random.default_rng(42).choice(test_rows))
    worst_row = int(aggregate['cp']['nmae_range_percent']['maximum_source_row'])
    rows = [random_row, worst_row]
    assert all(row in test_rows for row in rows)
    data = m.SuperWingData('D:/SuperWing', target='field', require_reynolds=True)
    before_model = _state_digest(model)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    records = []

    def selected_hash(row, geometry):
        digest = hashlib.sha256()
        for array in (data.index[row], data.configs[geometry], data.centers[geometry],
                      data.vertices[geometry], data.data[row]):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    for r, row in enumerate(rows):
        geometry = int(data.shape_ids[row]); before_source = selected_hash(row, geometry)
        sample = data[row]
        sld, details = estimate_spanwise_lift(data.centers[geometry], data.vertices[geometry],
            mach=float(data.index[row, 3]), aoa_deg=float(data.index[row, 2]), chord_panels=4, span_panels=32)
        assert details['uses_cfd'] is False
        batch = {k: value[None] if isinstance(value, torch.Tensor) else value for k, value in sample.items()}
        batch['sld'] = m.tensor(sld)
        with torch.inference_mode():
            prediction = model(batch)[0].numpy()
        truth = sample['target'].numpy()
        assert np.isfinite(prediction).all()
        metrics = {}
        for c, (name, scale) in enumerate((('Cp', 1), ('Cf_tau', 150), ('Cf_z', 300))):
            reference, pred = truth[c].astype(float) / scale, prediction[c].astype(float) / scale
            mae = np.mean(np.abs(pred - reference))
            metrics[name] = {'nmae_range_percent': float(100 * mae / np.ptp(reference)),
                'relative_l1_percent': float(100 * mae / np.mean(np.abs(reference))), 'physical_mae': float(mae)}
            j = 64; ax = axes[r, c]
            x = sample['query_x'][j].numpy(); lower = sample['lower_mask'][j].numpy(); tail = sample['tail_mask'][j].numpy()
            for side, mask, color in (('lower', lower & ~tail, '#1f77b4'), ('upper', ~lower & ~tail, '#e07b20')):
                ix = np.flatnonzero(mask); ix = ix[np.argsort(x[ix])]
                ax.plot(x[ix], reference[j, ix], color=color, lw=1.8, label='CFD ' + side)
                ax.plot(x[ix], pred[j, ix], color=color, lw=1.8, ls='--', label='Network ' + side)
            ax.set_xlabel('Local x/c'); ax.set_ylabel(name); ax.grid(alpha=.25)
            if name == 'Cp': ax.invert_yaxis()
            selection = 'Seed-42 random held-out case' if r == 0 else f'Worst held-out whole-surface Cp NMAE at epoch {epoch}'
            ax.set_title(f'{selection}\nrow {row}; strip j={j}; {name}', fontsize=10)
            ax.legend(fontsize=8, ncol=2)
        assert selected_hash(row, geometry) == before_source
        records.append({'source_row': row, 'geometry_index': geometry, 'original_id': int(data.groups[row]),
            'selection': 'seed42_random_test_row' if r == 0 else 'worst_whole_surface_cp_nmae_at_selected_checkpoint',
            'test_membership_confirmed': True, 'mach': float(data.index[row, 3]), 'aoa_deg': float(data.index[row, 2]),
            'reynolds': data.reynolds_at(row), 'metrics_entire_surface': metrics,
            'source_selected_arrays_unchanged': True, 'vlm': details})
    fig.suptitle(f'Batch size 8 | epoch {epoch} | independent local inference with geometry-only PyTornado SLD\n'
        'Section examples only; these two cases do not represent the full test-set statistics', fontsize=12)
    target = OUT / 'checkpoint_comparison.png'
    fig.savefig(target, dpi=160, facecolor='white'); plt.close(fig)
    assert _state_digest(model) == before_model and file_sha(weight) == before_weight
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(weight), 'epoch': epoch,
        'checkpoint_sha256': before_weight, 'download_manifest_sha256_verified': True, 'optimizer_steps_performed': 0,
        'model_state_unchanged': True, 'checkpoint_file_unchanged': True, 'source_dataset_read_only': True,
        'comparison_plot': str(target), 'section_j': 64, 'cases': records,
        'aggregate_test_from_checkpoint': aggregate,
        'local_recalculation_note': 'Official VLM and complete network rerun locally on CPU; minor floating-point differences versus server GPU cached-prior metrics are possible.'}
    (OUT / 'comparison_metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for key in ('index', 'centers', 'vertices', 'data', 'reynolds_array'):
        mapping = getattr(getattr(data, key, None), '_mmap', None)
        if mapping is not None: mapping.close()
    print(json.dumps({'plot': str(target), 'epoch': epoch, 'rows': rows,
        'metrics': [record['metrics_entire_surface'] for record in records]}, indent=2))


if __name__ == '__main__':
    main()
