"""Plot recorded loss and complete-surface test errors without running a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.mplconfig'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'build/superwing_200_bs8_config.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    run = Path(cfg['local_output_dir']).resolve()
    assert run.is_relative_to(ROOT)
    raw = (run / 'history.json').read_bytes()
    history = json.loads(raw)
    assert history and [h['epoch'] for h in history] == list(range(1, len(history) + 1))
    report = json.loads((run / 'training_report.json').read_text(encoding='utf-8'))
    assert report['batch_size'] == int(cfg['train_argv'][cfg['train_argv'].index('--batch-size') + 1])
    count = history[-1]['test']['test_samples']
    assert all(h['test']['test_samples'] == count for h in history)
    epochs = np.array([h['epoch'] for h in history])
    train = np.array([h['train_normalized_mse'] for h in history])
    test = np.array([h['test']['normalized_mse'] for h in history])
    assert np.isfinite(train).all() and np.isfinite(test).all()
    assert (train > 0).all() and (test > 0).all()
    channels = [('cp', 'Cp（压力）', '#2274a5'),
                ('cf_tau', 'Cfτ（轮廓切向摩擦）', '#d87516'),
                ('cf_z', 'Cfz（展向摩擦）', '#168571')]
    curves = {}
    for key, label, color in channels:
        means = np.array([h['test'][key]['nmae_range_percent']['mean'] for h in history])
        maxima = np.array([h['test'][key]['nmae_range_percent']['maximum'] for h in history])
        assert np.isfinite(means).all() and np.isfinite(maxima).all()
        assert (means >= 0).all() and (maxima >= means).all()
        curves[key] = {'mean': means, 'maximum': maxima}

    font = Path('C:/Windows/Fonts/msyh.ttc')
    if not font.is_file():
        font = Path('C:/Windows/Fonts/simhei.ttf')
    assert font.is_file(), 'A Chinese font is required for the figure labels'
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 11,
        'axes.unicode_minus': False, 'axes.edgecolor': '#ccd3d9',
        'axes.labelcolor': '#31404d', 'text.color': '#152b3c',
        'xtick.color': '#465662', 'ytick.color': '#465662',
        'svg.fonttype': 'path', 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(15, 10.5), facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.075, right=.965, bottom=.15, top=.86,
                          hspace=.42, wspace=.23, height_ratios=[.95, 1.05])
    loss_ax = fig.add_subplot(grid[0, :])
    mean_ax = fig.add_subplot(grid[1, 0])
    max_ax = fig.add_subplot(grid[1, 1])
    for ax in (loss_ax, mean_ax, max_ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='both', color='#e3e8ed', linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_xlim(1, max(2, int(epochs[-1]) + 1))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))
        ax.set_xlabel('Epoch', labelpad=7)

    loss_ax.semilogy(epochs, train, color='#2274a5', linewidth=2,
                    label=f'Train · 最新 {train[-1]:.4g}')
    loss_ax.semilogy(epochs, test, color='#d87516', linewidth=2,
                    label=f'Test · 最新 {test[-1]:.4g}')
    loss_ax.scatter([epochs[-1]] * 2, [train[-1], test[-1]],
                    color=['#2274a5', '#d87516'], s=28, zorder=5)
    loss_ax.set_title('A   训练与测试 loss', loc='left', fontsize=14, pad=12)
    loss_ax.set_ylabel('归一化 MSE（对数坐标）', labelpad=10)
    loss_ax.legend(loc='upper right', frameon=False, fontsize=11)

    for key, label, color in channels:
        for ax, field in ((mean_ax, 'mean'), (max_ax, 'maximum')):
            values = curves[key][field]
            ax.plot(epochs, values, color=color, linewidth=1.8,
                    label=f'{label} · 最新 {values[-1]:.3f}%')
            ax.scatter([epochs[-1]], [values[-1]], color=color, s=22, zorder=5)
    mean_ax.set_title('B   平均差异百分比', loc='left', fontsize=14, pad=12)
    max_ax.set_title('C   最大差异百分比（最差样本）', loc='left', fontsize=14, pad=12)
    for ax in (mean_ax, max_ax):
        ax.set_ylim(bottom=0)
        ax.set_ylabel('表面归一化绝对误差 NMAE（%）', labelpad=8)
        ax.legend(loc='upper right', frameon=False, fontsize=10)

    last = int(epochs[-1])
    captured = datetime.now(timezone.utc)
    local_time = captured.astimezone().strftime('%Y-%m-%d %H:%M')
    fig.text(.075, .955, '三维网络训练进展', fontsize=23, weight='bold', ha='left')
    fig.text(.075, .914,
             f'Batch size {report["batch_size"]}  |  已记录 1–{last} / {cfg["epochs"]} 轮  |  完整 test 集 {count:,} 个样本  |  {local_time}（北京时间）',
             fontsize=12, color='#526773', ha='left')
    fig.text(.075, .085,
             '误差口径：每个机翼、每个分量先计算 NMAE = 100 × 表面平均绝对差 ÷ CFD 真值极差。',
             fontsize=11, ha='left', color='#526773')
    fig.text(.075, .053,
             f'平均/最大分别对全部 {count:,} 个测试样本统计；最大值指最差机翼样本的平均误差，不是单个网格点的最大误差。逐轮原值，未平滑。',
             fontsize=10.2, ha='left', color='#526773')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png, svg = OUTPUT / 'epoch_metrics.png', OUTPUT / 'epoch_metrics.svg'
    fig.savefig(png, dpi=190, facecolor='white')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)
    values = {
        'run_id': cfg['run_id'], 'training_run_id': report['run_id'],
        'captured_utc': captured.isoformat(), 'epochs_completed': last,
        'batch_size': report['batch_size'], 'test_samples': count,
        'history_sha256': hashlib.sha256(raw).hexdigest(), 'source_history': str(run / 'history.json'),
        'percentage_metric': history[-1]['test']['metric_definitions']['nmae_range_percent'],
        'maximum_definition': 'Maximum complete-surface mean NMAE across held-out wing samples, not maximum grid-point error',
        'rows': [{'epoch': int(e), 'train_mse': float(train[i]), 'test_mse': float(test[i]),
                  **{key: {f: float(curves[key][f][i]) for f in ('mean', 'maximum')}
                     for key, _, _ in channels}} for i, e in enumerate(epochs)],
        'outputs': {'png': str(png), 'svg': str(svg)}, 'training_or_weight_changes': False}
    (OUTPUT / 'epoch_metrics_data.json').write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'epoch': last, 'test_samples': count, 'latest': values['rows'][-1],
                      'outputs': values['outputs'], 'captured_utc': captured.isoformat()}, indent=2))


if __name__ == '__main__':
    main()
