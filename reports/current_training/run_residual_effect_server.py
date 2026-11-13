"""Run the read-only residual ablation through the existing project-scoped SSH helper."""
from pathlib import Path
import argparse, json, shlex, sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'build'))
from check_superwing_server import connect, project_command, REMOTE_ROOT
from run_superwing_200_server import mkdirs, read_json, checked_sftp_path, upload_new_or_identical, download_atomic, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'build/superwing_200_bs8_config.json').read_text())
    job = json.loads(Path(cfg['local_job_path']).read_text())
    assert job['status'] == 'complete'
    expected = job['checkpoints_downloaded']['200']['sha256']
    worker = 'evaluate_initial_residual.py' if args.initial else 'evaluate_residual_effect.py'
    work = cfg['remote_run_dir'] + ('/initial_residual_effect_seed42' if args.initial else '/residual_effect_epoch200')
    if args.initial:
        work += '_' + sha(OUT / worker)[:10]
    script, result = work + '/' + worker, work + '/result.json'
    destination = OUT / ('initial_residual_effect.json' if args.initial else 'residual_effect.json')
    with connect() as client:
        with client.open_sftp() as sftp:
            pipeline = read_json(sftp, cfg['remote_run_dir'] + '/pipeline_status.json')
            assert pipeline['status'] == 'complete'
            for sub in ('', '/home', '/tmp', '/cache', '/config', '/mpl', '/torch', '/cuda', '/triton', '/numba'):
                mkdirs(sftp, work + sub)
            for name in ('main.py', 'train_3d_train_test.py'):
                path = checked_sftp_path(sftp, REMOTE_ROOT + '/' + name)
                with sftp.open(path, 'rb') as f:
                    import hashlib
                    assert hashlib.sha256(f.read()).hexdigest() == sha(ROOT / name)
            upload_new_or_identical(sftp, script, (OUT / worker).read_bytes())
            previous = read_json(sftp, result, missing_ok=True)
        if previous is None:
            env = ['env', 'HOME=' + work + '/home', 'TMPDIR=' + work + '/tmp',
                'TMP=' + work + '/tmp', 'TEMP=' + work + '/tmp',
                'XDG_CACHE_HOME=' + work + '/cache', 'XDG_CONFIG_HOME=' + work + '/config',
                'MPLCONFIGDIR=' + work + '/mpl', 'TORCH_HOME=' + work + '/torch',
                'CUDA_CACHE_PATH=' + work + '/cuda', 'TRITON_CACHE_DIR=' + work + '/triton',
                'NUMBA_CACHE_DIR=' + work + '/numba', 'PYTHONDONTWRITEBYTECODE=1',
                'PYTHONNOUSERSITE=1', 'PYTHONUNBUFFERED=1', 'MPLBACKEND=Agg',
                'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
                'PYTHONPATH=' + cfg['remote_pythonpath']]
            argv = [cfg['remote_python'], '-B', script, '--checkpoint', cfg['remote_output_dir'] + '/epoch_200.pt',
                    '--cache-root', str(Path(cfg['remote_prepare_status']).parent).replace('\\', '/'),
                    '--output', result, '--expected-sha', expected]
            returned = project_command(client, shlex.join(env + argv), timeout=240)
            if returned['exit_code']:
                raise RuntimeError(returned)
            print(returned['stdout'], flush=True)
        with client.open_sftp() as sftp:
            value = read_json(sftp, result)
            assert value['checkpoint_sha256'] == expected
            assert value.get('model_parameters_unchanged', value.get('model_state_unchanged')) and value['optimizer_steps_performed'] == 0
            download_atomic(sftp, result, destination)
        print(json.dumps({'result': str(destination), 'summary': value['summary']}, indent=2))


if __name__ == '__main__':
    main()
