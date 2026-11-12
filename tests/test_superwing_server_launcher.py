"""Remote launch/synchronization safety contracts using a local fake SFTP only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wing_server_launcher', ROOT/'build/run_superwing_200_server.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FakeSFTP:
    def __init__(self, root):
        self.root = root
        self.escapes = set()

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def local(self, path):
        relative = Path(path).relative_to(Path(m.REMOTE_ROOT))
        return self.root/relative
    def stat(self, path): return self.local(path).stat()
    def normalize(self, path): return '/outside/project' if path in self.escapes else path
    def mkdir(self, path): self.local(path).mkdir()
    def open(self, path, mode='r'):
        if mode == 'wx': mode = 'xb'
        return self.local(path).open(mode)
    def get(self, source, destination): shutil.copyfile(self.local(source), destination)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'build', prefix='server-launcher-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fake = FakeSFTP(self.root/'remote')
        self.fake.root.mkdir()

    def write_remote(self, path, value):
        target = self.fake.local(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))

    def test_paths_reject_absolute_and_relative_escape(self):
        for path in ('/tmp/task', m.REMOTE_ROOT+'/../../escape', 'relative/path'):
            with self.subTest(path=path), self.assertRaises(ValueError): m.remote_path(path)
        with self.assertRaises(ValueError):
            m.command_argv([m.REMOTE_ROOT+'/train.py', '--cache-root', '../../elsewhere'], 'train')
        with self.assertRaises(ValueError):
            m.command_argv([m.REMOTE_ROOT+'/train.py', '--cache-root=/tmp/cache'], 'train')

    def test_missing_json_is_optional_but_symlink_escape_is_not(self):
        self.assertIsNone(m.read_json(self.fake, m.REMOTE_ROOT+'/missing.json', missing_ok=True))
        bad = m.REMOTE_ROOT+'/linked'
        self.fake.local(bad).mkdir()
        self.fake.escapes.add(bad)
        with self.assertRaises(ValueError):
            m.checked_sftp_path(self.fake, bad+'/new/file', allow_missing=True)

    def test_checkpoint_download_verifies_checksum_and_never_publishes_bad_bytes(self):
        remote = m.REMOTE_ROOT+'/output/epoch010.pt'
        source = self.fake.local(remote)
        source.parent.mkdir(); source.write_bytes(b'checkpoint bytes')
        local = self.root/'local/epoch010.pt'
        with self.assertRaises(ValueError):
            m.download_atomic(self.fake, remote, local, '0'*64, source.stat().st_size)
        self.assertFalse(local.exists())
        self.assertFalse(list(local.parent.glob('*.download-*')))
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertTrue(m.download_atomic(self.fake, remote, local, checksum, source.stat().st_size))
        self.assertEqual(source.read_bytes(), local.read_bytes())
        self.assertFalse(m.download_atomic(self.fake, remote, local, checksum, source.stat().st_size))

    def test_manifest_escape_is_rejected_before_download(self):
        output = m.REMOTE_ROOT+'/output'
        run = m.REMOTE_ROOT+'/run'
        self.write_remote(run+'/pipeline_status.json', {'status':'training'})
        self.write_remote(output+'/checkpoints.json', {'checkpoints':[{
            'epoch':10, 'path':'../../outside/epoch010.pt', 'sha256':'0'*64, 'size_bytes':4}]})
        client = types.SimpleNamespace(open_sftp=lambda:self.fake)
        cfg = {'remote_run_dir':run,'remote_output_dir':output,'local_output_dir':str(self.root/'local')}
        with self.assertRaises(ValueError): m.sync_once(client,cfg,{})

    def test_training_status_mirrors_published_checkpoint_and_metrics(self):
        output = m.REMOTE_ROOT+'/output'
        run = m.REMOTE_ROOT+'/run'
        self.write_remote(run+'/pipeline_status.json', {'status':'training'})
        self.write_remote(output+'/status.json', {'status':'training','epoch':10})
        checkpoint = self.fake.local(output+'/epoch010.pt'); checkpoint.write_bytes(b'epoch10')
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.write_remote(output+'/checkpoints.json', {'checkpoints':[{
            'epoch':10,'path':'epoch010.pt','sha256':checksum,'size_bytes':7}]})
        self.write_remote(output+'/history.json', [{'epoch':10,'train_loss':.1}])
        client = types.SimpleNamespace(open_sftp=lambda:self.fake)
        local = self.root/'local'
        cfg = {'remote_run_dir':run,'remote_output_dir':output,'local_output_dir':str(local)}
        record = m.sync_once(client,cfg,{})
        self.assertEqual(record['status'],'training')
        self.assertEqual(record['checkpoints_downloaded']['10']['sha256'],checksum)
        self.assertEqual((local/'checkpoints/epoch010.pt').read_bytes(),b'epoch10')
        self.assertEqual(json.loads((local/'status.json').read_text())['epoch'],10)


if __name__=='__main__':unittest.main()
