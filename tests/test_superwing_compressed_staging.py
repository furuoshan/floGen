"""Bounded streaming transfer precision and project-only decompression contracts."""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compressed_staging',ROOT/'build/stage_superwing_200.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CompressedStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='compressed-stage-test-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.array=np.random.default_rng(7).normal(size=(4,3,17,19)).astype(np.float64)

    def test_float32_npy_stream_matches_training_cast_and_preserves_source(self):
        source=self.root/'data.npy';np.save(source,self.array)
        original=source.read_bytes()
        zipped,meta=m.compress_npy(source,self.root/'cache',convert_float32=True,chunk_bytes=72)
        with gzip.open(zipped,'rb') as stream:result=np.load(stream,allow_pickle=False)
        self.assertEqual(result.dtype,np.float32)
        np.testing.assert_array_equal(result,self.array.astype(np.float32))
        self.assertEqual(source.read_bytes(),original)
        self.assertEqual(meta['source_sha256'],hashlib.sha256(original).hexdigest())
        self.assertEqual(meta['compressed_sha256'],hashlib.sha256(zipped.read_bytes()).hexdigest())
        expanded=gzip.decompress(zipped.read_bytes())
        self.assertEqual(meta['expanded_sha256'],hashlib.sha256(expanded).hexdigest())
        self.assertEqual(meta['expanded_size_bytes'],len(expanded))
        cached,again=m.compress_npy(source,self.root/'cache',convert_float32=True,chunk_bytes=72)
        self.assertEqual(cached,zipped);self.assertEqual(again,meta)

    def test_geometry_roundtrip_is_byte_exact(self):
        source=self.root/'origingeom.npy';np.save(source,self.array)
        zipped,meta=m.compress_npy(source,self.root/'cache',chunk_bytes=128)
        self.assertEqual(gzip.decompress(zipped.read_bytes()),source.read_bytes())
        self.assertEqual(meta['uploaded_dtype'],'float64')
        self.assertEqual(meta['source_sha256'],meta['expanded_sha256'])

    def test_decompression_publishes_verified_npy_without_touching_old_raw_partial(self):
        source=self.root/'data.npy';np.save(source,self.array)
        zipped,meta=m.compress_npy(source,self.root/'cache',convert_float32=True,chunk_bytes=72)
        target=self.root/'remote/data.npy';target.parent.mkdir()
        old=target.with_name('data.npy.partial');old.write_bytes(b'old raw upload must remain')
        def execute_local(client,code,timeout=300):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):exec(compile(code,'<remote-decompress-test>','exec'),{})
            return json.loads(output.getvalue())
        with patch.object(m,'REMOTE_ROOT',self.root.as_posix()),patch.object(m,'server_python',side_effect=execute_local):
            result=m.decompress_remote(None,zipped.as_posix(),target.as_posix(),meta)
            second=m.decompress_remote(None,zipped.as_posix(),target.as_posix(),meta)
        self.assertEqual(result['status'],'expanded_sha_verified')
        self.assertEqual(second['status'],'existing_expanded_sha_verified')
        self.assertEqual(old.read_bytes(),b'old raw upload must remain')
        np.testing.assert_array_equal(np.load(target),self.array.astype(np.float32))
        self.assertFalse(target.with_name('data.npy.expanded.partial').exists())

    def test_remote_escape_paths_are_rejected(self):
        for path in ('/tmp/data.npy',m.REMOTE_ROOT+'/../outside.npy',m.REMOTE_ROOT+'/dir/../../outside.npy'):
            with self.subTest(path=path),self.assertRaises(ValueError):m.safe(path)


if __name__=='__main__':unittest.main()
