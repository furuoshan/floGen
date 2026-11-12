"""Fresh-run cleanup removes old weights, but preserves data and frozen split."""
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import train_2d_train_test as t


class FreshResetTests(unittest.TestCase):
    def test_60_epoch_default_is_fresh(self):
        args=t.parser().parse_args([])
        self.assertEqual(args.epochs,60)
        self.assertFalse(args.resume)

    def test_old_weights_not_backed_up_and_new_background_not_touched(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            output=root/'weights/2d'
            (output/'split').mkdir(parents=True)
            (root/'build').mkdir()
            (output/'latest.pt').write_bytes(b'old-weight')
            (output/'history.json').write_text('old-history')
            (output/'split/train_rows.npy').write_bytes(b'fixed-split')
            (output/'background_60_job.json').write_text('new-supervisor')
            (output/'notes.txt').write_text('user-notes')
            with patch.object(t,'ROOT',root):
                deleted=t.reset_old_weights(output)
            self.assertEqual(len(deleted),1)
            self.assertFalse((output/'latest.pt').exists())
            self.assertFalse((output/'history.json').exists())
            self.assertEqual((output/'split/train_rows.npy').read_bytes(),b'fixed-split')
            self.assertEqual((output/'background_60_job.json').read_text(),'new-supervisor')
            self.assertEqual((output/'notes.txt').read_text(),'user-notes')
            self.assertFalse(list((root/'build').rglob('*.pt')))
            self.assertEqual(len(list((root/'build').rglob('history.json'))),1)


if __name__=='__main__':unittest.main()
