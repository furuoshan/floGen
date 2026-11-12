"""Analytical percent errors, held-out isolation, and epoch plotting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main as m
import superfoil_test_metrics as metrics


class TinyData(Dataset):
    def __init__(self):
        self.targets=torch.tensor([[[-1.,0.,1.],[1.,2.,3.]],
                                   [[-2.,0.,2.],[-2.,0.,2.]]])
        self.groups=np.array([101,202])
    def __len__(self):return len(self.targets)
    def __getitem__(self,row):
        return {'geometry':torch.zeros(1,3),'conditions':torch.tensor([float(row),0.,20.]),
                'target':self.targets[row], 'row':row,
                'subset':('VT-BUF','VT-GU')[row]}


class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy=nn.Parameter(torch.tensor(3.))
        self.register_buffer('predictions',torch.tensor([[[-.8,.2,1.2],[1.1,2.2,3.3]],
                                                       [[-2.2,0.,2.2],[-2.,0.,2.]]]))
        self.output_normalizer=m.FeatureNorm(2,channel_dim=1)
        self.output_normalizer.fit_bounds([0.,0.],[2.,4.])
    def forward(self,geometry,conditions):
        return self.predictions[conditions[:,0].long()]+self.dummy*0


class CachedData(TinyData):
    def __getitem__(self,row):raise AssertionError('DataLoader fallback must not be used')
    def iter_batches(self,rows,batch_size,device,shuffle=False,seed=42):
        self.iter_call={'rows':np.asarray(rows).tolist(),'batch_size':batch_size,'shuffle':shuffle,'seed':seed}
        for start in range(0,len(rows),batch_size):
            yield default_collate([TinyData.__getitem__(self,int(i)) for i in rows[start:start+batch_size]])


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_known_curve_errors_zero_crossings_and_physical_cf_units(self):
        result=metrics.evaluate_epoch(FixedModel(),TinyData(),np.array([1,0]),1,'cpu',True)
        self.assertAlmostEqual(result['cp']['nmae_range_percent']['mean'],20/3,places=5)
        self.assertAlmostEqual(result['cp']['nmae_range_percent']['minimum'],10/3,places=5)
        self.assertAlmostEqual(result['cp']['nmae_range_percent']['maximum'],10,places=5)
        self.assertAlmostEqual(result['cp']['nmae_range_percent']['error_range_percentage_points'],20/3,places=5)
        self.assertEqual(result['cp']['nmae_range_percent']['minimum_sample']['row'],1)
        self.assertEqual(result['cp']['nmae_range_percent']['maximum_sample']['running_id'],101)
        self.assertAlmostEqual(result['cp']['relative_l1_percent']['mean'],20,places=5)
        self.assertAlmostEqual(result['cf']['physical_mae'],.0001,places=9)
        self.assertAlmostEqual(result['cf']['physical_rmse'],np.sqrt(.14/6)/1000,places=9)
        self.assertAlmostEqual(result['cp']['distribution_range_error_percent']['mean'],5,places=5)
        self.assertEqual(result['test_samples'],2)
        self.assertEqual([r['row'] for r in result['per_sample']],[1,0])
        json.dumps(result,allow_nan=False)

    def test_evaluation_does_not_change_parameters_buffers_or_existing_gradients(self):
        model=FixedModel().train()
        model.dummy.grad=torch.tensor(7.)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        metrics.evaluate_epoch(model,TinyData(),[0,1],2,'cpu')
        self.assertFalse(model.training)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertEqual(float(model.dummy.grad),7.)

    def test_cached_batch_path_and_standard_loader_give_identical_summaries(self):
        model=FixedModel()
        standard=metrics.evaluate_epoch(model,TinyData(),[1,0],1,'cpu')
        cached=CachedData()
        result=metrics.evaluate_epoch(model,cached,[1,0],1,'cpu')
        self.assertEqual(result,standard)
        self.assertEqual(cached.iter_call['rows'],[1,0])
        self.assertFalse(cached.iter_call['shuffle'])

    def test_zero_denominators_are_explicitly_undefined_not_epsilon_percentages(self):
        data=TinyData();data.targets[:]=torch.tensor([[1.,1.,1.],[0.,0.,0.]])
        model=FixedModel();model.predictions[:]=torch.tensor([[2.,2.,2.],[0.,0.,0.]])
        result=metrics.evaluate_epoch(model,data,[0,1],2,'cpu',True)
        self.assertIsNone(result['cp']['nmae_range_percent']['mean'])
        self.assertEqual(result['cp']['nmae_range_percent']['undefined_samples'],2)
        self.assertEqual(result['cp']['relative_l1_percent']['mean'],100.)
        self.assertIsNone(result['cf']['relative_l1_percent']['mean'])
        self.assertEqual(result['cf']['physical_rmse'],0.)
        self.assertIsNone(result['per_sample'][0]['cp']['distribution_range_error_percent'])
        json.dumps(result,allow_nan=False)

    def test_nonfinite_predictions_are_rejected(self):
        model=FixedModel();model.predictions[0,0,0]=float('inf')
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            metrics.evaluate_epoch(model,TinyData(),[0],1,'cpu')

    def test_two_cumulative_plot_files_are_atomically_updated(self):
        result=metrics.evaluate_epoch(FixedModel(),TinyData(),[0,1],2,'cpu')
        history=[{'epoch':1,'train_normalized_mse':.02,'test':result},
                 {'epoch':2,'train_normalized_mse':.01,'test':result}]
        with tempfile.TemporaryDirectory() as temp:
            paths=metrics.plot_history(history,temp)
            before={k:Path(p).read_bytes() for k,p in paths.items()}
            for blob in before.values():self.assertTrue(blob.startswith(b'\x89PNG'))
            history.append({'epoch':3,'train_normalized_mse':.005,'test_metrics':result})
            metrics.plot_history(history,temp)
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()),
                             ['loss_vs_epoch.png','test_error_percent_vs_epoch.png'])
            self.assertNotEqual(Path(paths['loss_vs_epoch']).read_bytes(),before['loss_vs_epoch'])


if __name__=='__main__':unittest.main()
