"""Fresh SuperFoil training using the existing conditional U-Net (60 epochs by default).

The fixed held-out test set is monitored after every epoch, but never enters
normalization fitting, gradients, learning-rate decisions or checkpoint selection.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset

import main as m
import train_2d_once as previous
from superfoil_train_test_split import build_split,save_split,DEFAULT_META
from superfoil_test_metrics import evaluate_epoch,plot_history,METRIC_DEFINITIONS


ROOT=Path(__file__).resolve().parent
RUN_KIND='superfoil_fresh_train_fixed_test_v2'


def json_write(path,value):
    def convert(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v).__name__)
    temporary=path.with_name('.'+path.name+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8')
    temporary.replace(path)


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


class CachedJointSuperFoil(Dataset):
    """An I/O cache of unchanged native arrays; model and physical labels are identical."""
    def __init__(self,source,device):
        self.source=source
        self.groups=source.groups
        self.offsets=source.offsets
        self.root=source.root
        values=np.empty((len(source),3,401),dtype=np.float32)
        conditions=np.empty((len(source),3),dtype=np.float32)
        for i,data in enumerate(source.datasets):
            a,b=source.offsets[i:i+2]
            values[a:b]=np.asarray(data.data,dtype=np.float32)
            conditions[a:b]=data.condition_matrix()
        m.finite(values,'cached native SuperFoil values')
        m.finite(conditions,'cached native SuperFoil conditions')
        self.cpu_values=torch.from_numpy(values)
        self.cpu_conditions=torch.from_numpy(conditions)
        self.device=str(device)
        self.values=self.cpu_values.to(device)
        self.conditions=self.cpu_conditions.to(device)

    def __len__(self): return len(self.groups)

    def subset_for_row(self,row):
        return self.source.datasets[bisect_right(self.offsets,int(row))-1].subset

    def __getitem__(self,row):
        values=self.cpu_values[int(row)]
        return {'geometry':values[:1],'conditions':self.cpu_conditions[int(row)],
                'target':values[1:],'row':int(row),'subset':self.subset_for_row(row)}

    def iter_batches(self,rows,batch_size,device,shuffle=False,seed=42):
        rows=torch.as_tensor(np.asarray(rows,dtype=np.int64).copy())
        if shuffle:
            rows=rows[torch.randperm(len(rows),generator=torch.Generator().manual_seed(seed))]
        cached=str(device)==self.device
        values=self.values if cached else self.cpu_values
        conditions=self.conditions if cached else self.cpu_conditions
        indices=rows.to(values.device)
        for start in range(0,len(rows),batch_size):
            ix=indices[start:start+batch_size]
            original=rows[start:start+batch_size]
            block=values.index_select(0,ix)
            yield {'geometry':block[:,:1].to(device),'target':block[:,1:].to(device),
                   'conditions':conditions.index_select(0,ix).to(device),'row':original,
                   'subset':[self.subset_for_row(row) for row in original.tolist()]}


def reset_old_weights(output):
    output=Path(output).resolve()
    m.require(output==ROOT/'weights/2d','Old weight deletion is limited to this project weights/2d')
    m.require(not output.is_symlink(),'Weight directory cannot be a symlink')
    old=[]
    for p in output.iterdir() if output.exists() else []:
        if p.suffix.lower() in previous.WEIGHT_SUFFIXES or (p.name.startswith('.latest-') and p.name.endswith('.partial')):
            m.require(p.is_file() and not p.is_symlink() and p.resolve().parent==output,
                      'Refusing a weight link or unexpected directory')
            old.append({'path':str(p),'sha256':digest_file(p),'bytes':p.stat().st_size})
    # Preserve historical metadata, not the weights the user asked to delete.
    metadata=ROOT/'build'/('previous_2d_training_metadata_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    metadata.mkdir(parents=True,exist_ok=False)
    names=('status.json','training_report.json','train.log','verification.json','history.json','history.jsonl',
           'loss_vs_epoch.png','test_error_percent_vs_epoch.png','reset_record.json','checkpoint_health_check.json',
           'final_test_per_sample.jsonl','final_test_metrics.json','additional_test_diagnostics.json',
           'background_job.json','background_console.log','background_error.log','background_supervisor.lock')
    metadata_paths=[]
    for name in names:
        p=output/name
        if p.exists():
            m.require(p.is_file() and not p.is_symlink(),'Refusing unexpected old metadata link or directory')
            shutil.copy2(p,metadata/name)
            metadata_paths.append(p)
    json_write(metadata/'deleted_weight_records.json',{'user_requested_old_weight_deletion':True,'weights':old})
    for record in old:
        Path(record['path']).unlink()
    for p in metadata_paths:
        p.unlink()
    return old


def save_current(output,model,spec,optimizer,info,history,source_root):
    temporary=output/('.latest-'+uuid.uuid4().hex+'.partial')
    m.save_checkpoint(temporary,'airfoil',model,spec,info,[source_root])
    payload=torch.load(temporary,map_location='cpu',weights_only=True)
    payload.update(optimizer_state=cpu_tree(optimizer.state_dict()),epoch_history=history,
                   cpu_rng_state=torch.get_rng_state(),
                   cuda_rng_states=[x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    torch.save(payload,temporary)
    loaded,_,verified=m.load_checkpoint(temporary,'airfoil')
    m.require(verified['training']['epochs_completed']==info['epochs_completed'],'Saved epoch mismatch')
    m.require(all(bool(torch.isfinite(p).all()) for p in loaded.parameters()),'Non-finite saved model')
    del loaded,verified,payload
    temporary.replace(output/'latest.pt')
    return output/'latest.pt'


def run(args):
    output=Path(args.output_dir).resolve()
    m.protect_destination(output,[args.foil_root])
    m.require(output==ROOT/'weights/2d','This training run uses the established weights/2d destination')
    m.require(args.epochs>0 and args.batch_size>0 and args.test_batch_size>0,'Invalid training setting')
    m.require(not(args.resume and args.reset_old_weights),'Resume and fresh reset are mutually exclusive')
    m.seed_everything(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device=m.selected_device(args.device)
    source=previous.JointSuperFoil(args.foil_root,previous.SUBSETS)
    split=build_split(source,args.meta_path,args.seed)
    train=np.asarray(split['train_rows'],dtype=np.int64)
    test=np.asarray(split['test_rows'],dtype=np.int64)
    m.require(len(train)==9*len(test),'Train/test count ratio is not exactly 9:1')
    split_exists=(output/'split').exists()
    if split_exists:
        for name in ('train_rows','test_rows','excluded_rows','family_ids','test_family_ids'):
            path=output/'split'/(name+'.npy')
            m.require(path.is_file() and not path.is_symlink() and
                      np.array_equal(np.load(path,allow_pickle=False),split[name]),
                      'Existing fixed split differs; refusing to reset weights or silently repartition data')
        existing_manifest=json.loads((output/'split/manifest.json').read_text(encoding='utf-8'))
        m.require(existing_manifest==split['manifest'],'Existing split manifest differs')
    signature=hashlib.sha256(train.tobytes()+test.tobytes()+np.asarray(split['excluded_rows']).tobytes()).hexdigest()
    data=CachedJointSuperFoil(source,device)
    # Source identity is frozen before deletion or parameter updates.
    source_files={str(p):{'sha256':digest_file(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                  for d in source.datasets for p in (d.directory/'data.npy',d.directory/'index.npy')}
    output.mkdir(parents=True,exist_ok=True)
    spec=m.ModelSpec(condition_dim=3)
    model=m.AirfoilPredictor(spec).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    history=[];start_epoch=0
    info={'run_kind':RUN_KIND,'run_id':uuid.uuid4().hex,'stage':'airfoil',
          'epochs_requested':args.epochs,'epochs_completed':0,'training_steps':0,'samples_seen':0,
          'initialization':'fresh_random_model_and_Adam','frozen_split_reused':split_exists,
          'training_samples':len(train),'test_samples':len(test),'source_samples':len(source),
          'excluded_sample_count':len(split['excluded_rows']),'source_root':str(source.root),
          'train_counts':source.counts(train),'test_counts':source.counts(test),
          'foil_subsets':list(previous.SUBSETS),'data_audit':source.audit,'split_signature':signature,
          'train_rows':train.tolist(),'test_rows':test.tolist(),
          'train_shape_ids':np.unique(source.groups[train]).tolist(),
          'test_shape_ids':np.unique(source.groups[test]).tolist(),
          'train_family_ids':np.unique(split['family_ids'][train]).tolist(),
          'test_family_ids':np.unique(split['family_ids'][test]).tolist(),
          'seed':args.seed,'batch_size':args.batch_size,'test_batch_size':args.test_batch_size,
          'device':device,'device_name':torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU',
          'optimizer':'Adam','loss':'MSE','loss_space':'minmax_normalized','normalization':m.NORMALIZATION,
          'learning_rate_schedule':'20-epoch 1e-5 to 1e-4 warmup; then multiply by 0.95 each epoch',
          'checkpoint_selection':'last_completed_epoch_not_best_test','old_weights_loaded':False,
          'test_used_for_gradients':False,'test_used_for_normalization':False,
          'test_used_for_checkpoint_selection':False,'test_evaluated_every_epoch':True,
          'metric_definitions':METRIC_DEFINITIONS,'source_files':source_files,
          'reynolds_input':{'value':m.FOIL_REYNOLDS,'source':'previously_confirmed_project_setting','input_unit':'millions'},
          'foil_channels':['Cp','1000*Cf'],'foil_friction_sign':'Unchanged native SuperFoil convention',
          'cache_policy':'Unchanged float32 source tensors cached on selected device; architecture, batch16 and loss unchanged'}
    if args.resume:
        checkpoint=torch.load(output/'latest.pt',map_location='cpu',weights_only=True)
        prior=checkpoint['training']
        m.require(prior.get('run_kind')==RUN_KIND and prior['split_signature']==signature,
                  'Checkpoint is not this new training run with the frozen split')
        m.require(prior['source_files']==source_files,'Source data changed since this run began')
        m.require(prior['epochs_requested']==args.epochs and prior['batch_size']==args.batch_size and prior['seed']==args.seed,
                  'Resume training settings changed')
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        history=checkpoint['epoch_history'];start_epoch=prior['epochs_completed'];info=prior
    else:
        m.require(args.reset_old_weights or not(output/'latest.pt').exists(),
                  'Old weights exist; use the explicitly requested fresh-reset option')
        deleted=reset_old_weights(output) if args.reset_old_weights else []
        info['deleted_previous_weights']=deleted
        if not split_exists:save_split(split,output/'split')
        json_write(output/'reset_record.json',{'old_weights_deleted':deleted,'new_run_id':info['run_id'],
                   'old_checkpoint_used':False,'fresh_initialization_seed':args.seed})
        json_write(output/'status.json',{'status':'fitting_train_only_normalizers','run_id':info['run_id'],
                    'epochs_requested':args.epochs,'training_samples':len(train),'test_samples':len(test)})
        json_write(output/'verification.json',{'status':'pending_current_run','run_id':info['run_id'],
                  'epochs_completed':0,'epochs_requested':args.epochs})
        m.fit_normalizers(model,data,train,'airfoil',device,args.test_batch_size)
    normalization={name:hashlib.sha256(buffer.detach().cpu().numpy().tobytes()).hexdigest()
                   for name,buffer in model.named_buffers()}
    started=time.monotonic()
    elapsed_before=float(info.get('elapsed_seconds',0.))
    log_path=output/'train.log'
    with log_path.open('a' if args.resume else 'w',encoding='utf-8') as log:
        for epoch in range(start_epoch,args.epochs):
            epoch_start=time.monotonic()
            lr=1e-4*m.learning_rate_factor(epoch,'airfoil')
            for group in optimizer.param_groups:group['lr']=lr
            model.train()
            total=torch.zeros((),dtype=torch.float64,device=device)
            seen=0;steps=0
            for batch in data.iter_batches(train,args.batch_size,device,shuffle=True,seed=args.seed+epoch):
                optimizer.zero_grad(set_to_none=True)
                predicted=model(batch['geometry'],batch['conditions'])
                loss=torch.nn.functional.mse_loss(m.normalize_target(model,predicted,'airfoil'),
                                                  m.normalize_target(model,batch['target'],'airfoil'))
                m.require(bool(torch.isfinite(loss)),'Non-finite training loss')
                loss.backward();optimizer.step()
                n=len(predicted);seen+=n;steps+=1
                total+=loss.detach().double()*n
            m.require(seen==len(train),'Epoch did not use every selected training record')
            train_loss=float(total.cpu())/seen
            train_seconds=time.monotonic()-epoch_start
            evaluate_start=time.monotonic()
            metrics=evaluate_epoch(model,data,test,args.test_batch_size,device,
                                   return_per_sample=epoch==args.epochs-1)
            per_sample=metrics.pop('per_sample',None)
            test_seconds=time.monotonic()-evaluate_start
            m.require(normalization=={name:hashlib.sha256(buffer.detach().cpu().numpy().tobytes()).hexdigest()
                       for name,buffer in model.named_buffers()},'Normalization changed during train/test')
            record={'epoch':epoch+1,'learning_rate':lr,'train_normalized_mse':train_loss,'test':metrics,
                    'train_seconds':train_seconds,'test_seconds':test_seconds,'optimizer_steps':steps,
                    'epoch_seconds_before_plot_and_save':time.monotonic()-epoch_start}
            history.append(record)
            info.update(epochs_completed=epoch+1,training_steps=info['training_steps']+steps,
                        samples_seen=info['samples_seen']+seen,train_mse=train_loss,test=metrics,
                        learning_rate=lr,elapsed_seconds=elapsed_before+time.monotonic()-started)
            latest=save_current(output,model,spec,optimizer,info,history,source.root)
            json_write(output/'history.json',history)
            with (output/'history.jsonl').open('w',encoding='utf-8') as stream:
                for entry in history:stream.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+'\n')
            plots=plot_history(history,output)
            status={'status':'final_verification' if epoch+1==args.epochs else 'training',
                    'run_id':info['run_id'],'epoch':epoch+1,'epochs_requested':args.epochs,
                    'training_samples':len(train),'test_samples':len(test),'learning_rate':lr,
                    'train_normalized_mse':train_loss,'test_normalized_mse':metrics['normalized_mse'],
                    'test_cp_mean_error_percent':metrics['cp']['nmae_range_percent']['mean'],
                    'test_cf_mean_error_percent':metrics['cf']['nmae_range_percent']['mean'],
                    'test_cp_error_range_pp':metrics['cp']['nmae_range_percent']['error_range_percentage_points'],
                    'test_cf_error_range_pp':metrics['cf']['nmae_range_percent']['error_range_percentage_points'],
                    'epoch_wall_seconds':time.monotonic()-epoch_start,
                    'elapsed_seconds':elapsed_before+time.monotonic()-started,
                    'checkpoint':str(latest),'plots':plots}
            json_write(output/'status.json',status)
            json_write(output/'training_report.json',info)
            log.write(json.dumps(status,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps(status,ensure_ascii=False),flush=True)
            if per_sample is not None:
                with (output/'final_test_per_sample.jsonl').open('w',encoding='utf-8') as stream:
                    for value in per_sample:stream.write(json.dumps(value,allow_nan=False)+'\n')
                json_write(output/'final_test_metrics.json',metrics)
    m.require(info['epochs_completed']==args.epochs,'Training did not finish requested epochs')
    unchanged={str(p):{'sha256':digest_file(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                  for d in source.datasets for p in (d.directory/'data.npy',d.directory/'index.npy')}
    m.require(unchanged==source_files,'Read-only SuperFoil source changed during training')
    loaded,_,payload=m.load_checkpoint(output/'latest.pt','airfoil')
    m.require(payload['training']['epochs_completed']==args.epochs,'Final checkpoint is not the final epoch')
    loaded.to(device)
    reread=evaluate_epoch(loaded,data,test,args.test_batch_size,device)
    comparison={name:abs(reread[name]['nmae_range_percent']['mean']-history[-1]['test'][name]['nmae_range_percent']['mean'])
                for name in ('cp','cf')}
    m.require(max(comparison.values())<1e-4,'Reloaded model does not reproduce final test percentages')
    json_write(output/'verification.json',{'status':'passed','run_id':info['run_id'],'epochs_completed':info['epochs_completed'],
       'train_test_ratio':'9:1','split_signature':signature,'source_files_unchanged':True,
       'family_overlap_count':0,'test_used_for_parameter_updates':False,
       'normalizers_unchanged_after_train_only_fit':True,'checkpoint_selection':'last_epoch',
       'checkpoint_reload_mean_percent_abs_difference':comparison,
       'checkpoint_sha256':digest_file(output/'latest.pt')})
    status.update(status='complete',elapsed_seconds=elapsed_before+time.monotonic()-started)
    info.update(status='complete',elapsed_seconds=status['elapsed_seconds'],final_test=reread)
    json_write(output/'training_report.json',info)
    json_write(output/'status.json',status)
    print(json.dumps(status,ensure_ascii=False),flush=True)
    return output/'latest.pt'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foil-root',type=Path,default=m.FOIL_ROOT)
    p.add_argument('--meta-path',type=Path,default=DEFAULT_META)
    p.add_argument('--output-dir',type=Path,default=ROOT/'weights/2d')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--test-batch-size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--reset-old-weights',action='store_true')
    p.add_argument('--resume',action='store_true')
    return p


if __name__=='__main__':print(str(run(parser().parse_args())),flush=True)
