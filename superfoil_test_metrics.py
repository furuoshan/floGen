"""Held-out 2-D airfoil metrics and cumulative epoch plots.

Percentages are calculated per complete surface curve. Pointwise MAPE is not
used: Cp and Cf can cross zero. A zero curve-range or zero L1 denominator is
explicitly undefined, never replaced with an arbitrary epsilon.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import main as m


METRIC_DEFINITIONS = {
    "nmae_range_percent": "100 * mean(abs(predicted - reference)) / (max(reference) - min(reference)), per sample and channel",
    "error_range_percentage_points": "maximum sample error percentage minus minimum sample error percentage; a spread in percentage points",
    "relative_l1_percent": "100 * sum(abs(predicted - reference)) / sum(abs(reference)), per sample and channel",
    "distribution_range_error_percent": "100 * abs((max(predicted)-min(predicted)) - (max(reference)-min(reference))) / (max(reference)-min(reference))",
    "normalized_mse": "Mean squared difference after applying the checkpoint's training-fitted output min/max normalizer",
    "cf_units": "Dataset values are 1000*Cf; physical Cf MAE and RMSE are divided by 1000; percentage metrics are invariant to this scaling",
    "zero_denominator_policy": "Undefined percentages are null in JSON, excluded from percentage summaries, and counted with their source rows; no epsilon substitution",
}


def _summary(values, identities):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    good = values[valid]
    valid_rows = np.flatnonzero(valid)
    base = {"valid_samples": int(valid.sum()), "undefined_samples": int((~valid).sum()),
            "undefined_source_rows": [identities[i]["row"] for i in np.flatnonzero(~valid)]}
    if not len(good):
        return dict(base, mean=None, minimum=None, maximum=None,
                    error_range_percentage_points=None, p50=None, p95=None, p99=None,
                    minimum_sample=None, maximum_sample=None)
    minimum = int(valid_rows[int(good.argmin())])
    maximum = int(valid_rows[int(good.argmax())])
    return dict(base, mean=float(good.mean()), minimum=float(good.min()), maximum=float(good.max()),
                error_range_percentage_points=float(good.max()-good.min()),
                p50=float(np.quantile(good,.5)), p95=float(np.quantile(good,.95)),
                p99=float(np.quantile(good,.99)), minimum_sample=dict(identities[minimum]),
                maximum_sample=dict(identities[maximum]))


def _percent(numerator, denominator):
    out = np.full_like(np.asarray(numerator,dtype=np.float64), np.nan)
    denominator = np.asarray(denominator,dtype=np.float64)
    positive = denominator > 0
    np.divide(100*np.asarray(numerator,dtype=np.float64), denominator, out=out, where=positive)
    m.require(bool(np.isfinite(out[positive]).all()),"Percentage metric overflow with a nonzero denominator")
    return out


@torch.inference_mode()
def evaluate_epoch(model, dataset, rows, batch_size, device, return_per_sample=False):
    """Evaluate fixed test rows without fitting buffers or changing parameters.

    The model is left in eval mode. A training caller should call model.train()
    explicitly at the start of its next epoch. The primary result paths are
    result['cp'|'cf']['nmae_range_percent']['mean'|'error_range_percentage_points'].
    """
    rows = np.asarray(rows,dtype=np.int64)
    m.require(rows.ndim == 1 and len(rows)>0, "Test rows must be a non-empty vector")
    m.require(batch_size>0 and len(np.unique(rows))==len(rows), "Invalid/repeated test rows")
    m.require(bool(((rows>=0)&(rows<len(dataset))).all()), "Test row outside dataset")
    model.eval()
    blocks, identities = [], []
    points = None
    cursor = 0
    if hasattr(dataset,'iter_batches'):
        loader=dataset.iter_batches(rows,batch_size,device,shuffle=False,seed=42)
    else:
        loader=DataLoader(Subset(dataset,rows.tolist()),batch_size=batch_size,shuffle=False,
                          num_workers=0,drop_last=False)
    for batch in loader:
        batch = m.move_batch(batch,device)
        predicted = model(batch["geometry"],batch["conditions"])
        target = batch["target"]
        m.require(predicted.shape==target.shape and predicted.ndim==3 and predicted.shape[1]==2,
                  "Expected matching [batch,2,surface_point] Cp/1000Cf arrays")
        if points is None:
            points = predicted.shape[2]
        m.require(points==predicted.shape[2] and points>=2,"Inconsistent surface point count")
        norm_error=m.normalize_target(model,predicted,"airfoil")-m.normalize_target(model,target,"airfoil")
        p,t=predicted.double(),target.double()
        abs_error=(p-t).abs()
        truth_range=t.amax(dim=2)-t.amin(dim=2)
        prediction_range=p.amax(dim=2)-p.amin(dim=2)
        # One small CPU transfer per batch; all summary accumulation is float64.
        scores=torch.stack((abs_error.mean(dim=2),abs_error.square().mean(dim=2),truth_range,
                            t.abs().mean(dim=2),prediction_range,norm_error.double().square().mean(dim=2)),dim=-1).cpu().numpy()
        m.require(bool(np.isfinite(scores).all()),"Non-finite test prediction or reference")
        blocks.append(scores)
        count=len(predicted)
        chosen=rows[cursor:cursor+count]
        if "row" in batch:
            reported=batch["row"]
            reported=reported.detach().cpu().numpy() if isinstance(reported,torch.Tensor) else np.asarray(reported)
            m.require(np.array_equal(chosen,reported),"Test loader source identity/order changed")
        subset=batch.get("subset")
        for j,row in enumerate(chosen):
            identity={"row":int(row)}
            if subset is not None:
                identity["subset"]=str(subset[j])
            if hasattr(dataset,"groups"):
                identity["running_id"]=int(dataset.groups[int(row)])
            identities.append(identity)
        cursor+=count
    m.require(cursor==len(rows),"Test sample count changed")
    scores=np.concatenate(blocks,axis=0)
    # Shape: sample, channel, [MAE,MSE,true_range,mean_abs_true,pred_range,norm_MSE].
    nmae=_percent(scores[:,:,0],scores[:,:,2])
    l1=_percent(scores[:,:,0],scores[:,:,3])
    range_error=_percent(np.abs(scores[:,:,4]-scores[:,:,2]),scores[:,:,2])
    result={"test_samples":len(rows),"surface_points":int(points),
            "normalized_mse":float(scores[:,:,5].mean()),"metric_definitions":dict(METRIC_DEFINITIONS),
            "test_used_for_parameter_updates":False,"normalizers_fitted_here":False}
    for c,name,scale in ((0,"cp",1.),(1,"cf",1000.)):
        result[name]={
            "nmae_range_percent":_summary(nmae[:,c],identities),
            "relative_l1_percent":_summary(l1[:,c],identities),
            "distribution_range_error_percent":_summary(range_error[:,c],identities),
            "stored_mae":float(scores[:,c,0].mean()),
            "stored_rmse":float(math.sqrt(scores[:,c,1].mean())),
            "physical_mae":float(scores[:,c,0].mean()/scale),
            "physical_rmse":float(math.sqrt(scores[:,c,1].mean())/scale),
            "stored_unit":"Cp" if c==0 else "1000*Cf", "physical_unit":"Cp" if c==0 else "Cf",
            "degenerate_truth_range_samples":int(np.count_nonzero(scores[:,c,2]==0)),
            "zero_reference_l1_samples":int(np.count_nonzero(scores[:,c,3]==0)),
            "minimum_reference_curve_range_stored":float(scores[:,c,2].min()),
        }
    all_valid=nmae[np.isfinite(nmae)]
    result["combined_equal_sample_channel"]={
        "mean_nmae_range_percent":float(all_valid.mean()) if len(all_valid) else None,
        "minimum_nmae_range_percent":float(all_valid.min()) if len(all_valid) else None,
        "maximum_nmae_range_percent":float(all_valid.max()) if len(all_valid) else None,
        "error_range_percentage_points":float(np.ptp(all_valid)) if len(all_valid) else None,
        "valid_sample_channel_values":int(len(all_valid)),
    }
    if return_per_sample:
        per_sample=[]
        for i,identity in enumerate(identities):
            record=dict(identity)
            for c,name in enumerate(("cp","cf")):
                record[name]={metric:(float(v[i,c]) if np.isfinite(v[i,c]) else None)
                              for metric,v in (("nmae_range_percent",nmae),
                                               ("relative_l1_percent",l1),
                                               ("distribution_range_error_percent",range_error))}
            per_sample.append(record)
        result["per_sample"]=per_sample
    return result


def _atomic_figure(fig,path):
    temporary=path.with_name('.'+path.stem+'-'+uuid.uuid4().hex+'.png')
    try:
        fig.savefig(temporary,dpi=150,bbox_inches='tight',format='png',facecolor='white')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plot_history(history,output_dir):
    """Update two cumulative epoch PNGs atomically.

    Expected records: {'epoch': 1, 'train_normalized_mse': number,
                       'test': evaluate_epoch(...)}.
    """
    m.require(bool(history),"Cannot plot empty epoch history")
    folder=Path(output_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'build/mpl_superfoil_metrics'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=np.asarray([r['epoch'] for r in history],dtype=int)
    m.require(bool((np.diff(epochs)>0).all()),"Epochs must increase")
    def evaluation(record):
        for key in ('test','test_metrics','testmetrics'):
            if key in record:return record[key]
        raise KeyError('Epoch history lacks test metrics')
    train=np.asarray([r['train_normalized_mse'] for r in history],dtype=float)
    test=np.asarray([evaluation(r)['normalized_mse'] for r in history],dtype=float)
    m.require(np.isfinite(train).all() and np.isfinite(test).all(),"Non-finite loss history")
    fig,ax=plt.subplots(figsize=(10.8,5.6))
    ax.plot(epochs,train,label='Train: epoch-average normalized MSE',color='#2864b4',lw=1.8)
    ax.plot(epochs,test,label='Test: end-of-epoch normalized MSE',color='#e17c30',lw=1.8)
    if (train>0).all() and (test>0).all():
        ax.set_yscale('log')
    ax.set(xlabel='Epoch',ylabel='Normalized MSE',title=f'2-D airfoil U-Net loss | completed epoch {epochs[-1]}')
    ax.grid(alpha=.25,which='both');ax.legend(frameon=False)
    fig.tight_layout()
    loss_path=folder/'loss_vs_epoch.png'
    _atomic_figure(fig,loss_path);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12.6,8.4))
    panels=(('nmae_range_percent','mean','Mean test-curve error','NMAE / reference range (%)'),
            ('nmae_range_percent','maximum','Worst test-curve error','NMAE / reference range (%)'),
            ('nmae_range_percent','error_range_percentage_points','Spread of test-curve errors','Maximum - minimum (percentage points)'),
            ('distribution_range_error_percent','mean','Error in predicted curve peak-to-peak range','Mean peak-to-peak error (%)'))
    for ax,(metric,stat,title,ylabel) in zip(axes.flat,panels):
        for name,color,label in (('cp','#2864b4','Cp'),('cf','#e17c30','Cf')):
            values=[evaluation(r)[name][metric][stat] for r in history]
            values=np.asarray([np.nan if v is None else v for v in values],dtype=float)
            ax.plot(epochs,values,label=label,color=color,lw=1.8)
        ax.set(xlabel='Epoch',ylabel=ylabel,title=title)
        ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.suptitle(f'Fixed held-out test set | completed epoch {epochs[-1]}',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    percent_path=folder/'test_error_percent_vs_epoch.png'
    _atomic_figure(fig,percent_path);plt.close(fig)
    return {'loss_vs_epoch':str(loss_path),'test_error_percent_vs_epoch':str(percent_path)}
