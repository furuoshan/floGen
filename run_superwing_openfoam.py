"""Run project-confined compressible SST RANS cases for effective SuperWing sections.

Inputs are immutable. A case is accepted only after mesh, flow convergence, target
lift, finite surface fields and wall-unit checks. Failed/unattempted cases never
become fabricated pressure or friction labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import os
from pathlib import Path
import re
try:
    import resource
except ImportError:
    resource=None
import shlex
import subprocess
import time

import numpy as np

from openfoam_airfoil_mesh import write_mesh
from openfoam_airfoil_case import write_case
from openfoam_airfoil_post import parse_solution, read_force_history


def inside(root, path):
    root=Path(root).resolve(strict=True)
    p=Path(path)
    if not p.is_absolute():
        p=root/p
    p=p.resolve()
    if p != root and not p.is_relative_to(root):
        raise ValueError('Task path escapes the authorized project')
    return p


def write_json(path, value):
    temporary=path.with_name(path.name+'.partial')
    def convert(v):
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v).__name__)
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=convert)+'\n')
    os.replace(temporary,path)


def foam_command(root, case, command, name, timeout):
    """Launch only the installed OpenFOAM wrapper; all task locations are local."""
    case=inside(root,case)
    for name0 in ('home','tmp','cache'):
        inside(root,case/('.runtime/'+name0)).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,HOME=str(case/'.runtime/home'),TMPDIR=str(case/'.runtime/tmp'),
             TMP=str(case/'.runtime/tmp'),TEMP=str(case/'.runtime/tmp'),
             XDG_CACHE_HOME=str(case/'.runtime/cache'),HISTFILE=str(case/'.runtime/history.disabled'),
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log=inside(root,case/name)
    with log.open('wb') as output:
        p=subprocess.Popen(['/usr/bin/openfoam2412','-c',command],cwd=case,
                           env=env,stdout=output,stderr=subprocess.STDOUT,
                           stdin=subprocess.PIPE,start_new_session=True)
        p.stdin.close()
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid,signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL)
                p.wait()
            raise RuntimeError('OpenFOAM step exceeded its time limit')
    if p.returncode:
        raise RuntimeError(f'OpenFOAM command failed with exit {p.returncode}; see {name}')
    return log


def alpha_initial(x,y,cl):
    n=(len(x)+1)//2
    xx=x[n-1:]
    camber=(y[n-1:]+y[:n][::-1])/2
    slope=np.gradient(camber,xx)
    theta=np.arccos(np.clip(1-2*xx,-1,1))
    integrate=getattr(np,'trapezoid',None) or np.trapz
    a0=float(integrate(slope*(1-np.cos(theta)),theta)/np.pi)
    return float(np.clip(math.degrees(a0+cl/(2*np.pi)),-15,15))


def next_alpha(history,target):
    a,c=history[-1]
    slope=2*np.pi*np.pi/180
    if len(history)>1:
        da=a-history[-2][0]
        if abs(da)>1e-5:
            measured=(c-history[-2][1])/da
            if .01 < measured < .4:
                slope=measured
    return float(np.clip(a+np.clip((target-c)/slope,-2.,2.),-25,25))


def latest_time(case):
    values=[]
    for p in case.iterdir():
        if p.is_dir():
            try:
                values.append((float(p.name),p.name))
            except ValueError:
                pass
    return max(values,default=(0.,'0'))


def snapshot_provenance(record):
    return {n:record[n].item() for n in record.dtype.names}


def configured_case(config,*args,bootstrap=False,**kwargs):
    if config['solver']=='rhoPimpleFoam':
        from openfoam_airfoil_case import write_lts_case
        return write_lts_case(*args,max_co=config['bootstrap_max_co'] if bootstrap else config['max_co'],
                              max_local_dt=config['max_local_dt'],**kwargs)
    return write_case(*args,**kwargs)


def solver_identifier(config):
    return 'OpenFOAM-v2412-rhoPimpleFoam-LTS' if config['solver']=='rhoPimpleFoam' else 'OpenFOAM-v2412-rhoSimpleFoam'


def assess_mesh_log(text):
    """Accept aligned wall anisotropy, never bad volumes/topology/face coupling.

    checkMesh's isotropic AR/determinant warnings are expected in a y+<1,
    single-empty-layer mesh. They are recorded separately under finite bounds.
    All other native failed checks remain fatal.
    """
    if 'Cell volumes OK.' not in text or 'Face pyramids OK.' not in text:
        raise ValueError('Native cell volume or face pyramid check failed')
    for phrase in ('Boundary definition OK.','Cell to face addressing OK.',
                   'Upper triangular ordering OK.','Face vertices OK.',
                   'Number of regions: 1 (OK).','Non-orthogonality check OK.'):
        if phrase not in text: raise ValueError('Missing native mesh acceptance: '+phrase)
    failures=re.findall(r'^\s*\*\*\*(.+)$',text,re.M)
    allowed=('High aspect ratio cells found','Cells with small determinant')
    if any(not line.strip().startswith(allowed) for line in failures):
        raise ValueError('Native mesh contains a failed check beyond aligned wall anisotropy')
    def number(pattern):
        match=re.search(pattern,text)
        if not match: raise ValueError('Missing native quality metric '+pattern)
        return float(match.group(1))
    numeric=r'([+\-0-9.eE]+)'
    metrics={
        'max_aspect_ratio':number(r'Max aspect ratio:\s*'+numeric) if failures else None,
        'min_determinant':number(r'Cell determinant \(wellposedness\)\s*:\s*minimum:\s*'+numeric),
        'max_nonorthogonality':number(r'Mesh non-orthogonality Max:\s*'+numeric),
        'max_skewness':number(r'Max skewness\s*=\s*'+numeric),
        'min_face_weight':number(r'Face interpolation weight\s*:\s*minimum:\s*'+numeric),
        'min_volume_ratio':number(r'Face volume ratio\s*:\s*minimum:\s*'+numeric),
    }
    if ((metrics['max_aspect_ratio'] or 0)>1e5 or metrics['min_determinant']<1e-9
            or metrics['max_nonorthogonality']>65 or metrics['max_skewness']>4
            or metrics['min_face_weight']<.05 or metrics['min_volume_ratio']<.01):
        raise ValueError('Native mesh quantitative acceptance failed')
    failed=re.search(r'Failed\s+(\d+)\s+mesh checks',text)
    if failed and int(failed.group(1))!=len(failures):
        raise ValueError('Unaccounted native mesh failures')
    return {'accepted':True,'native_metrics':metrics,'wall_anisotropy_warnings':failures,
            'policy':'Only bounded aligned high-aspect wall cells and their determinant warning are permitted; all topology, volume, nonorthogonality, skewness and face-coupling checks remain enforced.'}


def run_case(root, input_data, output, split, row, config, submitted_at=None):
    start=time.monotonic()
    x=input_data['x']
    values=input_data[split]
    y=np.array(values['geometry'][row,0],dtype=float)
    conditions=np.array(values['conditions'][row],dtype=float)
    provenance=values['provenance'][row]
    case=inside(root,output/'cases'/split/f'{row:07d}')
    case.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(x.tobytes()+y.tobytes()+conditions.tobytes()+provenance.tobytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    result_file=case/'result.json'
    if result_file.is_file():
        old=json.loads(result_file.read_text())
        if old.get('fingerprint') != fingerprint:
            raise ValueError('Existing case has a different input or configuration')
        if old['status'] in ('success','failed'):
            return dict(old,resumed=True)
    result={'status':'running','split':split,'input_row':int(row),'fingerprint':fingerprint,
            'solver':solver_identifier(config),'turbulence_model':'kOmegaSST',
            'conditions':conditions.tolist(),'provenance':snapshot_provenance(provenance),
            'history':[],'resumed':False,'timings_seconds':{},
            'queue_seconds':0. if submitted_at is None else start-submitted_at}
    write_json(result_file,result)
    np.savez_compressed(case/'request.npz',x=x,geometry=y.astype(np.float32),
                        conditions=conditions,provenance=provenance,
                        config_json=np.asarray(json.dumps(config,sort_keys=True)))
    solution=None
    phase='validate_input'
    try:
        if not np.isfinite(y).all() or not np.isfinite(conditions).all():
            raise ValueError('Non-finite input')
        mach,cl_target,re=conditions
        temperature=float(provenance['temperature_kelvin'])
        alpha=alpha_initial(x,y,cl_target)
        # Requested first-cell height targets y+ below unity at these Reynolds numbers.
        cf_est=.026/max(re,1.)**(1/7)
        first_layer=min(config['first_layer'],2*config['target_yplus']/(re*math.sqrt(cf_est/2)))
        phase='mesh'; phase_start=time.monotonic()
        mesh=write_mesh(case,x,y,radial_cells=config['radial_cells'],first_layer=first_layer,
                        farfield=config['farfield'],span=config['span'],project_root=root)
        result['timings_seconds']['mesh']=time.monotonic()-phase_start
        flow=configured_case(config,case,mach,re,temp=temperature,alpha_deg=alpha,span=config['span'],
                        bootstrap=bool(config['bootstrap_iterations']),
                        end_time=config['bootstrap_iterations'] or config['max_iterations'],project_root=root,
                        convection='upwind' if config['bootstrap_iterations'] else config['convection'],
                        turbulence_intensity=config['turbulence_intensity'],
                        turbulence_length_scale=config['turbulence_length_scale'])
        phase='check_mesh'; phase_start=time.monotonic()
        mesh_log=foam_command(root,case,'checkMesh -allGeometry -allTopology','checkMesh.log',120)
        text=mesh_log.read_text(errors='replace')
        mesh_acceptance=assess_mesh_log(text)
        write_json(case/'mesh_acceptance.json',mesh_acceptance)
        result['timings_seconds']['check_mesh']=time.monotonic()-phase_start
        init=flow['potential_initialization']
        initialization=' && '.join((
            'foamDictionary 0/U -entry boundaryField.farfield -set '+shlex.quote(init['temporarily_set_U_farfield']),
            init['command'],
            'foamDictionary 0/U -entry boundaryField.farfield -set '+shlex.quote(init['restore_U_farfield'])))
        phase='potential_initialization'; phase_start=time.monotonic()
        result['phase']=phase;write_json(result_file,result)
        foam_command(root,case,initialization,'potential_initialization.log',180)
        result['timings_seconds']['initialization']=time.monotonic()-phase_start
        if config['bootstrap_iterations']:
            phase='rans_bootstrap';phase_start=time.monotonic()
            result['phase']=phase;write_json(result_file,result)
            foam_command(root,case,config['solver'],'solver_bootstrap.log',config['case_timeout'])
            result['timings_seconds']['bootstrap']=time.monotonic()-phase_start
            current_time,_=latest_time(case)
            flow=configured_case(config,case,mach,re,temp=temperature,alpha_deg=alpha,span=config['span'],
                end_time=int(current_time)+config['max_iterations'],project_root=root,
                start_from='latestTime',convection=config['convection'],
                turbulence_intensity=config['turbulence_intensity'],
                turbulence_length_scale=config['turbulence_length_scale'])
        history=[]
        solution=None
        for iteration in range(config['max_alpha_iterations']):
            current_time,_=latest_time(case)
            if iteration:
                flow=configured_case(config,case,mach,re,temp=temperature,alpha_deg=alpha,span=config['span'],
                    end_time=int(current_time)+config['max_iterations'],project_root=root,
                    start_from='latestTime',turbulence_intensity=config['turbulence_intensity'],
                    turbulence_length_scale=config['turbulence_length_scale'],convection=config['convection'])
                _,name=latest_time(case)
                angle=math.radians(alpha)
                speed=mach*math.sqrt(1.4*287.05*temperature)
                velocity=f'({speed*math.cos(angle):.15g} {speed*math.sin(angle):.15g} 0)'
                field=str(Path(name)/'U')
                value='uniform '+velocity
                command=' && '.join('foamDictionary '+shlex.quote(field)+' -entry '+entry+' -set '+shlex.quote(value)
                    for entry in ('boundaryField.farfield.freestreamValue','boundaryField.farfield.value'))
                foam_command(root,case,command,f'update_alpha_{iteration:02d}.log',30)
            phase=f'rans_alpha_{iteration:02d}';phase_start=time.monotonic()
            result['phase']=phase;result['current_alpha_deg']=alpha;write_json(result_file,result)
            log=foam_command(root,case,config['solver'],f'solver_alpha_{iteration:02d}.log',config['case_timeout'])
            solve_seconds=time.monotonic()-phase_start
            result['timings_seconds'][phase]=solve_seconds
            # A fully converged flow is required even while searching for the target lift.
            phase='postprocess';phase_start=time.monotonic()
            solution=parse_solution(case,flow,x,cl_target=None,mesh_metadata=mesh,
                log_path=log,force_window=config['force_window'],
                force_drift_tolerance=config['force_drift_tolerance'],require_converged=True)
            result['timings_seconds']['postprocess']=result['timings_seconds'].get('postprocess',0.)+time.monotonic()-phase_start
            if not solution.get('converged',False):
                raise ValueError('Flow convergence checks failed')
            cl=float(solution['cl'])
            history.append((alpha,cl))
            result['history'].append({'alpha_deg':alpha,'cl':cl,'cd':float(solution['cd']),
                                      'log':log.name,'solve_wall_seconds':solve_seconds})
            write_json(result_file,result)
            if abs(cl-cl_target)<=config['cl_tolerance']:
                break
            alpha_new=next_alpha(history,cl_target)
            if abs(alpha_new-alpha)<1e-5:
                raise ValueError('Target CL iteration stalled or exceeded angle bounds')
            alpha=alpha_new
        else:
            raise ValueError('Target CL not reached within the permitted angle iterations')
        labels=np.asarray(solution['labels'],dtype=np.float32)
        if labels.shape != (2,len(x)) or not np.isfinite(labels).all():
            raise ValueError('Invalid pressure/friction labels')
        np.savez_compressed(case/'labels.npz',geometry=y.astype(np.float32),labels=labels,
                            conditions=conditions,provenance=provenance)
        quality={k:v for k,v in solution.items() if k not in ('labels','raw_wall','force_history')}
        quality['residuals']={k:v for k,v in solution['residuals'].items() if k not in ('history','time_block_residuals')}
        write_json(case/'quality.json',quality)
        result.update(status='success',alpha_deg=float(solution.get('alpha_deg',alpha)),
                      cl=float(solution['cl']),cd=float(solution['cd']),labels_file='labels.npz')
    except Exception as exc:
        result.update(status='failed',phase_at_failure=phase,
                      error={'type':type(exc).__name__,'message':str(exc)})
    result['elapsed_seconds']=time.monotonic()-start
    write_json(result_file,result)
    if config.get('compact'):
        from openfoam_case_archive import compact_case
        try:
            result['archive']=compact_case(root,case,solution=solution)
        except Exception as exc:
            result['archive_error']={'type':type(exc).__name__,'message':str(exc)}
        write_json(result_file,result)
    return result


def re_search_failed_checks(text):
    return re.search(r'Failed\s+[1-9]\d*\s+mesh checks',text) is not None


def load_inputs(root,path):
    path=inside(root,path)
    manifest=json.loads((path/'manifest.json').read_text())
    complete=json.loads((path/'COMPLETE.json').read_text())
    if manifest.get('status')!='complete' or manifest['counts']!=complete['counts']:
        raise ValueError('Inputs are not complete')
    data={'x':np.load(path/'x.npy',allow_pickle=False)}
    for split in ('train','test'):
        data[split]={name:np.load(path/split/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                     for name in ('geometry','conditions','provenance')}
    a=set(data['train']['provenance']['family_id'].tolist())
    b=set(data['test']['provenance']['family_id'].tolist())
    if a&b:
        raise ValueError('Train/test baseline-airfoil families overlap')
    return data,manifest


def run(args):
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root=Path(args.project_root).resolve(strict=True)
    output=inside(root,args.output_root)
    output.mkdir(parents=True,exist_ok=True)
    data,manifest=load_inputs(root,args.input_root)
    config={k:getattr(args,k) for k in ('radial_cells','first_layer','target_yplus','farfield','span',
        'max_iterations','max_alpha_iterations','case_timeout','force_window','force_drift_tolerance',
        'cl_tolerance','turbulence_intensity','turbulence_length_scale','convection','bootstrap_iterations',
        'solver','max_co','bootstrap_max_co','max_local_dt')}
    files=('run_superwing_openfoam.py','openfoam_airfoil_mesh.py','openfoam_airfoil_case.py','openfoam_airfoil_post.py')
    config['implementation_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in files}
    config['compact']=args.compact
    if (output/'config.json').exists():
        if json.loads((output/'config.json').read_text())!=config:
            raise ValueError('Existing output uses a different numerical configuration')
    else:
        write_json(output/'config.json',config)
    selected=None
    if args.selection is not None:
        selected=json.loads(inside(root,args.selection).read_text())
        selected=selected.get('selected_input_rows',selected)
        for split in ('train','test'):
            values=selected.get(split,[])
            if len(values)!=len(set(values)) or any(type(v) is not int or v<0 or v>=manifest['counts'][split] for v in values):
                raise ValueError('Invalid benchmark input row selection')
    def jobs():
        if selected is not None:
            for rank in range(max(map(len,selected.values()),default=0)):
                for split in ('train','test'):
                    if rank<len(selected.get(split,[])): yield split,selected[split][rank]
            return
        # Interleave splits so a pilot and early checkpoints cover both sets.
        nmax=max(manifest['counts'].values())
        for row in range(nmax):
            for split in ('train','test'):
                if row>=manifest['counts'][split]:
                    continue
                if args.limit is not None and row>=args.limit:
                    continue
                yield split,row
    job_iterator=iter(jobs())
    summary={'status':'running','solver':args.solver,'model':'kOmegaSST','success':0,'failed':0,
             'source_counts':manifest['counts'],'resumed':0,
             'all_inputs_requested':args.limit is None and args.selection is None,
             'selected_counts':({s:len(selected.get(s,[])) for s in ('train','test')} if selected is not None else
                                {s:min(n,args.limit) if args.limit is not None else n for s,n in manifest['counts'].items()})}
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (output/'status.jsonl').open('a') as events:
        active={}
        def refill():
            while len(active)<args.workers*2:
                try:
                    split,row=next(job_iterator)
                except StopIteration:
                    return
                if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize < args.minimum_free_gb*1024**3:
                    raise RuntimeError('Insufficient project disk space; checkpoint retained for resumption')
                f=pool.submit(run_case,root,data,output,split,row,config,time.monotonic())
                active[f]=(split,row)
        refill()
        while active:
            ready,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for f in ready:
                active.pop(f)
                r=f.result()
                summary[r['status']]+=1
                summary['resumed']+=int(r.get('resumed',False))
                event={k:r[k] for k in ('status','split','input_row','elapsed_seconds')}
                if 'error' in r:
                    event['error']=r['error']
                events.write(json.dumps(event)+'\n');events.flush()
            summary['elapsed_seconds']=time.monotonic()-start
            write_json(output/'run_summary.json',summary)
            if ready:
                print(json.dumps(summary),flush=True)
            completed=summary['success']+summary['failed']
            if completed>=20 and summary['success']==0:
                raise RuntimeError('First 20 cases all failed; inspect the configuration before continuing')
            refill()
    summary.update(status='all_selected_inputs_attempted')
    write_json(output/'run_summary.json',summary)
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--solver',choices=('rhoSimpleFoam','rhoPimpleFoam'),default='rhoPimpleFoam')
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--selection',type=Path,help='Project-local JSON mapping train/test to original input row numbers')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--radial-cells',type=int,default=100)
    p.add_argument('--first-layer',type=float,default=1e-6)
    p.add_argument('--target-yplus',type=float,default=.8)
    p.add_argument('--farfield',type=float,default=20)
    p.add_argument('--span',type=float,default=.01)
    p.add_argument('--max-iterations',type=int,default=3000)
    p.add_argument('--bootstrap-iterations',type=int,default=1000)
    p.add_argument('--bootstrap-max-co',type=float,default=.2)
    p.add_argument('--max-co',type=float,default=.2)
    p.add_argument('--max-local-dt',type=float,default=.001)
    p.add_argument('--max-alpha-iterations',type=int,default=8)
    p.add_argument('--case-timeout',type=float,default=7200)
    p.add_argument('--force-window',type=int,default=100)
    p.add_argument('--force-drift-tolerance',type=float,default=1e-3)
    p.add_argument('--cl-tolerance',type=float,default=1e-3)
    p.add_argument('--turbulence-intensity',type=float,default=.01)
    p.add_argument('--turbulence-length-scale',type=float,default=.01)
    p.add_argument('--minimum-free-gb',type=float,default=10)
    p.add_argument('--convection',choices=('upwind','limitedLinear'),default='limitedLinear')
    p.add_argument('--compact',action='store_true',help='Keep verified wall data and diagnostics, remove reproducible volume intermediates')
    return p


if __name__=='__main__':
    print(json.dumps(run(parser().parse_args())),flush=True)
