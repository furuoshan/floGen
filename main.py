"""Physics-embedded 2-D -> 3-D prediction for SuperFoil / SuperWing.

Paper: Yang et al., AIAA Journal 63(6), 2545-2559 (2025), DOI 10.2514/1.J064576.
Reuses floGen's Conv U-Net and ResNet U-Net blocks. Source arrays are read-only.
Entry points include separated training, native prediction, predict-foilpara,
and test-3d (forward only; no optimizer or weight updates).
The 3-D interface exports the original SuperWing layout consumed by D:/dspy.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from flowvae.base_model.conv import convEncoder_Unet, convDecoder_Unet
from flowvae.base_model.resnet import ResnetEncoder_Unet, ResnetDecoder_Unet
from flowvae.base_model.mlp import mlp
from flowvae.vae import Unet
from flowvae.post import get_cellinfo_1d_t, get_dxyforce_2d_t

FOIL_ROOT = Path("D:/Superfoil")
FOIL_REYNOLDS = 2e7  # Actual Reynolds number confirmed by the dataset owner.
WING_ROOT = Path("D:/superwing")
CFD_ROOT = Path("D:/data/foilperf/cfd")
NETWORK_ROOT = Path("D:/data/foilperf/network")
GRID = (128, 256)
CHANNELS = ("cp", "cf_tau_scaled", "cf_z_scaled")
WING_MULTIPLIERS = (1, 150, 300)
CHECKPOINT_FORMAT = "physics_embedded_superwing_v2"
NORMALIZATION = "per_variable_minmax_training_only"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_npy(path: Path):
    require(path.is_file(), f"Missing file: {path}")
    with path.open("rb") as f:
        head = f.read(100)
    require(not head.startswith(b"version https://git-lfs.github.com/spec/v1"),
            f"Git LFS pointer, not downloaded data: {path}")
    require(head.startswith(b"\x93NUMPY"), f"Not an NPY array: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(array.dtype.kind in "fiu", f"Non-numeric array: {path}")
    return array


def finite(value, name):
    require(bool(np.isfinite(value).all()), f"Non-finite values in {name}")


def integer_values(value, upper, name):
    finite(value, name)
    require(bool(((value >= 0) & (value <= upper) & (value == np.floor(value))).all()),
            f"Invalid integer indices in {name}")
    return np.asarray(value, dtype=np.int64)


def tensor(value):
    # Copy: torch must never obtain a writable view of a source memory map.
    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))


def clustcos(n=201):
    k = np.linspace(0, 1, n)
    a0, a1 = .0079, .96
    return (np.cos(a0*np.pi)-np.cos((a0*(1-k)+a1*k)*np.pi)) / (
        np.cos(a0*np.pi)-np.cos(a1*np.pi))


@dataclass
class ModelSpec:
    condition_dim: int = 3  # paper: Mach, CL, Re in millions; 2 is an explicit ablation
    foil_points: int = 401
    span_points: int = 128
    contour_points: int = 256
    section_batch_size: int = 16

    def validate(self):
        require(self.condition_dim in (2, 3), "condition_dim must be 2 or 3")
        require((self.foil_points, self.span_points, self.contour_points) == (401,128,256),
                "This adapter uses SuperFoil 401 points and SuperWing 128x256 cells")
        require(self.section_batch_size > 0, "section_batch_size must be positive")


class FeatureNorm(nn.Module):
    """One training-set min/max per variable, shared over spatial positions.

    Unfitted buffers are the identity transform for synthetic diagnostics.
    Constant variables use a unit scale. Values outside the training range are
    deliberately not clipped, so inverse transforms and extrapolation remain valid.
    """
    def __init__(self, size, channel_dim=-1):
        super().__init__()
        self.channel_dim = channel_dim
        self.register_buffer("minimum", torch.zeros(size))
        self.register_buffer("scale", torch.ones(size))

    @torch.no_grad()
    def bounds(self, values):
        values = torch.as_tensor(values, dtype=torch.float32, device=self.minimum.device)
        require(values.ndim >= 2 and values.numel() > 0, "Empty normalization population")
        axis = self.channel_dim % values.ndim
        require(values.shape[axis] == len(self.minimum), "Normalization variable count mismatch")
        require(bool(torch.isfinite(values).all()), "Non-finite normalization features")
        reduce = tuple(i for i in range(values.ndim) if i != axis)
        return values.amin(dim=reduce), values.amax(dim=reduce)

    @torch.no_grad()
    def fit_bounds(self, minimum, maximum):
        minimum = torch.as_tensor(minimum, dtype=self.minimum.dtype, device=self.minimum.device)
        maximum = torch.as_tensor(maximum, dtype=self.minimum.dtype, device=self.minimum.device)
        require(minimum.shape == maximum.shape == self.minimum.shape, "Normalization bound shape mismatch")
        require(bool(torch.isfinite(minimum).all() and torch.isfinite(maximum).all()),
                "Non-finite normalization bounds")
        span = maximum - minimum
        require(bool(torch.isfinite(span).all() and (span >= 0).all()), "Invalid normalization range")
        self.minimum.copy_(minimum)
        self.scale.copy_(torch.where(span > 0, span, torch.ones_like(span)))

    @torch.no_grad()
    def fit(self, values):
        self.fit_bounds(*self.bounds(values))

    def _broadcast(self, x):
        axis = self.channel_dim % x.ndim
        require(x.shape[axis] == len(self.minimum), "Normalization variable count mismatch")
        shape = [1] * x.ndim
        shape[axis] = len(self.minimum)
        return self.minimum.view(shape), self.scale.view(shape)

    def forward(self, x):
        minimum, scale = self._broadcast(x)
        return (x - minimum) / scale

    def inverse(self, x):
        minimum, scale = self._broadcast(x)
        return x * scale + minimum

    def inverse_delta(self, x):
        # Differences of normalized fields have no minimum/offset term.
        _, scale = self._broadcast(x)
        return x * scale


class AirfoilPredictor(nn.Module):
    """Appendix A1, using 401 rather than 321 points: 100 -> 24 -> 5."""
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.condition_dim = spec.condition_dim
        encoder = convEncoder_Unet(1, [5], [16,32,64], basic_layers={"batchnorm":False})
        decoder = convDecoder_Unet(
            2, [5], [512,256,128,64], sizes=[24,100,401],
            encoder_hidden_dims=[64,32,16,1], basic_layers={"batchnorm":False})
        self.network = Unet(
            latent_dim=32+spec.condition_dim, encoder=encoder, decoder=decoder,
            code_mode="ed", code_dim=spec.condition_dim, decoder_input_layer=[64,128])
        self.normalizer = FeatureNorm(spec.condition_dim)
        self.geometry_normalizer = FeatureNorm(1, channel_dim=1)
        self.output_normalizer = FeatureNorm(2, channel_dim=1)

    def forward(self, geometry, conditions):
        require(geometry.shape[1:] == (1,401), "Airfoil geometry must be (B,1,401)")
        require(conditions.shape == (len(geometry),self.condition_dim), "Airfoil condition mismatch")
        normalized = self.network(self.geometry_normalizer(geometry),
                                  code=self.normalizer(conditions))[0]
        # Restore stored Cp/1000*Cf before interpolation or sweep-theory transforms.
        return self.output_normalizer.inverse(normalized)


class SpanwiseLiftPredictor(nn.Module):
    """Appendix A3: three MLP branches; SuperWing planform has 17 parameters."""
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.normalizer = FeatureNorm(40)
        self.output_normalizer = FeatureNorm(1, channel_dim=1)
        self.conditions = mlp(2,16,[],last_actv=True)
        self.planform = mlp(17,32,[32],last_actv=True)
        self.airfoil = mlp(21,64,[64],last_actv=True)
        self.output = mlp(112,spec.span_points,[64],last_actv=False)

    def forward(self, parameters, conditions):
        # configs[1:18] = 17 planform/spanwise parameters; [18:39] = CST+tcroot.
        x = self.normalizer(torch.cat((conditions,parameters),dim=1))
        normalized = self.output(torch.cat((
            self.conditions(x[:,:2]), self.planform(x[:,2:19]), self.airfoil(x[:,19:])),dim=1))
        return self.output_normalizer.inverse(normalized[:,None,:]).squeeze(1)


class WingResidualPredictor(nn.Module):
    """Appendix A2: preserve span, downsample contour six times, three decoders."""
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.normalizer = FeatureNorm(2)
        self.input_normalizer = FeatureNorm(5, channel_dim=1)
        self.output_normalizer = FeatureNorm(3, channel_dim=1)
        self.encoder = ResnetEncoder_Unet(
            5,[128,4],[32,64,64,128,128,256],strides=(1,2))
        self.decoders = nn.ModuleList([
            ResnetDecoder_Unet(
                out_channels=1,last_size=[128,4],
                hidden_dims=[258,128,128,64,64,32,32],
                encoder_hidden_dims=[256,128,128,64,64,32,5],
                sizes=[[128,k] for k in (8,16,32,64,128,256)],output_size=None)
            for _ in range(3)
        ])

    def forward(self, geometry_and_prior, conditions):
        z = self.encoder(self.input_normalizer(geometry_and_prior))
        c = self.normalizer(conditions)[:,:,None,None].expand(-1,-1,z.shape[2],z.shape[3])
        latent = torch.cat((z,c),dim=1)
        normalized_delta = torch.cat([d(latent,self.encoder.feature_maps) for d in self.decoders],dim=1)
        # The decoder learns (target - prior) / target_range. Its output is a
        # difference, so adding target_min here would corrupt the residual baseline.
        return self.output_normalizer.inverse_delta(normalized_delta)


def torch_interp(xp, values, query):
    """Piecewise-linear interpolation; values (K,C,L), query (K,Q)."""
    xp = xp.to(device=values.device,dtype=values.dtype)
    query = query.clamp(float(xp[0]),float(xp[-1]))
    right = torch.searchsorted(xp, query.contiguous()).clamp(1,len(xp)-1)
    left = right-1
    w = (query-xp[left])/(xp[right]-xp[left])
    gather_left = left[:,None,:].expand(-1,values.shape[1],-1)
    gather_right = right[:,None,:].expand_as(gather_left)
    return values.gather(-1,gather_left)*(1-w[:,None,:]) + values.gather(-1,gather_right)*w[:,None,:]


class PhysicsEmbeddedWingPredictor(nn.Module):
    def __init__(self, spec: ModelSpec, airfoil=None, lift=None):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.airfoil = airfoil if airfoil is not None else AirfoilPredictor(spec)
        self.lift = lift if lift is not None else SpanwiseLiftPredictor(spec)
        self.residual = WingResidualPredictor(spec)
        self.register_buffer("foil_x",torch.tensor(clustcos(),dtype=torch.float32))
        self.freeze_priors()

    def freeze_priors(self):
        for module in (self.airfoil,self.lift):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_priors()
        return self

    @torch.no_grad()
    def prior(self, batch):
        b,j = batch["foil_y"].shape[:2]
        cos = batch["cos_sweep"].reshape(b,1).expand(-1,j)
        if "sld" in batch:
            sld = batch["sld"]
            require(sld.shape == (b,j), "External SLD must be (B,128) sectional CL")
            require(bool(torch.isfinite(sld).all()), "Non-finite external sectional CL")
        else:
            sld = self.lift(batch["parameters"],batch["conditions"])
        ma = batch["conditions"][:,0:1].expand(-1,j)*cos
        cl = sld/cos.square()
        conditions = [ma,cl]
        if self.spec.condition_dim == 3:
            require("reynolds" in batch, "Reynolds number is required by this airfoil checkpoint")
            conditions.append(batch["reynolds"][:,None]*batch["chord"]*cos/1e6)
        conditions = torch.stack(conditions,dim=-1).reshape(b*j,-1)
        y = (batch["foil_y"]/cos[:,:,None]).reshape(b*j,1,401)
        predictions = []
        for start in range(0,len(y),self.spec.section_batch_size):
            predictions.append(self.airfoil(y[start:start+self.spec.section_batch_size],
                                            conditions[start:start+self.spec.section_batch_size]))
        foil = torch.cat(predictions,dim=0)
        query = batch["query_x"].reshape(b*j,256)
        lower = torch_interp(self.foil_x,foil[:,:,:201].flip(-1),query)
        upper = torch_interp(self.foil_x,foil[:,:,200:],query)
        lower_mask = batch["lower_mask"].reshape(b*j,256)
        field = torch.where(lower_mask[:,None,:],lower,upper)
        # SuperFoil Cf follows the local flow sign on both sides. SuperWing Cf_tau
        # follows the contour tangent: lower side has the opposite orientation.
        field[:,1,:] *= torch.where(lower_mask,-1.,1.)
        tail = batch["tail_mask"].reshape(b*j,256)
        te_cp = .5*(foil[:,0,0]+foil[:,0,-1])
        te_cf = .5*(-foil[:,1,0]+foil[:,1,-1])
        te = torch.stack((te_cp,te_cf),dim=1)
        field = torch.where(tail[:,None,:],te[:,:,None],field)
        field[:,1,:] *= 150./1000.  # stored SuperFoil -> stored SuperWing units
        field = field.reshape(b,j,2,256).permute(0,2,1,3)
        field = field*cos[:,None,:,None].square()  # paper Eqs. (3)-(4)
        return field,sld

    def forward(self, batch, return_details=False):
        prior,sld = self.prior(batch)
        delta = self.residual(torch.cat((batch["geometry"],prior),dim=1),batch["conditions"])
        baseline = torch.cat((prior,torch.zeros_like(prior[:,:1])),dim=1)
        result = delta+baseline
        return (result,prior,sld) if return_details else result


def _profile_interp(x, y, query):
    order = np.argsort(x,kind="stable")
    x,y = x[order],y[order]
    x,unique = np.unique(x,return_index=True)
    require(len(x) > 1, "Degenerate sectional contour")
    return np.interp(query,x,y[unique])


def geometry_features(centers, vertices):
    """Exact reference-grid sections at span-cell centers; no source resampling."""
    require(centers.shape == (3,128,256), "Expected SuperWing cell-center coordinates")
    require(vertices.shape == (3,129,257), "Expected origingeom.npy reference vertices")
    finite(centers,"geom0"); finite(vertices,"origingeom")
    section = .5*(np.asarray(vertices[:,:-1],dtype=float)+np.asarray(vertices[:,1:],dtype=float))
    expected = .5*(section[:,:,:-1]+section[:,:,1:])
    require(np.allclose(expected,centers,rtol=0,atol=1e-8),
            "geom0 does not equal the four-vertex cell average")
    require(np.allclose(section[:,:,0],section[:,:,-1],rtol=0,atol=1e-8),
            "SuperWing reference contour must close at the final vertex")
    half_x = clustcos()
    y_features,queries,lower_masks,chords,quarters = [],[],[],[],[]
    for j in range(128):
        points = section[:2,j,:-1].T
        te = .5*(points[0]+points[-1])
        le_idx = int(np.argmin(points[:,0]))
        direction = te-points[le_idx]
        require(np.linalg.norm(direction)>0,"Zero section chord")
        # Refine LE in chord coordinates, rather than assuming world x is untwisted.
        le_idx = int(np.argmin(points@direction))
        require(0 < le_idx < len(points)-1,"Invalid leading-edge position")
        le = points[le_idx]
        direction = te-le
        chord = float(np.linalg.norm(direction))
        require(chord>0,"Zero section chord")
        ex = direction/chord
        ey = np.array([-ex[1],ex[0]])
        xy = np.stack(((points-le)@ex/chord,(points-le)@ey/chord),axis=-1)
        lower = xy[:le_idx+1]
        upper = xy[le_idx:]
        yl = _profile_interp(lower[:,0],lower[:,1],half_x)
        yu = _profile_interp(upper[:,0],upper[:,1],half_x)
        y_features.append(np.r_[yl[::-1],yu[1:]])
        cell_xy = np.asarray(centers[:2,j]).T
        queries.append((cell_xy-le)@ex/chord)
        lower_masks.append(np.arange(256)<le_idx)
        chords.append(chord)
        quarters.append(le+.25*direction)
    quarters = np.asarray(quarters)
    dz = float(section[2,-1,0]-section[2,0,0])
    require(dz>0,"Span stations must increase in z")
    dx = float(quarters[-1,0]-quarters[0,0])
    cosine = dz/math.hypot(dx,dz)
    tail = np.zeros((128,256),dtype=bool);tail[:,-1]=True
    return {
        "geometry":np.asarray(centers,dtype=np.float32),
        "foil_y":np.asarray(y_features,dtype=np.float32),
        "query_x":np.asarray(queries,dtype=np.float32),
        "lower_mask":np.asarray(lower_masks,dtype=bool),
        "tail_mask":tail,"chord":np.asarray(chords,dtype=np.float32),
        "cos_sweep":np.asarray(cosine,dtype=np.float32),
    }


@torch.no_grad()
def sectional_lift(vertices, stored_field, chord, alpha_deg):
    """SLD training labels only, integrating the original reference-grid cells."""
    v=torch.from_numpy(np.array(vertices,dtype=np.float64,copy=True)).permute(1,2,0)
    f=torch.from_numpy(np.array(stored_field,dtype=np.float64,copy=True))
    tangents,_=get_cellinfo_1d_t(v[...,:2])
    tangents=.5*(tangents[1:]+tangents[:-1])
    cf=torch.cat((f[1,:,:,None]*tangents/150.,f[2,:,:,None]/300.),dim=-1)
    forces=get_dxyforce_2d_t(v,cp=f[0],cf=cf).sum(dim=1)
    angle=math.radians(float(alpha_deg))
    lift=forces[:,1]*math.cos(angle)-forces[:,0]*math.sin(angle)
    dz=torch.abs(v[1:,0,2]-v[:-1,0,2])
    area=torch.from_numpy(np.array(chord,dtype=np.float64,copy=True))*dz
    require(bool((area>0).all()),"Degenerate span strip")
    result=lift/area
    require(bool(torch.isfinite(result).all()),"Non-finite integrated SLD")
    return result.numpy()


def training_rows(path, n):
    require(path.is_file(),f"Missing official training-index file: {path}")
    values=np.loadtxt(path,ndmin=1)
    ids=integer_values(values,n-1,str(path))
    require(len(ids)>0 and len(np.unique(ids))==len(ids),"Empty or duplicate training indices")
    return ids


def group_split(rows, groups, fraction=.1, seed=42, max_train_samples=None):
    require(0<fraction<1,"Validation fraction must lie between zero and one")
    unique=np.unique(groups[rows])
    require(len(unique)>=2,"At least two training shapes are required for a held-out validation set")
    rng=np.random.default_rng(seed);rng.shuffle(unique)
    nval=min(len(unique)-1,max(1,int(round(len(unique)*fraction))))
    val=rows[np.isin(groups[rows],unique[:nval])]
    train=rows[~np.isin(groups[rows],unique[:nval])]
    if max_train_samples is not None:
        require(max_train_samples>0,"max_train_samples must be positive")
        train=train[:max_train_samples]
    require(len(train)>0 and len(val)>0,"Empty training or validation split")
    return train,val


class SuperFoilData(Dataset):
    """Original VT arrays; geometry is data[:,0], targets are data[:,1:3]."""
    def __init__(self, root, subset, condition_dim, reynolds=None, reynolds_file=None):
        self.root=Path(root).resolve()
        require(subset in ("VT-BUF","VT-DD","VT-GU"),"Only the three VT surface datasets are supported")
        self.directory=self.root/"VT"/subset
        self.subset=subset
        self.data=read_npy(self.directory/"data.npy")
        self.index=read_npy(self.directory/"index.npy")
        require(self.index.ndim==2 and self.index.shape[1]==13,"SuperFoil VT index must be (N,13)")
        require(self.data.shape==(len(self.index),3,401),"SuperFoil VT data must be (N,3,401)")
        finite(self.index,"SuperFoil index")
        self.groups=integer_values(self.index[:,11],2**53-1,"Running_ID")
        self.condition_dim=condition_dim
        require(condition_dim in (2,3),"condition_dim must be 2 or 3")
        require(not(reynolds is not None and reynolds_file is not None),"Use Reynolds scalar OR file")
        if reynolds_file is not None:
            re=read_npy(Path(reynolds_file))
            require(re.shape==(len(self.index),),"Reynolds file must be (N,) in source row order")
            self.reynolds=np.asarray(re)
            self.reynolds_info={"source":"file_override","path":str(Path(reynolds_file).resolve()),
                                "unit":"actual_reynolds","order":"source_index_rows"}
        elif reynolds is not None:
            self.reynolds=np.full(len(self.index),float(reynolds))
            self.reynolds_info={"source":"scalar_override","value":float(reynolds),"unit":"actual_reynolds"}
        elif condition_dim==3:
            self.reynolds=np.full(len(self.index),FOIL_REYNOLDS)
            self.reynolds_info={"source":"user_confirmed_default","value":FOIL_REYNOLDS,
                                "unit":"actual_reynolds"}
        else:
            self.reynolds=None
            self.reynolds_info={"source":"omitted_condition_dim_2"}
        if condition_dim==3:
            finite(self.reynolds,"SuperFoil Reynolds")
            require(bool((self.reynolds>0).all()),"Reynolds numbers must be positive")

    def __len__(self): return len(self.index)

    def condition_matrix(self):
        if self.subset=="VT-DD":
            columns=[self.index[:,3],self.index[:,6]]  # Mach, target CL; col 9 is AoA
        else:
            columns=[self.index[:,5],self.index[:,9]]
        if self.condition_dim==3:
            columns.append(self.reynolds/1e6)
        return np.column_stack(columns).astype(np.float32)

    def official_train(self):
        return training_rows(self.directory/"traindataindex.txt",len(self))

    def __getitem__(self,row):
        values=self.data[row]
        finite(values,f"SuperFoil field row {row}")
        return {"geometry":tensor(values[:1]),"conditions":tensor(self.condition_matrix_row(row)),
                "target":tensor(values[1:]),"row":int(row)}

    def condition_matrix_row(self,row):
        ix=self.index[row]
        values=[ix[3],ix[6]] if self.subset=="VT-DD" else [ix[5],ix[9]]
        if self.condition_dim==3: values.append(self.reynolds[row]/1e6)
        return values


class SuperWingData(Dataset):
    """Native layout; inference never reads CFD target fields or SLD labels."""
    def __init__(self, root, target="field", require_reynolds=False,
                 reynolds=None,reynolds_file=None, *, geometry_root=None,index_columns=(12,),
                 config_columns=(41,55)):
        self.root=Path(root).resolve()
        self.geometry_root=(Path(geometry_root).resolve() if geometry_root is not None else self.root)
        self.index=read_npy(self.root/"index.npy")
        self.centers=read_npy(self.geometry_root/"geom0.npy")
        self.vertices=read_npy(self.geometry_root/"origingeom.npy")
        self.configs=np.loadtxt(self.geometry_root/"configs.dat",skiprows=1,ndmin=2)
        require(self.index.ndim==2 and self.index.shape[1] in index_columns and len(self.index)>0,
                f"SuperWing index must be non-empty with columns in {index_columns}")
        require(self.configs.ndim==2 and self.configs.shape[1] in config_columns,
                f"configs.dat must have numeric columns in {config_columns}")
        require(self.centers.shape==(len(self.configs),3,128,256),"geom0 shape must be (G,3,128,256)")
        require(self.vertices.shape==(len(self.configs),3,129,257),"origingeom shape must be (G,3,129,257)")
        finite(self.index,"SuperWing index");finite(self.configs,"configs")
        self.shape_ids=integer_values(self.index[:,0],len(self.configs)-1,"shape index")
        single_case=self.configs.shape[1]==41
        if single_case:
            require(len(self.configs)==1 and len(self.index)==1,
                    "Single-case configs requires exactly one geometry and one condition")
            require(bool((self.index[0,:2]==0).all()),
                    "Single-case shape_index and condition_index must both be 0")
        self.condition_ids=integer_values(self.index[:,1],0 if single_case else 7,"condition index")
        self.original_ids=integer_values(self.configs[:,0],2**53-1,"original shape ID")
        require(len(np.unique(self.original_ids))==len(self.original_ids),"Duplicate original shape IDs")
        keys=list(zip(self.original_ids[self.shape_ids].tolist(),self.condition_ids.tolist()))
        require(len(set(keys))==len(keys),"Duplicate shape/condition samples")
        aoa_columns=40 if single_case else 47+self.condition_ids
        mach_columns=39 if single_case else 39+self.condition_ids
        require(np.allclose(self.index[:,2],self.configs[self.shape_ids,aoa_columns],
                            rtol=0,atol=1e-8),"AoA does not match configs")
        require(np.allclose(self.index[:,3],self.configs[self.shape_ids,mach_columns],
                            rtol=0,atol=1e-8),"Mach does not match configs")
        self.groups=self.original_ids[self.shape_ids]
        self.target=target
        require(target in ("field","sld","none"),"Unknown target")
        self.data=read_npy(self.root/"data.npy") if target!="none" else None
        if self.data is not None:
            require(self.data.shape==(len(self.index),3,128,256),"data shape must be (N,3,128,256)")
        self.require_reynolds=require_reynolds
        require(not(reynolds is not None and reynolds_file is not None),"Use Reynolds scalar OR file")
        self.reynolds_override=float(reynolds) if reynolds is not None else None
        self.reynolds_array=read_npy(Path(reynolds_file)) if reynolds_file is not None else None
        if self.reynolds_array is not None:
            require(self.reynolds_array.shape==(len(self.index),),"Reynolds file must be (N,)")
        self.logs={}
        if require_reynolds and reynolds is None and reynolds_file is None:
            logfile=self.root/"running_info.jsonl"
            require(logfile.is_file(),"Reynolds unavailable: provide --reynolds/--reynolds-file or original running_info.jsonl")
            with logfile.open(encoding="utf-8") as f:
                self.logs={int(row["running_id"]):row for row in map(json.loads,f)}
        self._features=OrderedDict()

    def __len__(self): return len(self.index)

    def condition_matrix(self):
        return self.index[:,[3,2]].astype(np.float32)  # model order: Mach, AoA

    def parameter_matrix(self):
        return self.configs[self.shape_ids,1:39].astype(np.float32)

    def official_train(self):
        return training_rows(self.root/"training_samples_index.txt",len(self))

    def geometry_at(self,g):
        if g not in self._features:
            self._features[g]=geometry_features(self.centers[g],self.vertices[g])
            if len(self._features)>16: self._features.popitem(last=False)
        self._features.move_to_end(g)
        return self._features[g]

    def reynolds_at(self,row):
        if self.reynolds_override is not None: value=self.reynolds_override
        elif self.reynolds_array is not None: value=float(self.reynolds_array[row])
        else:
            original=int(self.groups[row]); k=int(self.condition_ids[row])
            try:
                value=float(self.logs[original][f"{k}_running_info"]["working_conditions"]["re"])
            except KeyError as error:
                raise ValueError(f"Missing Reynolds for original_id={original}, condition={k}") from error
        require(math.isfinite(value) and value>0,"Invalid Reynolds number")
        return value

    def __getitem__(self,row):
        g=int(self.shape_ids[row]);features=self.geometry_at(g)
        result={key:(torch.from_numpy(value.copy()) if value.dtype==bool else tensor(value))
                for key,value in features.items()}
        result.update(parameters=tensor(self.configs[g,1:39]),
                      conditions=tensor(self.index[row,[3,2]]),row=int(row))
        if self.require_reynolds: result["reynolds"]=tensor(self.reynolds_at(row))
        if self.target!="none":
            field=self.data[row];finite(field,f"SuperWing field row {row}")
            if self.target=="field": result["target"]=tensor(field)
            else:
                result["target"]=tensor(sectional_lift(
                    self.vertices[g],field,features["chord"],self.index[row,2]))
        return result


def move_batch(batch, device):
    return {key:(value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value)
            for key,value in batch.items()}


def stage_forward(model,batch,stage):
    if stage=="airfoil": return model(batch["geometry"],batch["conditions"])
    if stage=="sld": return model(batch["parameters"],batch["conditions"])
    return model(batch)


def normalize_target(model, values, stage):
    """Use the same per-variable target ranges for training and validation MSE."""
    normalizer = model.residual.output_normalizer if stage == "wing" else model.output_normalizer
    if stage == "sld":
        return normalizer(values[:,None,:]).squeeze(1)
    return normalizer(values)


@torch.no_grad()
def fit_normalizers(model, dataset, rows, stage, device="cpu", batch_size=16):
    """Stream only the selected training rows; never fit on held-out samples.

    Spatial channels share one min/max across all their training points. Wing
    prior-channel bounds come from the frozen predictors, not CFD target fields.
    """
    require(len(rows) > 0, "Empty normalization population")
    model.to(device).eval()
    bounds = {}

    def collect(normalizer, values):
        lo, hi = normalizer.bounds(values)
        if normalizer in bounds:
            old_lo, old_hi = bounds[normalizer]
            lo, hi = torch.minimum(old_lo, lo), torch.maximum(old_hi, hi)
        bounds[normalizer] = (lo, hi)

    loader = DataLoader(Subset(dataset, list(map(int, rows))), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    for batch in loader:
        batch = move_batch(batch, device)
        if stage == "airfoil":
            collect(model.normalizer, batch["conditions"])
            collect(model.geometry_normalizer, batch["geometry"])
            collect(model.output_normalizer, batch["target"])
        elif stage == "sld":
            collect(model.normalizer, torch.cat((batch["conditions"], batch["parameters"]), dim=1))
            collect(model.output_normalizer, batch["target"][:,None,:])
        else:
            require(stage == "wing", "Unknown normalization stage")
            prior, _ = model.prior(batch)
            collect(model.residual.normalizer, batch["conditions"])
            collect(model.residual.input_normalizer, torch.cat((batch["geometry"], prior), dim=1))
            collect(model.residual.output_normalizer, batch["target"])
    for normalizer, (lo, hi) in bounds.items():
        normalizer.fit_bounds(lo, hi)


def validation_loss(model,loader,device,stage):
    model.eval();total=count=0
    with torch.inference_mode():
        for batch in loader:
            batch=move_batch(batch,device)
            prediction = stage_forward(model,batch,stage)
            error = normalize_target(model, prediction, stage) - normalize_target(model, batch["target"], stage)
            require(bool(torch.isfinite(error).all()),"Non-finite validation output")
            total+=float(error.square().sum());count+=error.numel()
    require(count>0,"Empty validation set")
    return total/count


def learning_rate_factor(epoch,stage):
    # Paper: wing schedule is 3x the airfoil schedule. The repository's SLD
    # example uses warmup_lr_4, hence a 4x SLD schedule.
    scale={"airfoil":1,"wing":3,"sld":4}[stage]
    warmup=20*scale
    return .1+.9*epoch/(warmup-1) if epoch<warmup else .95**((epoch-(warmup-1))/scale)


def validate_sld_split(sld_training,validation_shape_ids):
    require("train_shape_ids" in sld_training,
            "SLD checkpoint lacks geometry split metadata; retrain it with this entry point")
    overlap=set(map(int,sld_training["train_shape_ids"])) & set(map(int,validation_shape_ids))
    require(not overlap,
            f"SLD training shapes overlap wing validation shapes: {sorted(overlap)[:10]}. "
            "Use compatible geometry splits to avoid validation leakage.")


def fit_model(model,dataset,train_rows,val_rows,stage,device,epochs,batch_size,
              learning_rate=1e-4,max_steps=None):
    require(epochs>0 and batch_size>0 and learning_rate>0,"Invalid training settings")
    if max_steps is not None: require(max_steps>0,"max_steps must be positive")
    fit_normalizers(model, dataset, train_rows, stage, device, batch_size)
    train_loader=DataLoader(Subset(dataset,train_rows.tolist()),batch_size=batch_size,
                            shuffle=True,num_workers=0)
    val_loader=DataLoader(Subset(dataset,val_rows.tolist()),batch_size=batch_size,
                          shuffle=False,num_workers=0)
    model.to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                lr=learning_rate)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: learning_rate_factor(epoch,stage))
    best=float("inf");best_state=None;steps=0;history=[]
    for epoch in range(epochs):
        model.train();total=count=0
        for batch in train_loader:
            batch=move_batch(batch,device);optimizer.zero_grad(set_to_none=True)
            prediction=stage_forward(model,batch,stage)
            loss=nn.functional.mse_loss(normalize_target(model, prediction, stage),
                                        normalize_target(model, batch["target"], stage))
            require(bool(torch.isfinite(loss)),"Non-finite training loss")
            loss.backward();optimizer.step();steps+=1
            total+=float(loss.detach())*prediction.numel();count+=prediction.numel()
            if max_steps is not None and steps>=max_steps: break
        valid=validation_loss(model,val_loader,device,stage)
        record={"epoch":epoch+1,"steps":steps,"train_mse":total/count,"validation_mse":valid,
                "learning_rate":scheduler.get_last_lr()[0]}
        history.append(record);print(json.dumps(record),flush=True)
        if valid<best:
            best=valid
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step()
        if max_steps is not None and steps>=max_steps: break
    require(best_state is not None,"Training completed no usable steps")
    model.load_state_dict(best_state)
    return {"training_steps":steps,"best_validation_mse":best,"history":history,
            "normalization":NORMALIZATION,"loss_space":"minmax_normalized",
            "train_rows":train_rows.tolist(),"validation_rows":val_rows.tolist()}


def protect_destination(destination,sources):
    destination=Path(destination).resolve()
    for source in sources:
        source=Path(source).resolve()
        require(destination!=source and source not in destination.parents,
                f"Output must not overwrite or be placed inside input data: {source}")


def prediction_directory_available(path):
    """Allow a new output directory or one containing only its format document."""
    path=Path(path)
    if not path.exists():
        return not path.is_symlink()
    if not path.is_dir():
        return False
    entries=list(path.iterdir())
    return not entries or (len(entries)==1 and entries[0].name=="format.md" and
                           entries[0].is_file() and not entries[0].is_symlink())


def save_checkpoint(path,stage,model,spec,training_info,source_roots):
    path=Path(path).resolve();protect_destination(path,source_roots)
    require(not path.exists(),f"Checkpoint already exists; choose a new path: {path}")
    require(training_info.get("training_steps",0)>0,"Cannot label an untrained model as trained")
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={"format":CHECKPOINT_FORMAT,"stage":stage,"spec":asdict(spec),
             "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
             "training":training_info,
             "conventions":{"wing_channels":list(CHANNELS),"wing_multipliers":[1,150,300],
                            "foil_channels":["cp","cf_times_1000"],"foil_cf_multiplier":1000,
                            "reynolds_input_unit":"millions","normalization":NORMALIZATION}}
    with path.open("xb") as output: torch.save(payload,output)
    return path


def load_checkpoint(path,expected_stage=None):
    payload=torch.load(Path(path),map_location="cpu",weights_only=True)
    require(payload.get("format")==CHECKPOINT_FORMAT,
            "Incompatible checkpoint format: retrain with this entry point's min-max normalization")
    stage=payload.get("stage")
    require(expected_stage is None or stage==expected_stage,
            f"Expected {expected_stage} checkpoint, got {stage}")
    require(payload.get("training",{}).get("training_steps",0)>0,"Checkpoint has not been trained")
    require(payload.get("conventions",{}).get("wing_multipliers")==[1,150,300],
            "Checkpoint has incompatible field conventions")
    require(payload.get("conventions",{}).get("normalization")==NORMALIZATION,
            "Checkpoint has incompatible normalization conventions")
    spec=ModelSpec(**payload["spec"]);spec.validate()
    constructors={"airfoil":AirfoilPredictor,"sld":SpanwiseLiftPredictor,
                  "wing":PhysicsEmbeddedWingPredictor}
    require(stage in constructors,"Unknown checkpoint stage")
    model=constructors[stage](spec)
    model.load_state_dict(payload["state_dict"],strict=True)
    return model,spec,payload


def build_wing(airfoil_checkpoint,sld_checkpoint):
    airfoil,spec,a=load_checkpoint(airfoil_checkpoint,"airfoil")
    lift,lift_spec,s=load_checkpoint(sld_checkpoint,"sld")
    require(lift_spec.span_points==spec.span_points,"SLD grid does not match airfoil/wing grid")
    require("train_shape_ids" in s["training"],"SLD checkpoint lacks geometry split metadata")
    return PhysicsEmbeddedWingPredictor(spec,airfoil,lift),spec,{
        "sld_train_shape_ids":s["training"]["train_shape_ids"],
        "airfoil_training_steps":a["training"]["training_steps"],
        "airfoil_reynolds_input":a["training"].get("reynolds_input"),
        "sld_training_steps":s["training"]["training_steps"],
        "airfoil_checkpoint":str(Path(airfoil_checkpoint).resolve()),
        "sld_checkpoint":str(Path(sld_checkpoint).resolve())}


@torch.inference_mode()
def predict_directory(model,spec,checkpoint_info,input_root,output_root,device="cpu",
                      batch_size=1,reynolds=None,reynolds_file=None):
    require(checkpoint_info.get("stage")=="wing" and
            checkpoint_info.get("training",{}).get("training_steps",0)>0,
            "Prediction export requires a trained wing checkpoint")
    trained=checkpoint_info["training"]
    require(trained.get("airfoil_training_steps",0)>0 and trained.get("sld_training_steps",0)>0,
            "Wing checkpoint must record trained airfoil and SLD predecessors")
    dataset=SuperWingData(input_root,target="none",require_reynolds=spec.condition_dim==3,
                          reynolds=reynolds,reynolds_file=reynolds_file)
    output_root=Path(output_root).resolve();protect_destination(output_root,[dataset.root])
    require(output_root not in dataset.root.parents,"Output cannot be an ancestor of input data")
    require(prediction_directory_available(output_root),
            f"Prediction directory must be empty or contain only a regular format.md file: {output_root}")
    require(batch_size>0,"batch_size must be positive")
    output_root.mkdir(parents=True,exist_ok=True)
    marker=output_root/"prediction_info.json"
    metadata={"status":"in_progress","source_root":str(dataset.root),
              "sample_count":len(dataset),"channels":list(CHANNELS),"stored_multipliers":[1,150,300],
              "condition_dim":spec.condition_dim,
              "index_coefficients_role":"copied_source_metadata_not_network_predictions",
              "training_steps":trained["training_steps"]}
    marker.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    partial=output_root/"data.npy.partial"
    output=np.lib.format.open_memmap(partial,mode="w+",dtype=np.float32,
                                    shape=(len(dataset),3,128,256))
    model.to(device).eval()
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    offset=0
    for batch in loader:
        batch=move_batch(batch,device)
        prediction=model(batch)
        require(bool(torch.isfinite(prediction).all()),"Non-finite neural prediction; export left incomplete")
        values=prediction.cpu().numpy()
        output[offset:offset+len(values)]=values;offset+=len(values)
    require(offset==len(dataset),"Prediction row count mismatch")
    output.flush();del output
    # Preserve full metadata byte-for-byte, including original row/shape numbering.
    for name in ("index.npy","configs.dat","geom0.npy","origingeom.npy"):
        shutil.copy2(dataset.root/name,output_root/name)
    if (dataset.root/"running_info.jsonl").is_file():
        shutil.copy2(dataset.root/"running_info.jsonl",output_root/"running_info.jsonl")
    partial.replace(output_root/"data.npy")
    metadata["status"]="complete"
    marker.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return output_root/"data.npy"


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def selected_device(name):
    if name=="auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda"):
        require(torch.cuda.is_available(),"CUDA was requested but is unavailable")
    return name


def add_training_options(parser,epochs,batch_size):
    parser.add_argument("--checkpoint",type=Path,required=True,help="New output checkpoint; never overwrites")
    parser.add_argument("--epochs",type=int,default=epochs)
    parser.add_argument("--batch-size",type=int,default=batch_size)
    parser.add_argument("--learning-rate",type=float,default=1e-4)
    parser.add_argument("--validation-fraction",type=float,default=.1)
    parser.add_argument("--max-train-samples",type=int)
    parser.add_argument("--max-steps",type=int,help="Explicit short diagnostic run")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cpu")


def add_reynolds_options(parser):
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--reynolds",type=float,help="Actual Reynolds number, NOT in millions")
    group.add_argument("--reynolds-file",type=Path,help="Explicit actual-Reynolds (N,) NPY in source row order")


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands=parser.add_subparsers(dest="command",required=True)
    inspect_parser=commands.add_parser("inspect",help="Read native metadata, no training")
    inspect_parser.add_argument("--wing-root",type=Path,default=WING_ROOT)
    inspect_parser.add_argument("--foil-root",type=Path,default=FOIL_ROOT)
    inspect_parser.add_argument("--foil-subset",choices=("VT-BUF","VT-DD","VT-GU"),default="VT-GU")
    p=commands.add_parser("pretrain-2d",help="Paper Appendix A1 on original SuperFoil VT")
    p.add_argument("--foil-root",type=Path,default=FOIL_ROOT,
                   help="Original VT data root (default: D:/Superfoil)")
    p.add_argument("--foil-subset",choices=("VT-BUF","VT-DD","VT-GU"),required=True)
    p.add_argument("--condition-dim",type=int,choices=(2,3),default=3,
                   help="3 follows the paper (default Re=2e7); 2 explicitly omits Reynolds")
    add_reynolds_options(p);add_training_options(p,300,16)
    p=commands.add_parser("train-sld",help="Paper's data-driven SLD option; train labels only")
    p.add_argument("--wing-root",type=Path,default=WING_ROOT)
    add_training_options(p,900,16)
    p=commands.add_parser("train-3d",help="Freeze 2-D/SLD models and train the 3-D residual")
    p.add_argument("--wing-root",type=Path,default=WING_ROOT)
    p.add_argument("--airfoil-checkpoint",type=Path,required=True)
    p.add_argument("--sld-checkpoint",type=Path,required=True)
    add_reynolds_options(p);add_training_options(p,900,2)
    p=commands.add_parser("predict",help="Export native SuperWing files for A/B")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--input-root",type=Path,default=CFD_ROOT)
    p.add_argument("--output-root",type=Path,default=NETWORK_ROOT)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--device",default="cpu")
    add_reynolds_options(p)
    p=commands.add_parser("predict-foilpara",help="Predict from foiloc/foilgeo/foilsld; external sectional CL required")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--input-root",type=Path,default=Path("D:/data/foilpara"))
    p.add_argument("--output-root",type=Path,default=NETWORK_ROOT)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--device",default="cpu")
    p=commands.add_parser("test-3d",help="Synthetic foilpara forward test only; zero training steps")
    p.add_argument("--device",default="cpu")
    p=commands.add_parser("smoke-test",help="Synthetic forward/backward only; no production export")
    p.add_argument("--device",default="cpu")
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=="inspect":
        wing=SuperWingData(args.wing_root,target="none")
        report={"wing_samples":len(wing),"wing_shapes":len(wing.configs),
                "wing_grid":list(wing.centers.shape[2:]),"read_only":not wing.centers.flags.writeable}
        if args.foil_root is not None:
            foil=SuperFoilData(args.foil_root,args.foil_subset,condition_dim=3)
            report.update(foil_samples=len(foil),foil_subset=args.foil_subset,foil_root=str(foil.root),
                          foil_grid=list(foil.data.shape[1:]),foil_reynolds_in_source=False,
                          foil_reynolds_input=foil.reynolds_info)
        print(json.dumps(report,indent=2));return 0
    device=selected_device(args.device)
    if args.command=="test-3d":
        from foilpara import forward_test
        print(json.dumps(forward_test(device),indent=2));return 0
    if args.command=="predict-foilpara":
        from foilpara import predict_foilpara
        model,spec,payload=load_checkpoint(args.checkpoint,"wing")
        result=predict_foilpara(model,spec,payload,args.input_root,args.output_root,
                               device,args.batch_size)
        print(f"Prediction data saved: {result}");return 0
    if args.command=="smoke-test":
        print(json.dumps(smoke_test(device),indent=2));return 0
    if args.command=="predict":
        model,spec,payload=load_checkpoint(args.checkpoint,"wing")
        result=predict_directory(model,spec,payload,args.input_root,args.output_root,device,
                                 args.batch_size,args.reynolds,args.reynolds_file)
        print(f"Prediction data saved: {result}");return 0
    seed_everything(args.seed)
    predecessor={}
    if args.command=="pretrain-2d":
        spec=ModelSpec(condition_dim=args.condition_dim)
        data=SuperFoilData(args.foil_root,args.foil_subset,spec.condition_dim,args.reynolds,args.reynolds_file)
        stage="airfoil";model=AirfoilPredictor(spec);source=data.root
    elif args.command=="train-sld":
        spec=ModelSpec();data=SuperWingData(args.wing_root,target="sld")
        stage="sld";model=SpanwiseLiftPredictor(spec);source=data.root
    else:
        model,spec,predecessor=build_wing(args.airfoil_checkpoint,args.sld_checkpoint)
        data=SuperWingData(args.wing_root,target="field",require_reynolds=spec.condition_dim==3,
                           reynolds=args.reynolds,reynolds_file=args.reynolds_file)
        stage="wing";source=data.root
    protect_destination(args.checkpoint,[source])
    require(not args.checkpoint.exists(),f"Checkpoint already exists: {args.checkpoint}")
    train,val=group_split(data.official_train(),data.groups,args.validation_fraction,args.seed,args.max_train_samples)
    if stage=="wing":
        validate_sld_split({"train_shape_ids":predecessor["sld_train_shape_ids"]},data.groups[val])
    info=fit_model(model,data,train,val,stage,device,args.epochs,args.batch_size,
                   args.learning_rate,args.max_steps)
    info["train_shape_ids"]=np.unique(data.groups[train]).tolist()
    info["validation_shape_ids"]=np.unique(data.groups[val]).tolist()
    info.update(predecessor)
    if stage=="airfoil":
        info["foil_subset"]=data.subset
        info["reynolds_input"]=data.reynolds_info
    info.update(source_root=str(Path(source).resolve()),validation_split="shape-held-out from official training IDs")
    result=save_checkpoint(args.checkpoint,stage,model,spec,info,[source])
    print(f"Trained checkpoint saved: {result}");return 0


def synthetic_geometry():
    lower=np.linspace(1,0,128);upper=np.linspace(0,1,129)[1:]
    x=np.r_[lower,upper]
    y=np.r_[-.06*np.sin(np.pi*lower)-.001*lower,
             .06*np.sin(np.pi*upper)+.001*upper]
    x=np.r_[x,x[0]];y=np.r_[y,y[0]]
    vertices=np.empty((3,129,257),dtype=np.float64)
    for j,z in enumerate(np.linspace(0,2,129)):
        chord=1-.2*z
        vertices[0,j]=x*chord+.2*z
        vertices[1,j]=y*chord
        vertices[2,j]=z
    centers=.25*(vertices[:,:-1,:-1]+vertices[:,1:,:-1]+
                 vertices[:,:-1,1:]+vertices[:,1:,1:])
    return centers,vertices


def smoke_test(device="cpu"):
    """One real optimizer step per stage, synthetic inputs; no accuracy claim."""
    seed_everything(7)
    if device=="cpu": torch.set_num_threads(min(4,torch.get_num_threads()))
    spec=ModelSpec()
    airfoil=AirfoilPredictor(spec).to(device)
    c=torch.tensor([[.7,.5,6.],[.72,.6,6.5]],device=device)
    airfoil.normalizer.fit(c)
    y=torch.sin(torch.linspace(0,2*math.pi,401,device=device))[None,None,:].repeat(2,1,1)*.05
    foil_target=torch.cat((y*10, y*20+2),dim=1)
    airfoil.geometry_normalizer.fit(y)
    airfoil.output_normalizer.fit(foil_target)
    optimizer=torch.optim.SGD(airfoil.parameters(),lr=1e-4)
    loss=nn.functional.mse_loss(normalize_target(airfoil,airfoil(y,c),"airfoil"),
                                normalize_target(airfoil,foil_target,"airfoil"))
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    lift=SpanwiseLiftPredictor(spec).to(device)
    parameters=torch.zeros(2,38,device=device)
    conditions=torch.tensor([[.78,3.],[.8,4.]],device=device)
    lift.normalizer.fit(torch.cat((conditions,parameters),dim=1))
    lift_target=torch.linspace(.3,.7,128,device=device)[None,:].expand(2,-1)
    lift.output_normalizer.fit(lift_target[:,None,:])
    optimizer=torch.optim.SGD(lift.parameters(),lr=1e-4)
    lift_loss=nn.functional.mse_loss(normalize_target(lift,lift(parameters,conditions),"sld"),
                                     normalize_target(lift,lift_target,"sld"))
    lift_loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    model=PhysicsEmbeddedWingPredictor(spec,airfoil,lift).to(device).train()
    centers,vertices=synthetic_geometry()
    features=geometry_features(centers,vertices)
    batch={key:(torch.from_numpy(value.copy())[None].to(device) if value.dtype==bool
                else tensor(value)[None].to(device)) for key,value in features.items()}
    batch.update(parameters=parameters[:1],conditions=conditions[:1],
                 reynolds=torch.tensor([2e7],device=device))
    prior,_=model.prior(batch)
    model.residual.normalizer.fit(batch["conditions"])
    model.residual.input_normalizer.fit(torch.cat((batch["geometry"],prior),dim=1))
    wing_target=batch["geometry"]*torch.tensor([1.,.1,.01],device=device)[None,:,None,None]
    model.residual.output_normalizer.fit(wing_target)
    frozen={name:value.detach().clone() for name,value in model.airfoil.state_dict().items()}
    optimizer=torch.optim.SGD(model.residual.parameters(),lr=1e-4)
    result,prior,sld=model(batch,return_details=True)
    wing_loss=nn.functional.mse_loss(normalize_target(model,result,"wing"),
                                     normalize_target(model,wing_target,"wing"))
    require(bool(torch.isfinite(wing_loss)),"Non-finite smoke loss")
    wing_loss.backward();optimizer.step()
    require(all(p.grad is None for p in model.airfoil.parameters()),"Frozen 2-D model received gradients")
    require(all(torch.equal(value,model.airfoil.state_dict()[name]) for name,value in frozen.items()),
            "Frozen 2-D model changed during wing training")
    return {"synthetic_only":True,"device":device,"normalization":NORMALIZATION,
            "airfoil_shape":[2,2,401],
            "sld_shape":list(sld.shape),"prior_shape":list(prior.shape),
            "wing_shape":list(result.shape),"finite":bool(torch.isfinite(result).all()),
            "frozen_airfoil_unchanged":True,"optimizer_steps_per_stage":1}


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as error:
        print(f"error: {error}",file=sys.stderr)
        raise SystemExit(2)
