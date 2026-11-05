# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Small supervised image segmenter for this synthetic arena, implemented in NumPy.

Class IDs: 0 background, 1 visible flower geometry, 2 hive/conversion station.
Training labels are simulator-only; runtime inference uses RGB and local patches.
"""
from pathlib import Path
import json
import numpy as np

MODEL_PATH=Path(__file__).resolve().parents[2]/'models/perception-v3.npz'


CONTEXT_SCALES=(5,11,21)


def box_mean(values,size):
    """Mean over a size x size window (edge padded), via an integral image."""
    pad=size//2
    height,width=values.shape[:2]
    padded=np.pad(values.astype(np.float64),((pad,pad),(pad,pad))+((0,0),)*(values.ndim-2),mode='edge')
    integral=np.zeros((height+2*pad+1,width+2*pad+1)+values.shape[2:])
    integral[1:,1:]=padded.cumsum(0).cumsum(1)
    window=(integral[size:size+height,size:size+width]-integral[:height,size:size+width]
            -integral[size:size+height,:width]+integral[:height,:width])
    return (window/(size*size)).astype(np.float32)


def pixel_features(image,resolution=(200,320),context=False):
    image=np.asarray(image)
    # Infer at 320 x 200 by default; the camera itself remains 640 x 400. Small targets can use a finer grid.
    height,width=image.shape[:2]
    ys=np.linspace(0,height-1,min(resolution[0],height)).astype(int)
    xs=np.linspace(0,width-1,min(resolution[1],width)).astype(int)
    rgb=image[ys[:,None],xs[None,:]].astype(np.float32)/255
    h,w=rgb.shape[:2]
    pad=np.pad(rgb,((2,2),(2,2),(0,0)),mode='edge')
    patches=[pad[2+dy:2+dy+h,2+dx:2+dx+w] for dy in (-2,0,2) for dx in (-2,0,2)]
    local=sum(patches)/9
    lum=rgb@np.array([.2126,.7152,.0722],dtype=np.float32)
    gy,gx=np.gradient(lum)
    # Local texture/context, not world coordinates or object IDs.
    features=np.concatenate(patches+[local,np.abs(rgb-local),gx[...,None],gy[...,None]],axis=2)
    if context:
        # Wider windows let the model see whole blobs: surrounding color, deviation from it, and local texture.
        extra=[]
        for size in CONTEXT_SCALES:
            mean=box_mean(rgb,size)
            variance=np.maximum(box_mean(lum[...,None]**2,size)-box_mean(lum[...,None],size)**2,0)
            extra+=[mean,np.abs(rgb-mean),np.sqrt(variance)]
        features=np.concatenate([features]+extra,axis=2)
    return features.reshape(-1,features.shape[-1]),(h,w),ys,xs


class PerceptionModel:
    def __init__(self,seed=71,classes=3,hidden=32,context=False):
        rng=np.random.default_rng(seed)
        self.classes=classes
        self.context=context
        self.resolution=(200,320)
        self.mean=None
        self.scale=None
        self.w1=rng.normal(0,.12,(35+7*len(CONTEXT_SCALES)*(1 if context else 0),hidden)).astype(np.float32)
        self.b1=np.zeros(hidden,dtype=np.float32)
        self.w2=rng.normal(0,.1,(hidden,classes)).astype(np.float32)
        self.b2=np.zeros(classes,dtype=np.float32)
        self.thresholds=np.array([0]+[.7]*(classes-1),dtype=np.float32)
        self.metadata={'version':'synthetic-flower-hive-mlp-v3','classes':['background','flower','hive'],
                       'runtime_inputs':'RGB patches only','status':'untrained'}

    def probabilities(self,features):
        if self.mean is None:
            raise ValueError('Perception model must be trained before inference')
        x=np.clip((features-self.mean)/self.scale,-8,8)
        hidden=np.maximum(0,x@self.w1+self.b1)
        logits=hidden@self.w2+self.b2
        logits-=logits.max(axis=1,keepdims=True)
        probabilities=np.exp(logits)
        return probabilities/probabilities.sum(axis=1,keepdims=True)

    def fit(self,features,labels,steps=500,seed=71,hard=None):
        """`hard` (optional bool per sample) marks confusable background (tokens, fakes): half of each
        batch's background is drawn from it, so rare lookalikes are not drowned out by grass."""
        rng=np.random.default_rng(seed)
        if self.mean is None:
            self.mean=features.mean(axis=0)
            self.scale=np.maximum(features.std(axis=0),.05)
        normalized=np.clip((features-self.mean)/self.scale,-8,8)
        parameters=[self.w1,self.b1,self.w2,self.b2]
        moments=[np.zeros_like(p) for p in parameters]
        variances=[np.zeros_like(p) for p in parameters]
        class_indices=[np.flatnonzero(labels==c) for c in range(self.classes)]
        hard_indices=np.flatnonzero((labels==0)&hard) if hard is not None else np.array([],dtype=int)
        easy_indices=np.flatnonzero((labels==0)&~hard) if hard is not None else class_indices[0]
        if any(len(indices)==0 for indices in class_indices):
            raise ValueError('Training data must contain every class')
        for step in range(1,steps+1):
            groups=[easy_indices,hard_indices]+class_indices[1:] if len(hard_indices) else class_indices
            sizes=[128 if len(hard_indices) and i<2 else 256 for i in range(len(groups))]
            indices=np.concatenate([rng.choice(c,n,replace=len(c)<n) for c,n in zip(groups,sizes)])
            rng.shuffle(indices)
            x=normalized[indices]
            y=labels[indices]
            hidden=np.maximum(0,x@self.w1+self.b1)
            logits=hidden@self.w2+self.b2
            logits-=logits.max(axis=1,keepdims=True)
            probability=np.exp(logits)
            probability/=probability.sum(axis=1,keepdims=True)
            loss=-np.log(np.maximum(probability[np.arange(len(y)),y],1e-9)).mean()
            probability[np.arange(len(y)),y]-=1
            probability/=len(y)
            dh=(probability@self.w2.T)*(hidden>0)
            gradients=[x.T@dh,dh.sum(axis=0),hidden.T@probability,probability.sum(axis=0)]
            for p,g,m,v in zip(parameters,gradients,moments,variances):
                m*=.9
                m+=.1*g
                v*=.999
                v+=.001*g*g
                p-=.003*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            if step%100==0:
                print(f'Perception step {step}/{steps}, balanced training loss {loss:.4f}',flush=True)

    def segment(self,image):
        features,shape,_,_=pixel_features(image,self.resolution,self.context)
        probability=self.probabilities(features).reshape(*shape,self.classes)
        masks=np.zeros(shape,dtype=np.uint8)
        for c in range(1,self.classes):
            masks[(probability[:,:,c]>=self.thresholds[c])&(probability[:,:,c]==probability.max(axis=2))]=c
        return probability,masks

    def save(self,path=MODEL_PATH):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,w1=self.w1,b1=self.b1,w2=self.w2,b2=self.b2,mean=self.mean,scale=self.scale,
                            thresholds=self.thresholds,metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls,path=MODEL_PATH):
        result=cls()
        with np.load(path,allow_pickle=False) as saved:
            for name in ('w1','b1','w2','b2','mean','scale','thresholds'):
                setattr(result,name,saved[name].copy())
            result.metadata=json.loads(str(saved['metadata']))
            result.classes=result.w2.shape[1]
            result.resolution=tuple(result.metadata.get('resolution',(200,320)))
            result.context=result.w1.shape[0]!=35
        return result


def metrics(truth,predicted):
    result={}
    for c,name in ((1,'flower'),(2,'hive')):
        tp=int(np.count_nonzero((truth==c)&(predicted==c)))
        fp=int(np.count_nonzero((truth!=c)&(predicted==c)))
        fn=int(np.count_nonzero((truth==c)&(predicted!=c)))
        result[name]={'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'iou':tp/max(1,tp+fp+fn),
                      'true_positive':tp,'false_positive':fp,'false_negative':fn}
    return result
