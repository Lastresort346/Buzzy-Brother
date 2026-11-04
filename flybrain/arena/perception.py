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


def pixel_features(image):
    image=np.asarray(image)
    # Infer at 320 x 200; the camera itself remains 640 x 400.
    height,width=image.shape[:2]
    ys=np.linspace(0,height-1,min(200,height)).astype(int)
    xs=np.linspace(0,width-1,min(320,width)).astype(int)
    rgb=image[ys[:,None],xs[None,:]].astype(np.float32)/255
    h,w=rgb.shape[:2]
    pad=np.pad(rgb,((2,2),(2,2),(0,0)),mode='edge')
    patches=[pad[2+dy:2+dy+h,2+dx:2+dx+w] for dy in (-2,0,2) for dx in (-2,0,2)]
    local=sum(patches)/9
    lum=rgb@np.array([.2126,.7152,.0722],dtype=np.float32)
    gy,gx=np.gradient(lum)
    # Local texture/context, not world coordinates or object IDs.
    features=np.concatenate(patches+[local,np.abs(rgb-local),gx[...,None],gy[...,None]],axis=2)
    return features.reshape(-1,features.shape[-1]),(h,w),ys,xs


class PerceptionModel:
    def __init__(self,seed=71):
        rng=np.random.default_rng(seed)
        self.mean=None
        self.scale=None
        self.w1=rng.normal(0,.12,(35,32)).astype(np.float32)
        self.b1=np.zeros(32,dtype=np.float32)
        self.w2=rng.normal(0,.1,(32,3)).astype(np.float32)
        self.b2=np.zeros(3,dtype=np.float32)
        self.thresholds=np.array([0,.7,.7],dtype=np.float32)
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

    def fit(self,features,labels,steps=500,seed=71):
        rng=np.random.default_rng(seed)
        if self.mean is None:
            self.mean=features.mean(axis=0)
            self.scale=np.maximum(features.std(axis=0),.05)
        normalized=np.clip((features-self.mean)/self.scale,-8,8)
        parameters=[self.w1,self.b1,self.w2,self.b2]
        moments=[np.zeros_like(p) for p in parameters]
        variances=[np.zeros_like(p) for p in parameters]
        class_indices=[np.flatnonzero(labels==c) for c in range(3)]
        if any(len(indices)==0 for indices in class_indices):
            raise ValueError('Training data must contain all three classes')
        for step in range(1,steps+1):
            indices=np.concatenate([rng.choice(c,256,replace=len(c)<256) for c in class_indices])
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
        features,shape,_,_=pixel_features(image)
        probability=self.probabilities(features).reshape(*shape,3)
        masks=np.zeros(shape,dtype=np.uint8)
        for c in (1,2):
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
