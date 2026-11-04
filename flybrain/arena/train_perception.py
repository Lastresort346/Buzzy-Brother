# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Render train/validation/test splits, fit RGB recognition, report held-out metrics."""
import argparse
import json
from pathlib import Path
import numpy as np

from .app import ArenaApp
from .perception import PerceptionModel,pixel_features,metrics,MODEL_PATH


def render_split(app,seed,count):
    rng=np.random.default_rng(seed)
    app.reset(seed=seed)
    frames=[]
    truth=[]
    for i in range(count):
        if i%3==0:
            cx,cy,_,_=app.world.patches[int(rng.integers(3))]
            x,y=cx+rng.uniform(-3,3),cy+rng.uniform(-5,3)
        elif i%3==1:
            x,y=app.world.config.hive_x+rng.uniform(-5,5),app.world.config.hive_y+rng.uniform(-3,5)
        else:
            x,y=rng.uniform(-20,20),rng.uniform(-20,20)
        app.world.x,app.world.y,app.world.z=float(x),float(y),0
        app.world.camera_yaw=float(rng.uniform(-18,18))
        app.world.camera_distance=float(rng.uniform(16,20))
        for flower in app.world.flowers:
            flower.pollen=float(rng.uniform(.1,1))
        app.render_state()
        frames.append(app.observe())
        truth.append(app.semantic_labels())
    return frames,truth


def evaluate(model,frames,labels):
    probabilities=[]
    truth=[]
    for image,label in zip(frames,labels):
        features,shape,ys,xs=pixel_features(image)
        probabilities.append(model.probabilities(features))
        truth.append(label[ys[:,None],xs[None,:]].ravel())
    return np.concatenate(probabilities),np.concatenate(truth)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--output',type=Path,default=MODEL_PATH)
    args=parser.parse_args()
    app=ArenaApp(offscreen=True,width=640,height=400)
    try:
        train_images,train_labels=render_split(app,101,30)
        val_images,val_labels=render_split(app,202,12)
        test_images,test_labels=render_split(app,303,15)
    finally:
        app.destroy()
    rng=np.random.default_rng(71)
    train_x=[]
    train_y=[]
    for image,label in zip(train_images,train_labels):
        features,shape,ys,xs=pixel_features(image)
        y=label[ys[:,None],xs[None,:]].ravel()
        for c in range(3):
            indices=np.flatnonzero(y==c)
            if len(indices):
                selected=rng.choice(indices,min(1000,len(indices)),replace=False)
                train_x.append(features[selected])
                train_y.append(y[selected])
    model=PerceptionModel()
    model.fit(np.concatenate(train_x),np.concatenate(train_y),steps=args.steps)
    validation,validation_truth=evaluate(model,val_images,val_labels)
    for c in (1,2):
        options=[]
        for threshold in np.linspace(.25,.995,100):
            predicted=np.argmax(validation,axis=1)==c
            predicted&=validation[:,c]>=threshold
            actual=validation_truth==c
            tp=np.count_nonzero(predicted&actual)
            precision=tp/max(1,np.count_nonzero(predicted))
            recall=tp/max(1,np.count_nonzero(actual))
            f1=2*precision*recall/max(1e-9,precision+recall)
            options.append((precision>=.8,f1,threshold))
        model.thresholds[c]=max(options)[2]
    probability,truth=evaluate(model,test_images,test_labels)
    prediction=np.zeros(len(truth),dtype=np.uint8)
    for c in (1,2):
        prediction[(np.argmax(probability,axis=1)==c)&(probability[:,c]>=model.thresholds[c])]=c
    report={'split_seeds':{'train':101,'validation':202,'test':303},
            'frames':{'train':30,'validation':12,'test':15},'camera':[640,400],
            'segmentation_resolution':[320,200],'steps':args.steps,
            'held_out':metrics(truth,prediction),'thresholds':model.thresholds.tolist(),
            'scope':'synthetic arena flower/hive pixel recognition; not real Roblox generalization',
            'ground_truth':'oracle render pass used only for supervised training/evaluation'}
    model.metadata.update(status='trained',evaluation=report)
    model.save(args.output)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/perception-v3.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
