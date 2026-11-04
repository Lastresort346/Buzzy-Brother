# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Single-camera spatial adapter. No synthetic stereo or inferred metric depth.

Receptive fields are engineered in screen coordinates, not biological retinotopy.
"""
import numpy as np


class ThirdPersonVision:
    version='semantic-third-person-v3'
    rows=16
    columns=24
    channel_names=('luminance','red_green','blue_yellow','edges','on','off','right','left','down','up','flower','hive','visibility')

    def __init__(self,root_ids,sensory,sides=None,avatar_mask=(.5,.535,.070,.105),perception=None):
        self.root_ids=list(root_ids)
        self.sensory=np.asarray(sensory,dtype=int)
        self.sides=list(sides) if sides is not None else ['unknown']*len(root_ids)
        if len(self.sides)!=len(root_ids):
            raise ValueError('Side annotations must match neuron IDs')
        self.avatar_mask=avatar_mask
        self.perception=perception
        self.mapping=[]
        for side in ('left','right','unknown'):
            slots=[j for j,i in enumerate(self.sensory) if (self.sides[i] if self.sides[i] in ('left','right') else 'unknown')==side]
            # Both hemispheres receive regions of ONE camera; center regions overlap.
            columns=range(0,16) if side=='left' else range(8,24) if side=='right' else range(24)
            cells=[(r,c) for r in range(self.rows) for c in columns]
            for rank,slot in enumerate(slots):
                row,column=cells[min(len(cells)-1,rank*len(cells)//max(1,len(slots))) ]
                self.mapping.append((slot,row,column,rank%4))
        self.mapping.sort()
        self.reset()

    def reset(self):
        self.previous_luminance=None
        self.previous_valid=None
        self.camera_geometry=None
        self.features=np.zeros((self.rows,self.columns,len(self.channel_names)))
        self.valid_fraction=np.ones((self.rows,self.columns))
        self.last_frame=None
        self.last_mask=None
        self.drives=np.zeros(len(self.sensory))
        self.motion_reset=True
        self.effective_mask=tuple(self.avatar_mask)
        self.semantic_mask=None

    def encode(self,image,camera_geometry=None):
        image=np.asarray(image)
        if image.ndim!=3 or image.shape[2]!=3 or image.shape[0]<self.rows or image.shape[1]<self.columns:
            raise ValueError('RGB frame is smaller than the spatial input grid')
        if not np.isfinite(image).all() or image.min()<0 or image.max()>255:
            raise ValueError('RGB values must be finite and in [0,255]')
        height,width=image.shape[:2]
        geometry=None if camera_geometry is None else tuple(float(x) for x in camera_geometry)
        if geometry is not None and (len(geometry)!=4 or not np.isfinite(geometry).all()
                                     or geometry[2]<=0 or not 1<geometry[3]<179):
            raise ValueError('Expected finite camera yaw, pitch, positive distance and horizontal FOV')
        rgb=image.astype(float)/255
        yy,xx=np.mgrid[:height,:width]
        cx,cy,rx,ry=self.avatar_mask
        if geometry is not None:
            angular_scale=18/geometry[2]*np.tan(np.deg2rad(68/2))/np.tan(np.deg2rad(geometry[3]/2))
            angular_scale=float(np.clip(angular_scale,.35,3))
            rx*=angular_scale
            ry*=angular_scale
        self.effective_mask=(cx,cy,rx,ry)
        valid=(((xx+.5)/width-cx)/rx)**2+(((yy+.5)/height-cy)/ry)**2>1
        coarse=np.zeros((self.rows,self.columns,3))
        quality=np.zeros((self.rows,self.columns))
        flower_grid=np.zeros((self.rows,self.columns))
        hive_grid=np.zeros_like(flower_grid)
        self.semantic_mask=np.zeros((height,width),dtype=np.uint8)
        flower_pixels=hive_pixels=np.zeros((height,width))
        if self.perception is not None:
            probability,semantic=self.perception.segment(image)
            yi=np.arange(height)*semantic.shape[0]//height
            xi=np.arange(width)*semantic.shape[1]//width
            self.semantic_mask=semantic[yi[:,None],xi[None,:]]
            self.semantic_mask[~valid]=0
            flower_pixels=probability[yi[:,None],xi[None,:],1]*(self.semantic_mask==1)
            hive_pixels=probability[yi[:,None],xi[None,:],2]*(self.semantic_mask==2)
        ys=np.linspace(0,height,self.rows+1).astype(int)
        xs=np.linspace(0,width,self.columns+1).astype(int)
        for row in range(self.rows):
            for column in range(self.columns):
                pixels=rgb[ys[row]:ys[row+1],xs[column]:xs[column+1]]
                mask=valid[ys[row]:ys[row+1],xs[column]:xs[column+1]]
                quality[row,column]=mask.mean()
                if mask.any():
                    coarse[row,column]=pixels[mask].mean(axis=0)
                    flower_grid[row,column]=flower_pixels[ys[row]:ys[row+1],xs[column]:xs[column+1]].max()
                    hive_grid[row,column]=hive_pixels[ys[row]:ys[row+1],xs[column]:xs[column+1]].max()
        luminance=coarse@np.array([.2126,.7152,.0722])
        gy,gx=np.gradient(luminance)
        reliable=quality>.85
        edge_valid=reliable.copy()
        edge_valid[1:]&=reliable[:-1]
        edge_valid[:-1]&=reliable[1:]
        edge_valid[:,1:]&=reliable[:,:-1]
        edge_valid[:,:-1]&=reliable[:,1:]
        edges=np.clip(2*np.hypot(gx,gy),0,1)*edge_valid
        reset=self.previous_luminance is None or geometry!=self.camera_geometry
        previous=luminance if reset else self.previous_luminance
        stable=reliable if reset else reliable & self.previous_valid
        difference=(luminance-previous)*stable
        on=np.maximum(difference,0)
        off=np.maximum(-difference,0)
        right=np.zeros_like(luminance)
        down=np.zeros_like(luminance)
        left=np.zeros_like(luminance)
        up=np.zeros_like(luminance)
        if not reset:
            for offset in (1,2):
                horizontal=(previous[:,:-offset]*luminance[:,offset:]-previous[:,offset:]*luminance[:,:-offset])*stable[:,offset:]*stable[:,:-offset]
                vertical=(previous[:-offset]*luminance[offset:]-previous[offset:]*luminance[:-offset])*stable[offset:]*stable[:-offset]
                right[:,offset:]+=np.maximum(horizontal,0)
                down[offset:]+=np.maximum(vertical,0)
                left[:,:-offset]+=np.maximum(-horizontal,0)
                up[:-offset]+=np.maximum(-vertical,0)
        right=np.maximum(right,0)
        down=np.maximum(down,0)
        red_green=np.clip((coarse[:,:,0]-coarse[:,:,1]+1)/2,0,1)
        blue_yellow=np.clip((coarse[:,:,2]-(coarse[:,:,0]+coarse[:,:,1])/2+1)/2,0,1)
        features=np.stack((luminance,red_green,blue_yellow,edges,on,off,
                           np.clip(3*right,0,1),np.clip(3*left,0,1),
                           np.clip(3*down,0,1),np.clip(3*up,0,1),flower_grid,hive_grid,quality),axis=-1)
        features[quality<.25]=0
        # Four fixed local response profiles, spatially mapped rather than randomly wired.
        profiles=np.stack((.45*luminance+.10*red_green+.10*blue_yellow+.35*flower_grid,
                           .25*edges+.20*on+.15*off+.40*hive_grid,
                           .65*features[:,:,6]+.20*features[:,:,8]+.15*luminance,
                           .65*features[:,:,7]+.20*features[:,:,9]+.15*luminance),axis=-1)
        profiles[quality<.25]=0
        drives=np.zeros(len(self.sensory))
        for slot,row,column,profile in self.mapping:
            drives[slot]=profiles[row,column,profile]
        self.previous_luminance=luminance.copy()
        self.previous_valid=reliable.copy()
        self.camera_geometry=geometry
        self.features=features
        self.valid_fraction=quality
        self.last_frame=image.astype(np.uint8).copy()
        self.last_mask=~valid
        self.drives=np.clip(drives,0,1)
        self.motion_reset=reset
        return self.drives

    def previews(self):
        if self.last_frame is None:
            return None
        frame=self.last_frame.copy()
        frame[self.last_mask]=(.25*frame[self.last_mask]+.75*np.array([235,175,75])).astype(np.uint8)
        height,width=frame.shape[:2]
        yi=(np.arange(height)*self.rows//height).astype(int)
        xi=(np.arange(width)*self.columns//width).astype(int)
        edges=self.features[:,:,3][yi[:,None],xi[None,:]]
        edge_rgb=np.repeat((edges*255).astype(np.uint8)[...,None],3,axis=2)
        if self.perception is not None:
            edge_rgb=(self.last_frame*.2).astype(np.uint8)
            edge_rgb[self.semantic_mask==1]=[90,235,140]
            edge_rgb[self.semantic_mask==2]=[250,185,65]
        flow=np.stack((self.features[:,:,6],np.maximum(self.features[:,:,8],self.features[:,:,9]),self.features[:,:,7]),axis=-1)
        flow_rgb=(np.clip(flow[yi[:,None],xi[None,:]],0,1)*255).astype(np.uint8)
        return frame,edge_rgb,flow_rgb

    def snapshot(self):
        return {'version':self.version,'camera_count':1,'stereo_depth':False,'grid':[self.columns,self.rows],
                'optic_input_neurons':len(self.sensory),'hemisphere_neurons':{
                    side:sum(self.sides[i]==side for i in self.sensory) for side in ('left','right')},
                'avatar_mask_fraction':0 if self.last_mask is None else float(self.last_mask.mean()),
                'motion_history_reset':self.motion_reset,'mean_edge_energy':float(self.features[:,:,3].mean()),
                'motion_energy':{name:float(self.features[:,:,j].mean()) for j,name in [(6,'right'),(7,'left'),(8,'down'),(9,'up')]},
                'mapping':'engineered screen regions with central overlap; not biological retinotopy',
                'avatar_mask':list(self.effective_mask),
                'perception_model':None if self.perception is None else self.perception.metadata['version'],
                'flower_regions':int(np.count_nonzero(self.features[:,:,10]>.5)),
                'hive_regions':int(np.count_nonzero(self.features[:,:,11]>.5))}

    def mapping_record(self):
        return [{'root_id':self.root_ids[self.sensory[slot]],'side':self.sides[self.sensory[slot]],
                 'row':row,'column':column,'profile':profile} for slot,row,column,profile in self.mapping]
