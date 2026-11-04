# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Experimental camera -> connectome LIF reservoir -> reward-trained action readout.

Connectivity is anatomical. Encoding, dynamics, weight normalization, and the action
readout are engineered hypotheses, not a reconstruction of fly visual/motor physiology.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from flybrain.cli import source_fingerprint, write_json
from .world import Action
from .reward import RewardSystem
from .vision import ThirdPersonVision

ROOT = Path(__file__).resolve().parents[2]
MOVES = np.array([(0,1),(-1,1),(1,1),(-1,0),(1,0),(-1,-1),(1,-1),(0,-1),(0,0)], dtype=float)
MOVE_NAMES = ['forward','forward-left','forward-right','left','right','back-left','back-right','backward','idle']


@dataclass
class ConnectomeGraph:
    root_ids: list
    groups: list
    pre: np.ndarray
    post: np.ndarray
    weights: np.ndarray
    sensory: np.ndarray
    motor: np.ndarray
    source_identity: str
    anatomical_synapses: int
    sides: list | None = None


def load_graph(database=ROOT/'data/connectome.sqlite', selection=ROOT/'selections/experiment-002.json',
               output=ROOT/'selections/arena-vision-002.json', optic_count=2048, projection_count=256):
    """Add strongest incoming visual projection paths, then strongest optic inputs.

    Record a new selection; never change the original benchmark or raw sources.
    """
    base=json.loads(Path(selection).read_text(encoding='utf-8'))
    conn=sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro',uri=True)
    try:
        manifest=json.loads(conn.execute("SELECT value FROM metadata WHERE key='source_manifest'").fetchone()[0])
        identity=source_fingerprint(manifest)
        if base.get('schema_version')!=2 or base['source_manifest_sha256']!=identity:
            raise ValueError('A portable schema-v2 selection matching the database sources is required')
        annotations={int(i):json.loads(a) for i,a in conn.execute('SELECT root_id,annotations FROM neurons')}
        ids={int(i) for i in base['root_ids']}
        if not ids.issubset(annotations):
            raise ValueError('Base selection contains unknown neurons')
        if Path(output).exists():
            saved=json.loads(Path(output).read_text(encoding='utf-8'))
            if saved.get('source_manifest_sha256')!=identity or saved.get('base_selection_sha256')!=hashlib.sha256(Path(selection).read_bytes()).hexdigest():
                raise ValueError('Existing arena selection has incompatible sources or base selection')
            ids={int(i) for i in saved['root_ids']}
            if not ids.issubset(annotations):
                raise ValueError('Arena selection contains unknown neurons')
        else:
            conn.execute('CREATE TEMP TABLE selected(id INTEGER PRIMARY KEY)')
            conn.executemany('INSERT INTO selected VALUES (?)',[(i,) for i in ids])
            for classes,budget in [({'visual_projection','visual_centrifugal'},projection_count),({'optic'},optic_count)]:
                conn.execute('DROP TABLE IF EXISTS temp.candidates')
                conn.execute('CREATE TEMP TABLE candidates(id INTEGER PRIMARY KEY)')
                conn.executemany('INSERT INTO candidates VALUES (?)',[(i,) for i,a in annotations.items()
                    if a.get('super_class') in classes and i not in ids])
                # CROSS JOIN fixes the small selected set as the outer loop for indexed incoming lookups.
                rows=conn.execute('''SELECT e.pre,SUM(e.syn_count) AS weight FROM selected s
                    CROSS JOIN edges e INDEXED BY edges_post ON e.post=s.id
                    JOIN candidates c ON c.id=e.pre GROUP BY e.pre ORDER BY weight DESC,e.pre LIMIT ?''',(budget,)).fetchall()
                added=[i for i,_ in rows]
                ids.update(added)
                conn.executemany('INSERT OR IGNORE INTO selected VALUES (?)',[(i,) for i in added])
                print(f'Added {len(added)} annotated {" / ".join(sorted(classes))} neurons',flush=True)
            result={'schema_version':2,'source_manifest_sha256':identity,
                    'base_selection_sha256':hashlib.sha256(Path(selection).read_bytes()).hexdigest(),
                    'method':'base benchmark + strongest incoming visual-projection neighbors + strongest incoming optic neighbors',
                    'biological_status':'experimental engineered sensorimotor substrate; no validated retinotopic mapping',
                    'root_ids':[str(i) for i in sorted(ids)]}
            write_json(output,result)
        sorted_ids=sorted(ids)
        index={root:i for i,root in enumerate(sorted_ids)}
        conn.execute('DROP TABLE IF EXISTS temp.selected')
        conn.execute('CREATE TEMP TABLE selected(id INTEGER PRIMARY KEY)')
        conn.executemany('INSERT INTO selected VALUES (?)',[(i,) for i in sorted_ids])
        rows=conn.execute('''SELECT e.pre,e.post,SUM(e.syn_count) FROM selected a
            CROSS JOIN edges e INDEXED BY edges_pre ON e.pre=a.id JOIN selected b ON b.id=e.post
            GROUP BY e.pre,e.post ORDER BY e.pre,e.post''').fetchall()
        groups=[annotations[i].get('super_class','unknown') for i in sorted_ids]
        sides=[annotations[i].get('side','unknown') for i in sorted_ids]
    finally:
        conn.close()
    pre=np.array([index[a] for a,b,w in rows],dtype=np.int32)
    post=np.array([index[b] for a,b,w in rows],dtype=np.int32)
    weights=np.array([w for a,b,w in rows],dtype=float)
    total=int(weights.sum())
    incoming=np.bincount(post,weights=weights,minlength=len(sorted_ids))
    weights/=np.maximum(1,incoming[post])
    sensory=np.array([i for i,g in enumerate(groups) if g=='optic'],dtype=np.int32)
    motor=np.array([i for i,g in enumerate(groups) if g in ('descending','motor')],dtype=np.int32)
    if not len(sensory) or not len(motor) or not len(rows):
        raise ValueError('Graph requires optic input neurons, motor/descending readout neurons, and connections')
    return ConnectomeGraph([str(i) for i in sorted_ids],groups,pre,post,weights,sensory,motor,identity,total,sides)


class NeuralController:
    neural_dt=.005
    decision_frames=15  # 4 Hz control; 50 LIF steps per 0.25 s camera/action interval.
    motor_features=32
    status_features=13
    feature_count=151  # motor32 + status13 + projection16 + optic12 + status-pair interactions78

    def __init__(self,graph,seed=42,learning=True,perception=None,sample_actions=True):
        self.graph=graph
        self.seed=seed
        self.learning=learning
        self.sample_actions=sample_actions
        self.rng=np.random.default_rng(seed)
        n=len(graph.root_ids)
        self.v=np.zeros(n)
        self.rates=np.zeros(n)
        self.spikes=np.zeros(n,dtype=bool)
        self.last_counts=np.zeros(n,dtype=np.int32)
        self.refractory=np.zeros(n,dtype=np.int8)
        self.adaptation=np.zeros(n)
        self.pool=self.rng.integers(0,self.motor_features,size=len(graph.motor))
        self.pool_sizes=np.maximum(1,np.bincount(self.pool,minlength=self.motor_features))
        self.threshold=self.rng.uniform(.85,1.15,size=n)
        self.vision=ThirdPersonVision(graph.root_ids,graph.sensory,graph.sides,perception=perception)
        self.proprio=np.array([i for i,g in enumerate(graph.groups) if g=='sensory_ascending'],dtype=int)
        self.proprio_groups=np.arange(len(self.proprio))%self.status_features
        self.proprio_sizes=np.maximum(1,np.bincount(self.proprio_groups,minlength=self.status_features))
        self.projection=np.array([i for i,g in enumerate(graph.groups) if g in ('visual_projection','visual_centrifugal')],dtype=int)
        self.projection_groups=np.arange(len(self.projection))%16
        self.projection_sizes=np.maximum(1,np.bincount(self.projection_groups,minlength=16))
        self.optic_groups=np.zeros(len(graph.sensory),dtype=int)
        for slot,row,column,profile in self.vision.mapping:
            self.optic_groups[slot]=min(2,row*3//self.vision.rows)*4+min(3,column*4//self.vision.columns)
        self.optic_sizes=np.maximum(1,np.bincount(self.optic_groups,minlength=12))
        self.pair_indices=np.triu_indices(self.status_features,k=1)
        self.weights=self.rng.normal(0,.025,size=(16,self.feature_count+1))
        self.weights[:9,-1]=[.2,.1,.1,0,0,0,0,0,-.4]
        self.weights[9,-1]=.15
        self.weights[10,-1]=-2.8
        self.weights[11,-1]=-.4
        self.weights[12:,-1]=[1.,-1.,-1.,-1.]
        self.eligibility=np.zeros_like(self.weights)
        self.baseline=0.
        self.update_count=0
        self.decisions=0
        self.total_spikes=0
        self.last_action=Action()
        self.action_index=8
        self.last_delta=0.
        self.last_reward=0.
        self.last_features=np.zeros(self.feature_count+1)
        self.last_probs=np.ones(9)/9
        self.reward_system=RewardSystem(self.feature_count+1)
        self._pending_feedback=None

    @property
    def previous_image(self):
        return self.vision.previous_luminance

    def encode(self,image,camera_geometry=None):
        return self.vision.encode(image,camera_geometry)

    def act(self,image,seconds=.25,camera_geometry=None,inventory=None):
        if not np.isfinite(seconds) or seconds<=0:
            raise ValueError('seconds must be finite and positive')
        optic_drives=self.encode(image,camera_geometry)
        g=self.graph
        drive=np.full(len(self.v),.95)
        drive[g.sensory]=.55+3.8*optic_drives
        drive[g.motor]*=1+.10*self.reward_system.tonic
        if inventory is not None and len(self.proprio):
            supplied=np.asarray(inventory,dtype=float)
            if supplied.shape not in ((2,),(3,),(13,)) or not np.isfinite(supplied).all():
                raise ValueError('Expected normalized public inventory/status channels')
            if len(supplied)==13:
                status=supplied.copy()
                status[11]=float(self.vision.features[5:12,7:17,10].max())
            else:
                status=np.zeros(13)
                status[0]=supplied[0]
                status[1]=supplied[0]>=.999
                status[10]=supplied[1]
                status[2]=supplied[2] if len(supplied)==3 else 0
            drive[self.proprio]=.55+3.8*np.clip(status,0,1)[self.proprio_groups]
        counts=np.zeros(len(self.v),dtype=np.int32)
        steps=max(1,round(seconds/self.neural_dt))
        for _ in range(steps):
            recurrent=np.bincount(g.post,weights=g.weights*self.spikes[g.pre],minlength=len(self.v))
            current=drive+4*recurrent+self.rng.normal(0,.10,size=len(self.v))
            self.adaptation*=np.exp(-self.neural_dt/.250)
            self.refractory=np.maximum(0,self.refractory-1)
            self.v+=self.neural_dt/.020*(-self.v+current)
            self.v[self.refractory>0]=0
            self.spikes=(self.v>=self.threshold+self.adaptation)&(self.refractory==0)
            self.v[self.spikes]=0
            self.refractory[self.spikes]=2
            self.adaptation[self.spikes]+=.15
            counts+=self.spikes
            self.rates*=np.exp(-self.neural_dt/.100)
            self.rates+=self.spikes*(1-np.exp(-self.neural_dt/.100))/self.neural_dt
        self.last_counts=counts
        self.total_spikes+=int(counts.sum())
        motor=np.tanh(np.bincount(self.pool,weights=self.rates[g.motor],minlength=32)/self.pool_sizes/35)
        status=np.tanh(np.bincount(self.proprio_groups,weights=self.rates[self.proprio],minlength=13)/self.proprio_sizes/35)
        projection=np.tanh(np.bincount(self.projection_groups,weights=self.rates[self.projection],minlength=16)/self.projection_sizes/35)
        optic=np.tanh(np.bincount(self.optic_groups,weights=self.rates[g.sensory],minlength=12)/self.optic_sizes/35)
        features=np.concatenate((motor,status,projection,optic,status[self.pair_indices[0]]*status[self.pair_indices[1]],[1.]))
        if self._pending_feedback is not None:
            events=self._pending_feedback
            self._pending_feedback=None
            self.feedback(**events,next_features=features)
        logits=self.weights@features
        exp=np.exp(logits[:9]-np.max(logits[:9]))
        probs=exp/exp.sum()
        chosen=int(self.rng.choice(9,p=probs)) if self.sample_actions else int(np.argmax(probs))
        dip_prob=1/(1+np.exp(-np.clip(logits[9],-20,20)))
        jump_prob=1/(1+np.exp(-np.clip(logits[10],-20,20)))
        convert_prob=1/(1+np.exp(-np.clip(logits[11],-20,20)))
        dip=bool(self.rng.random()<dip_prob) if self.sample_actions else bool(dip_prob>=.5)
        jump=bool(self.rng.random()<jump_prob) if self.sample_actions else bool(jump_prob>=.5)
        convert=bool(self.rng.random()<convert_prob) if self.sample_actions else bool(convert_prob>=.5)
        purchase_exp=np.exp(logits[12:16]-logits[12:16].max())
        purchase_probs=purchase_exp/purchase_exp.sum()
        buy=int(self.rng.choice(4,p=purchase_probs)) if self.sample_actions else int(np.argmax(purchase_probs))
        gradient=np.zeros_like(self.weights)
        gradient[:9]=-probs[:,None]*features
        gradient[chosen]+=features
        gradient[9]=(int(dip)-dip_prob)*features
        gradient[10]=(int(jump)-jump_prob)*features
        gradient[11]=(int(convert)-convert_prob)*features
        gradient[12:16]=-purchase_probs[:,None]*features
        gradient[12+buy]+=features
        self.eligibility=.94*self.eligibility+gradient
        self.last_features=features
        self.last_probs=probs
        self.action_index=chosen
        self.last_action=Action(strafe=float(MOVES[chosen,0]),forward=float(MOVES[chosen,1]),jump=jump,dip=dip,convert=convert,buy=buy)
        self.decisions+=1
        return self.last_action

    def fit_demonstrations(self,features,actions,steps=600,seed=123):
        """Training-only behavior cloning of an explicitly labelled teacher dataset."""
        x=np.asarray(features,dtype=float)
        if x.shape!=(len(actions),self.feature_count+1) or not len(actions):
            raise ValueError('Demonstration features do not match this readout')
        move=np.array([next(i for i,(sx,fy) in enumerate(MOVES) if sx==a.strafe and fy==a.forward) for a in actions])
        binary=np.array([[a.dip,a.jump,a.convert] for a in actions],dtype=float)
        buy=np.array([a.buy for a in actions])
        binary_weight=np.minimum(12,(len(actions)-binary.sum(axis=0))/np.maximum(1,binary.sum(axis=0)))
        buy_weight=np.minimum(12,np.sqrt(len(actions)/np.maximum(1,np.bincount(buy,minlength=4))))
        rng=np.random.default_rng(seed)
        m=np.zeros_like(self.weights)
        v=np.zeros_like(self.weights)
        for step in range(1,steps+1):
            idx=rng.integers(0,len(x),size=min(256,len(x)))
            batch=x[idx]
            logits=batch@self.weights.T
            move_p=np.exp(logits[:,:9]-logits[:,:9].max(axis=1,keepdims=True))
            move_p/=move_p.sum(axis=1,keepdims=True)
            binary_p=1/(1+np.exp(-np.clip(logits[:,9:12],-30,30)))
            buy_p=np.exp(logits[:,12:16]-logits[:,12:16].max(axis=1,keepdims=True))
            buy_p/=buy_p.sum(axis=1,keepdims=True)
            grad=np.zeros_like(logits)
            grad[:,:9]=move_p
            grad[np.arange(len(idx)),move[idx]]-=1
            grad[:,9:12]=(binary_p-binary[idx])*np.where(binary[idx]>0,binary_weight,1)
            grad[:,12:16]=buy_p
            grad[np.arange(len(idx)),12+buy[idx]]-=1
            grad[:,12:16]*=buy_weight[buy[idx],None]
            gradient=grad.T@batch/len(idx)+.0001*self.weights
            m=.9*m+.1*gradient
            v=.999*v+.001*gradient*gradient
            self.weights-=.015*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        predicted=np.argmax(x@self.weights[:9].T,axis=1)
        return {'examples':len(actions),'steps':steps,'training_movement_accuracy':float(np.mean(predicted==move)),
                'source':'privileged training demonstrations; never a runtime controller'}

    def queue_feedback(self,reward,empty_dips=0,blocked_fraction=0,honey=0,purchases=0,full_wait=0,progress=0):
        if self.decisions:
            if self._pending_feedback is not None:
                raise ValueError('Process the queued transition before queuing another')
            self._pending_feedback=dict(reward=reward,empty_dips=empty_dips,blocked_fraction=blocked_fraction,
                                        honey=honey,purchases=purchases,full_wait=full_wait,progress=progress)

    def feedback(self,reward,empty_dips=0,blocked_fraction=0,next_features=None,terminal=False,honey=0,
                 purchases=0,full_wait=0,progress=0):
        if not np.isfinite(reward):
            raise ValueError('Reward must be finite')
        following=self.last_features if next_features is None else next_features
        delta=self.reward_system.evaluate(self.last_features,following,float(reward),empty_dips,blocked_fraction,
                                          terminal=terminal,learning=self.learning,honey=honey,purchases=purchases,
                                          full_wait=full_wait,progress=progress)
        self.baseline=self.reward_system.expected_value
        if self.learning and abs(delta)>1e-10:
            # Experimental reward-modulated policy gradient; connectivity remains fixed.
            self.weights+=.012*np.clip(delta,-2,2)*np.clip(self.eligibility,-4,4)
            np.clip(self.weights,-12,12,out=self.weights)
            self.update_count+=1
        self.last_reward=float(reward)
        self.last_delta=delta
        return delta

    def reset_dynamics(self):
        self.v.fill(0)
        self.rates.fill(0)
        self.spikes.fill(False)
        self.last_counts.fill(0)
        self.refractory.fill(0)
        self.adaptation.fill(0)
        self.eligibility.fill(0)
        self.vision.reset()
        self.last_action=Action()
        self.baseline=0
        self.last_delta=0
        self.last_reward=0
        self._pending_feedback=None
        self.reward_system.reset_transient()

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        metadata={'seed':self.seed,'source_identity':self.graph.source_identity,'root_ids':self.graph.root_ids,
                  'decisions':self.decisions,'updates':self.update_count,
                  'status':'experimental fixed-connectome LIF reservoir with reward-trained readout',
                  'encoding':self.vision.snapshot(),'optic_mapping':self.vision.mapping_record(),
                  'dynamics':'unsigned incoming-normalized recurrent weights, adaptive firing thresholds; global LIF parameters not fitted to biology',
                  'rng_state':self.rng.bit_generator.state}
        metadata['conversion_control']='E rising-edge toggle; automatic simulated-bee batches until empty'
        metadata['readout_architecture']='neural-regions-and-public-status-v5'
        metadata['reward_system']=self.reward_system.snapshot()
        np.savez_compressed(path,weights=self.weights,value_weights=self.reward_system.value_weights,metadata=json.dumps(metadata))

    def load(self,path,allow_migration=False):
        with np.load(path,allow_pickle=False) as stored:
            metadata=json.loads(str(stored['metadata']))
            if metadata['root_ids']!=self.graph.root_ids or metadata['source_identity']!=self.graph.source_identity:
                raise ValueError('Checkpoint graph differs from this controller')
            if metadata['seed']!=self.seed or metadata['encoding']['version']!=self.vision.version:
                raise ValueError('Checkpoint seed or visual encoder differs')
            if metadata['encoding'].get('perception_model')!=self.vision.snapshot()['perception_model']:
                raise ValueError('Checkpoint perception model differs')
            weights=stored['weights']
            value=stored['value_weights']
            migration=weights.shape!=self.weights.shape or value.shape!=self.reward_system.value_weights.shape
            if migration and not (allow_migration and weights.shape==(12,33) and value.shape==(33,)):
                raise ValueError('Checkpoint architecture differs')
            if not np.isfinite(weights).all() or not np.isfinite(value).all():
                raise ValueError('Nonfinite checkpoint weights')
            if migration:
                self.weights[:12,:32]=weights[:,:32]
                self.weights[:12,-1]=weights[:,-1]
                self.reward_system.value_weights[:32]=value[:32]
                self.reward_system.value_weights[-1]=value[-1]
            else:
                self.weights[:]=weights
                self.reward_system.value_weights[:]=value
            self.update_count=int(metadata['updates'])
            self.decisions=int(metadata['decisions'])

    def snapshot(self):
        return {'neurons':len(self.v),'connections':len(self.graph.pre),
                'active_neurons':int(np.count_nonzero(self.last_counts)),
                'spikes_last_interval':int(self.last_counts.sum()),'spikes_total':self.total_spikes,
                'mean_rate_hz':float(self.rates.mean()),'motor_mean_rate_hz':float(self.rates[self.graph.motor].mean()),
                'decisions':self.decisions,'readout_updates':self.update_count,'last_reward':self.last_reward,
                'prediction_error':self.last_delta,'action':MOVE_NAMES[self.action_index],
                'dip':self.last_action.dip,'jump':self.last_action.jump,'convert':self.last_action.convert,
                'buy':self.last_action.buy,'readout_architecture':'neural-regions-and-public-status-v5',
                'learning_enabled':self.learning,'stochastic_actions':self.sample_actions,
                'reward_system':self.reward_system.snapshot(),'vision':self.vision.snapshot()}
