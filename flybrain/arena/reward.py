# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Engineered dopamine-inspired actor/critic signal, not a biological dopamine center."""
import math
import numpy as np


class RewardSystem:
    def __init__(self,feature_count,gamma=.98):
        self.gamma=gamma
        self.value_weights=np.zeros(feature_count)
        self.value_trace=np.zeros(feature_count)
        self.expected_value=0.
        self.next_value=0.
        self.task_reward=0.
        self.prediction_error=0.
        self.phasic=0.
        self.tonic=0.
        self.appetitive=0.
        self.aversive=0.
        self.components={'pollen':0.,'honey':0.,'purchases':0.,'full_wait':0.,'progress':0.,'empty_dips':0.,'blocked':0.}

    def evaluate(self,previous,next_features,pollen,empty_dips=0,blocked_fraction=0,terminal=False,learning=True,honey=0,
                 purchases=0,full_wait=0,progress=0):
        if not all(math.isfinite(float(x)) for x in (pollen,empty_dips,blocked_fraction,honey,purchases,full_wait,progress)):
            raise ValueError('Reward events must be finite')
        if pollen<0 or empty_dips<0 or honey<0 or purchases<0 or not 0<=blocked_fraction<=1 or not 0<=full_wait<=1:
            raise ValueError('Expected nonnegative pollen/dips and blockage in [0,1]')
        self.components={'pollen':min(pollen/100,5),'honey':2*min(honey/100,5),'purchases':purchases,
                         'full_wait':-.01*full_wait,'progress':float(np.clip(progress,-1,1)),
                         'empty_dips':-.025*empty_dips,'blocked':-.05*blocked_fraction}
        self.task_reward=sum(self.components.values())
        self.expected_value=float(self.value_weights@previous)
        self.next_value=0. if terminal else float(self.value_weights@next_features)
        self.prediction_error=float(np.clip(self.task_reward+self.gamma*self.next_value-self.expected_value,-10,10))
        self.appetitive=max(0,self.prediction_error)
        self.aversive=max(0,-self.prediction_error)
        self.phasic=math.tanh(self.prediction_error)
        # Slow recent-success trace; controls a modest, bounded excitability change.
        self.tonic=.975*self.tonic+.025*math.tanh(self.task_reward)
        self.value_trace=self.gamma*.9*self.value_trace+previous
        if learning:
            norm=max(1,float(self.value_trace@self.value_trace))
            self.value_weights+=.05*self.prediction_error*self.value_trace/norm
            np.clip(self.value_weights,-20,20,out=self.value_weights)
        if terminal:
            self.value_trace.fill(0)
        return self.prediction_error

    def reset_transient(self):
        self.value_trace.fill(0)
        self.expected_value=self.next_value=self.task_reward=self.prediction_error=0.
        self.phasic=self.tonic=self.appetitive=self.aversive=0.
        self.components={'pollen':0.,'honey':0.,'purchases':0.,'full_wait':0.,'progress':0.,'empty_dips':0.,'blocked':0.}

    def snapshot(self):
        return {'objective':'collect, convert, and purchase rewards with honey','expected_value':self.expected_value,
                'next_value':self.next_value,'task_reward':self.task_reward,'prediction_error':self.prediction_error,
                'phasic_signal':self.phasic,'tonic_signal':self.tonic,
                'appetitive_channel':self.appetitive,'aversive_channel':self.aversive,
                'reward_components':dict(self.components),'model':'engineered TD actor-critic neuromodulation proxy'}
