# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Honey-only toy shop. Rewards occur only after a successful paid transaction."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    name: str
    base_price: int
    utility: float
    stock: int | None
    growth_percent: int = 100


CATALOG=(Item('Bee sticker',25,2.,None,165),
         Item('Royal snack',80,6.,3),
         Item('Field charm',200,12.,1))


class HoneyShop:
    def __init__(self):
        self.prices=[item.base_price for item in CATALOG]
        self.stock=[item.stock for item in CATALOG]
        self.owned=[0]*len(CATALOG)
        self.spent=0
        self.total_utility=0.

    @property
    def purchases(self):
        return sum(self.owned)

    def buy(self,slot,wallet):
        if not isinstance(slot,int) or not 1<=slot<=len(CATALOG):
            return wallet,0.,'invalid_item'
        i=slot-1
        if self.stock[i] is not None and self.stock[i]<=0:
            return wallet,0.,'sold_out'
        price=self.prices[i]
        if wallet<price:
            return wallet,0.,'insufficient_honey'
        item=CATALOG[i]
        # Unlimited collectibles have diminishing novelty, as well as rising prices.
        utility=item.utility/(1+.3*self.owned[i]) if item.stock is None else item.utility
        self.owned[i]+=1
        if self.stock[i] is not None:
            self.stock[i]-=1
        self.prices[i]=(price*item.growth_percent+99)//100
        self.spent+=price
        self.total_utility+=utility
        return wallet-price,utility,'purchased'

    def snapshot(self,wallet):
        return {'purchases':self.purchases,'honey_spent':self.spent,'purchase_utility':self.total_utility,
                'items':[{'slot':i+1,'name':item.name,'price':self.prices[i],'stock':self.stock[i],
                          'owned':self.owned[i],'affordable':wallet>=self.prices[i] and self.stock[i]!=0}
                         for i,item in enumerate(CATALOG)]}
