# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
import numpy as np
from flybrain.arena import ArenaWorld,ArenaConfig,Action
from flybrain.arena.economy import HoneyShop
from flybrain.arena.reward import RewardSystem


class EconomyTests(unittest.TestCase):
    def test_paid_purchase_and_escalating_price(self):
        shop=HoneyShop()
        wallet,reward,status=shop.buy(1,100)
        self.assertEqual((wallet,status,shop.owned[0],shop.prices[0]),(75,'purchased',1,42))
        self.assertEqual(reward,2)
        wallet,second,_=shop.buy(1,wallet)
        self.assertEqual(wallet,33)
        self.assertEqual(shop.prices[0],70)
        self.assertLess(second,reward)
        self.assertEqual(shop.spent,67)

    def test_limited_stock_cannot_be_bought_again(self):
        shop=HoneyShop()
        wallet=1000
        for _ in range(3):
            wallet,reward,status=shop.buy(2,wallet)
            self.assertEqual(status,'purchased')
        before=wallet
        wallet,reward,status=shop.buy(2,wallet)
        self.assertEqual((wallet,reward,status),(before,0,'sold_out'))
        self.assertEqual(shop.stock[1],0)

    def test_failed_purchase_does_not_change_wallet_stock_or_reward(self):
        shop=HoneyShop()
        before=shop.snapshot(24)
        self.assertEqual(shop.buy(1,24),(24,0,'insufficient_honey'))
        self.assertEqual(shop.snapshot(24),before)
        self.assertEqual(shop.buy(99,24),(24,0,'invalid_item'))

    def test_shop_requires_hive_and_held_key_buys_only_once(self):
        w=ArenaWorld()
        w.honey=100
        w.step(Action(buy=1))
        self.assertEqual(w.honey,100)
        self.assertEqual(w.shop.purchases,0)
        w.x,w.y=w.config.hive_x,w.config.hive_y
        w.step()
        for _ in range(60):
            w.step(Action(buy=1))
        self.assertEqual((w.honey,w.shop.purchases),(75,1))
        self.assertEqual(w.last_purchase_utility,0)

    def test_episode_preservation_keeps_wallet_and_scarcity(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.honey=100
        w.step(Action(buy=1))
        w.reset(preserve_economy=True)
        self.assertEqual((w.honey,w.shop.purchases,w.shop.prices[0]),(75,1,42))
        w.reset()
        self.assertEqual((w.honey,w.shop.purchases,w.shop.prices[0]),(0,0,25))

    def test_reward_bonus_only_uses_successful_purchase_utility(self):
        reward=RewardSystem(1)
        delta=reward.evaluate(np.ones(1),np.ones(1),0,purchases=2,learning=False)
        self.assertEqual(delta,2)
        self.assertGreater(reward.phasic,.9)
        reward.evaluate(np.ones(1),np.ones(1),0,learning=False)
        self.assertEqual(reward.components['purchases'],0)

    def test_conversion_and_purchase_conserve_earned_honey(self):
        w=ArenaWorld(ArenaConfig(spawn_x=-9,spawn_y=-17))
        w.pollen=60
        w.step(Action(convert=True))
        for _ in range(120):
            w.step()
        self.assertEqual((w.pollen,w.honey,w.honey_earned),(0,60,60))
        _,_,_,info=w.step(Action(buy=1))
        self.assertEqual(info['purchase_utility'],2)
        self.assertEqual(w.honey+w.shop.spent,w.honey_earned)
        self.assertEqual((w.honey,w.shop.spent,w.shop.purchases),(35,25,1))
        _,_,_,info=w.step()
        self.assertEqual(info['purchase_utility'],0)


if __name__=='__main__':
    unittest.main()
