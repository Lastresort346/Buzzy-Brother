# Bee Swarm Simulator facts

Reference notes on the Roblox game *Bee Swarm Simulator* (BSS), the project's long-term
target. Summarized in our own words from the community wiki
(https://bee-swarm-simulator.fandom.com, content CC-BY-SA), read on 2026-11-03.
Wiki values can lag behind the live game. Lines marked **(unverified)** come from search
snippets only, not from a page we read.

## Pollen, flowers and conversion

- Pollen is gathered from flowers by the player's tool (click or hold) and by bees, then
  turned into honey, mainly at the player's hive. Honey is the main currency.
- Flower sizes and pollen: Source: [Pollen](https://bee-swarm-simulator.fandom.com/wiki/Pollen), [Flowers](https://bee-swarm-simulator.fandom.com/wiki/Flowers)

  | Size | Single | Double | Triple | Large | Star |
  |---|---|---|---|---|---|
  | Pollen multiplier per harvest | 1 | 2 | 3 | 4 | 5 |
  | Total pollen before regrowth | 30 | 45 | 60 | 75 | 90 |

- Only single, double and triple flowers spawn naturally; large and star come from
  pollination (Fuzzy Bee, certain beequips and passives).
- A depleted flower must regrow; larger flowers regrow faster. Sprinklers, clouds and
  bubbles also regenerate flowers. Exact regrowth timing was not on the pages read.
- Flower colors: white, red, blue. Red or blue bees get +50% pollen from flowers of their
  own color. White flowers are in every field except Mountain Top.
- When the bag is full the game says the container is full and to go make honey at the
  hive. Alternatives to the hive: Instant Converter (1 ticket), Micro-Converter, Coconut,
  Honeysuckle (partial), plus instant-conversion percentages from some gear.
- Conversion is done by bees at the hive. Example: a Basic Bee converts 80 honey every
  4 s (20 honey/s). Convert Rate bonuses from bags, badges and gear stack additively and
  have no cap. Source: [Honey](https://bee-swarm-simulator.fandom.com/wiki/Honey)
- Base honey per pollen without bonuses is about 1:1 **(unverified)**.

## Bags (capacity)

The Pouch is given to new players. The bag's capacity is shown top-middle as "Pollen".
Source: [Bags](https://bee-swarm-simulator.fandom.com/wiki/Bags), [Capacity](https://bee-swarm-simulator.fandom.com/wiki/Capacity)

| Bag | Capacity | Bonus | Cost | Shop |
|---|---|---|---|---|
| Pouch | 200 | none | free | Noob Shop |
| Jar | 750 | none | 650 honey | Noob Shop |
| Backpack | 3,500 | none | 5,500 honey | Noob Shop |
| Canister | 10,000 | +30% convert rate | 22,000 honey | Noob Shop |
| Mega-Jug | 25,000 | +40% convert rate | 50,000 honey | Pro Shop |
| Compressor | 50,000 | +55% convert rate | 160,000 honey | Pro Shop |
| Elite Barrel | 125,000 | +70% convert rate | 650,000 honey | Pro Shop |
| Port-O-Hive | 250,000 | +100% convert rate, +5% instant conversion | 1.25M honey | Pro Shop |
| Red / Blue Port-O-Hive | 400,000 | +150% convert rate, color bonuses | 12.5M honey + materials | Red / Blue HQ |
| Porcelain Port-O-Hive | 600,000 | +250% convert rate, more | 250M honey + materials | Mountain Top Shop |
| Coconut Canister | 1,000,000 | x1.25 capacity, +400% convert rate, passives | 25B honey + materials | Coconut Cave |

Other capacity sources stack on top: belts (+5,000 up to +350,000, some with % bonuses),
guards (+5,000 to +250,000), masks (x1.25 to x3), amulets, gifted Bumble (x1.1) and
Buoyant (x1.2) bees, potions and other buffs.

## Collectors (tools)

20 tools exist. Every player starts with the Scooper. Holding the mouse button auto-swings.
Source: [Tools](https://bee-swarm-simulator.fandom.com/wiki/Tools) and each tool's page.

| Tool | Cost | Flowers hit | Pattern | Swing time | Pollen per flower (single) | Base pollen/s |
|---|---|---|---|---|---|---|
| Scooper | free | 2 | line | 0.8 s | 2 | 5 |
| Rake | 800 honey | 3 | line | 0.7 s | 2 | 8.57 |
| Clippers | 2,200 honey | 1 | single | 0.6 s | 9 | 15 |
| Magnet | 5,500 honey | 9 | surround | 0.8 s | 2 | 22.5 |
| Vacuum | 14,000 honey | 13 | surround | 0.8 s | 2 | 32.5 |
| Honey Dipper | 1.5M honey | 49 | surround | 0.9 s | 2 | 108.89 |

Per-flower pollen scales with flower size (e.g. Scooper: 2/4/6/8/10 for single to star).
Price bands: Noob Shop 0 to 14K; Pro Shop 40K to 1.5M; Red/Blue HQ 3.5M to 2.5T;
Mountain Top 20M to 150M; Petal Shop 1.5B; Gummyballer (most expensive) 10T plus materials.
Other tools not read in detail: Super-Scooper, Pulsar, Electro-Magnet, Scissors, Bubble
Wand, Scythe, Sticker-Seeker, Golden Rake, Spark Staff, Porcelain Dipper, Petal Wand,
Tide Popper, Dark Scythe, Gummyballer.

## Fields

23 fields, including challenge-only ones: Sunflower, Dandelion, Mushroom, Blue Flower,
Clover, Strawberry, Spider, Bamboo, Pineapple, Stump, Cactus, Pumpkin, Pine Tree, Rose,
Hub, Mountain Top, Pepper, Coconut, Ant, and Mixed/Blue/Red/White Brick. Flower layouts are
regenerated per server. Fields can drop honey and treat tokens.
Source: [Fields](https://bee-swarm-simulator.fandom.com/wiki/Fields)

Example, Sunflower Field (a starter field, no bee requirement, no mobs): 20 × 33 = 660
flowers; about 70% white, 15% red, 15% blue; 90% single and 10% double flowers.
Source: [Sunflower Field](https://bee-swarm-simulator.fandom.com/wiki/Sunflower_Field)

- Field Boost: +100% pollen per stack in one field, up to +400%, 15 minutes; Field Booster
  can give +300% at once. Field Winds give instant conversion in that field.
- Access gates by bee count (5/10/15/20/25/35 bees) exist; the Pro Shop sits behind the
  10-bee gate **(unverified; the gate page was not found)**.

## Bees and the hive

Bees follow the player, gather pollen, fight mobs, make ability tokens, and convert pollen
at the hive. They sleep at the hive when out of energy. Bonding with treats raises levels,
which increases gathering, converting, energy and the mob levels they can hit.
Source: [Bees](https://bee-swarm-simulator.fandom.com/wiki/Bees)

- 46 bee types; rarities Common, Rare, Epic, Legendary, Mythic, Event.
- Hive: 25 default slots, up to 25 more from the Mountain Top Shop (50 total), with
  exponentially rising prices.
- Gifted bees have better stats and give one hive-wide bonus. Chance when feeding a
  favourite treat: 1/8,000 Rare, 1/10,000 Epic, 1/12,000 Legendary or Common, 1/24,000
  Mythic. Royal Jelly or a Basic Egg gives about 1/287. Star Jelly, Star Treat and Star Egg
  guarantee gifted.
- Example stats (energy, speed, gather, convert):

  | Bee | Rarity | Color | Energy | Speed | Gather | Convert | Gifted bonus |
  |---|---|---|---|---|---|---|---|
  | Basic | Common | colorless | 20 | 14 | 10 pollen / 4 s | 80 honey / 4 s | x1.2 pollen |
  | Hasty | Rare | colorless | 20 | 19.6 | 10 / 3 s | 80 / 3 s | +15% player movespeed |
  | Bumble | Rare | blue | 50 | 10.5 | 18 / 4 s | 80 / 4 s | x1.1 capacity |
  | Honey | Epic | colorless | 20 | 14 | 10 / 4 s | 360 / 2 s | x1.5 honey from tokens |
  | Diamond | Legendary | blue | 20 | 14 | 10 / 4 s | 1,000 / 4 s | x1.2 convert rate |
  | Buoyant | Mythic | blue | 60 | 14 | 15 / 5 s | 150 / 3 s | x1.2 capacity |

## Eggs

Source: [Egg](https://bee-swarm-simulator.fandom.com/wiki/Egg)

- Types: Basic, Silver, Gold, Diamond, Mythic, Star, plus gifted variants, specific-bee
  eggs and event-bee eggs. An egg can be hatched onto an existing bee to replace it,
  keeping the level.
- Basic Egg: 87% Basic Bee, 10% Rare, 2.5% Epic, 0.5% Legendary, about 0.35% Gifted Basic.
  New players get one free; the shop near Sunflower Field starts at 1,000 honey and the
  price rises per purchase up to 10M.
- Silver about 64.9% Rare, Gold about 79% Epic, Diamond about 95% Legendary, Mythic always
  Mythic **(unverified)**.

## Mobs

34 mob kinds. Field mobs attack players who enter their field and respawn on a timer.
Defeated mobs drop a ring of tokens. Bees below a mob's level miss more often.
Source: [Mobs](https://bee-swarm-simulator.fandom.com/wiki/Mobs)

- Ladybug: level 1 to 3; Mushroom, Clover, Strawberry; respawn 5 min **(unverified)**.
- Rhino Beetle: most common (5 spawn points); level 1 to 5 **(unverified)**.
- Spider: level 4, Spider Field, respawn 30 min **(unverified)**.
- Werewolf: level 7; Cactus, Pine Tree, Pumpkin; respawn 1 h **(unverified)**.
- Stump Snail has the most health and the longest respawn, 96 h.
- Bosses: King Beetle, Coconut Crab, Tunnel Bear, Mondo Chick, Stick Bug.

## Boosts and tokens (unverified)

Values from search snippets of the Buffs & Debuffs, Field Boost and Ability Tokens pages.

| Buff | Per stack | Cap | Duration |
|---|---|---|---|
| Red / Blue / White Boost | x1.2 pollen of that color | x3 | 15 s |
| Haste | x1.1 movespeed | x2 | 20 s |
| Focus | +3% critical chance | +30% | 20 s |

## Shops and currencies

- Currencies: pollen, honey, tickets, bricks, cogs, snowflakes, gingerbread bears.
- Honey buys tools, bags, accessories, tickets, Basic Eggs, Royal Jelly, treats and hive
  slots; many consumables get pricier with each purchase.
- Noob Shop (next to Dandelion Field): 4 bags, 5 tools, 1 hat, boots and a belt. Pro Shop
  (beyond the 10-bee gate, by Pineapple Patch): 18 items **(unverified)**.
- Royal Jelly Shop: starts at 250,000 honey, +10,000 each, capped at 1M **(unverified)**.

## Not found

Player walk speed and jump power, player health and defeat rules, exact flower regrowth
timing, the full bee-gate table, quest bears, and the remaining tool stats.

## Compared with our arena

The arena (see [ARENA.md](ARENA.md)) is an original simplified prototype, not a replica.

| | BSS start | FlyBrain arena |
|---|---|---|
| Starting bag | Pouch, 200 | 300 |
| Starting tool | Scooper: 2 flowers, 2 pollen each, 0.8 s | dipper, 0.8 s cooldown |
| Conversion | per bee, Basic Bee 20 honey/s | 5 simulated bees, 60 pollen/s total |
| Honey per pollen | about 1 (unverified) | 1 |
| Flower colors | white, red, blue, 5 sizes | single type, 12 s regrowth |
