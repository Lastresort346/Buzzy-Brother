# Communication center

`flybrain/arena/comms.py` lets the bee tell the trainer what it is measuring, using a
fixed library of sentences. It is read-only: it does not feed the controller, change
weights or checkpoints, or affect training. Each message carries the numbers that
triggered it (`evidence`) so it can be audited. Messages appear in a "BEE SAYS" panel
in the neural monitor, are printed as `[bee] ...`, and are written to the session
`telemetry.jsonl` as records `{"simulation_seconds", "message": {id, text, evidence, t}}`.

## Honesty caveat

The sentences are templates selected by threshold rules on measured state. "I want X"
is shorthand for a measured priority, not a feeling, intent or neural decision: the
in-stock item with the highest current utility per honey (the same marginal utility
`HoneyShop.buy` pays), affordable now for "I want", not yet affordable for "saving".
"I prefer blue flowers" means blue has the highest pollen yield so far. Nothing here
claims subjective experience. Inputs are public status (pack, wallet, shop), the
bee's own odometry memory and goal, and history kept in the center. The one world
read is the colour of the flower nearest the dipper when pollen arrives (contact
sensing); no world coordinates are used otherwise. A swing near two colours can
mis-attribute a gain.

## Presets

| id | sentence | triggered when |
|---|---|---|
| `want_item` | I want a {item}. | an in-stock item is affordable; best utility/price among them |
| `saving_for` | I am saving honey for a {item}. | nothing in stock is affordable; best utility/price overall |
| `bought` | I bought a {item}. | an item's owned count rose |
| `low_honey` | My honey is low. | honey is below the cheapest in-stock price |
| `bag_full` | My bag is full. | pollen at capacity |
| `bag_empty` | My bag is empty. | pollen 0 and not converting |
| `stuck` | I am stuck. | movement commanded but odometry moved under 0.5 for 3 s |
| `color_pref` | I prefer {color} flowers. | at least 100 pollen over 2+ colours and the leader has 1.2x the runner-up |
| `new_field` | I need to find a new field. | a goal is set (including DISCOVER) that memory has no named node for |
| `go_field` | I want to go to the {field}. | the goal field is remembered; evidence includes odometry distance |
| `name_field` | I found a field that needs a name. | `memory.discoveries()` is non-empty |

## Rate limiting

A preset is not repeated while its state key is unchanged (item name, goal, leading
colour, purchase count, discovery count), is silent for 10 simulated seconds after
being said even if its condition ends and returns, and at most one message is sent per
simulated second (others wait for the next update).

## Reading it

Read the text as a status report and check `evidence` before acting: a `stuck` with
`seconds_without_progress`, a `color_pref` with `pollen_by_color`, a `want_item` with
price, honey and utility per price. To act on `name_field`, name the field as usual
through the tutor/memory flow. Thresholds are constants at the top of `comms.py`.
