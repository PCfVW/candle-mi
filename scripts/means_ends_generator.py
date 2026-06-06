#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate goal-contrastive means-ends planning items.

Step A (linguistic cell) of the prolepsis-in-planning experiment: emit a JSON
list of means-ends operator-selection items where each prompt states a current
state and a goal, then ends at the planning site (no trailing space). The
model's next token should be the goal-correct single-token action.

## Goal-contrastive design

Every device appears in **both directions** of its action family, e.g. for the
`on/off` family the lamp yields one item whose goal calls for `on` and one whose
goal calls for `off`. The set is **balanced** across the six (family, token)
groups, so the "override the lexical default" sides (`off`, `shut`, `down`) carry
equal weight — those are the cases that distinguish genuine goal-conditioning
from emitting the high-frequency collocation ("turn the lamp on").

## Planning-site convention (must match the Rust scorer)

Each prompt ends at a content token with **no trailing space**. A trailing space
tokenizes as a standalone metaspace token, after which the space-prefixed answer
token is unreachable. The model's next token is the space-prefixed action token
(scored via `find_token_id`).

## Action families

Three families, each a contrastive single-token pair and a stem ending at the
planning site:

- `on/off`     — `"... Turn the {device}"`       -> `on` | `off`
- `open/shut`  — `"... The {device} should be"`  -> `open` | `shut`
- `up/down`    — `"... The {device} should be"`  -> `up` | `down`

(Short, high-frequency tokens chosen so both sides are single tokens; the Rust
scorer validates this and reports any that are not.)

## Output schema (flat list)

```json
[{"prompt": str, "correct": str, "alternative": str, "family": str}, ...]
```

## Usage

```bash
python scripts/means_ends_generator.py --num-instances 200 --seed 0 \
    --output docs/experiments/means-ends-prolepsis/means_ends_items.json
```
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OUTPUT = "docs/experiments/means-ends-prolepsis/means_ends_items.json"

# Each family: token pair (a, b), the stem ending at the planning site, and a
# device -> {a_goal, b_goal} table. The state clause shows the OPPOSITE token
# (you act to change it): for the `a` direction the device currently shows `b`.
FAMILIES = {
    "on_off": {
        "tokens": ("on", "off"),
        "stem": "Turn the {device}",
        "devices": {
            "lamp": ("We want the room to be bright", "We want the room to be dark"),
            "heater": ("We want the room to be warm", "We want the room to be cool"),
            "fan": ("We want the air to keep moving", "We want the air to stay still"),
            "radio": ("We want to hear some music", "We want complete silence"),
            "television": ("We want to watch the news", "We want some quiet"),
            "oven": ("We want to bake the bread", "We want to stop the heat"),
            "kettle": ("We want to boil the water", "We want to stop boiling"),
            "computer": ("We want to start working", "We want to save power"),
            "printer": ("We want to print a page", "We want to save power"),
            "speaker": ("We want to play the song", "We want it to be silent"),
            "projector": ("We want to show the slides", "We want a dark screen"),
            "stove": ("We want to cook dinner", "We want to stop cooking"),
        },
    },
    "open_closed": {
        "tokens": ("open", "closed"),
        "stem": "The {device} should be",
        "devices": {
            "door": ("We want to walk through", "We want some privacy"),
            "window": ("We want some fresh air", "We want to keep out the cold"),
            "gate": ("We want to let the car in", "We want to keep the dog inside"),
            "valve": ("We want to let the water flow", "We want to stop the leak"),
            "lid": ("We want to reach inside", "We want to keep it sealed"),
            "curtain": ("We want to let the sunlight in", "We want to block the glare"),
            "drawer": ("We want to get a spoon", "We want to tidy the desk"),
            "cabinet": ("We want to take a plate", "We want to hide the clutter"),
            "jar": ("We want to take a cookie", "We want to keep it fresh"),
            "trunk": ("We want to load the bags", "We want to drive away"),
            "hatch": ("We want to climb out", "We want to stay warm"),
            "box": ("We want to see what is inside", "We want to keep it safe"),
        },
    },
    "up_down": {
        "tokens": ("up", "down"),
        "stem": "The {device} should be",
        "devices": {
            "shade": ("We want to see the view", "We want to keep the room cool"),
            "roller blind": ("We want to let the sunlight in", "We want to block the bright sun"),
            "volume": ("We want to hear the song clearly", "We want to avoid waking the baby"),
            "thermostat": ("We want to make the room warmer", "We want to make the room cooler"),
            "car window": ("We want to keep the rain out", "We want to get some fresh air"),
            "garage door": ("We want to drive the car out", "We want to secure the garage"),
            "window blind": ("We want to let the morning light in", "We want to darken the room for sleep"),
            "projector screen": ("We want to use the whiteboard behind it", "We want to show the movie"),
            "hospital bed": ("We want to sit up and eat", "We want to lie flat and rest"),
            "recliner": ("We want to sit upright to work", "We want to lie back and relax"),
            "tray table": ("We want to get ready for landing", "We want to eat the meal"),
            "seat back": ("We want to prepare for takeoff", "We want to recline and rest"),
        },
    },
}


# --- Step B controlled set (on_off only) -------------------------------------
#
# Device-once: the device is named only in the stem ("Turn the {device}"); the
# state and goal clauses describe the *world* (no device token), so the
# planning-site spike can be attributed to a clause without a repeated device
# mention. World-state framing also avoids pronoun anaphora that would bind to
# the wrong noun in one of the two orders. Each device has a (state, goal) for
# the "on" direction and for the "off" direction.
CONTROLLED_OUTPUT = "docs/experiments/means-ends-prolepsis/step_b_items.json"

CONTROLLED_ON_OFF = {
    "lamp": {"on": ("The room is dark.", "We want the room to be bright."),
             "off": ("The room is bright.", "We want the room to be dark.")},
    "heater": {"on": ("The room is cold.", "We want the room to be warm."),
               "off": ("The room is hot.", "We want the room to be cool.")},
    "fan": {"on": ("The air is stuffy.", "We want the air to keep moving."),
            "off": ("The air is too windy.", "We want the air to be still.")},
    "radio": {"on": ("The room is silent.", "We want to hear some music."),
              "off": ("The room is noisy.", "We want some silence.")},
    "television": {"on": ("There is news to catch up on.", "We want to watch the news."),
                   "off": ("The show is over.", "We want some quiet.")},
    "oven": {"on": ("The dough is raw.", "We want to bake the bread."),
             "off": ("The bread is baked.", "We want to stop the heat.")},
    "kettle": {"on": ("The water is cold.", "We want to boil the water."),
               "off": ("The water is boiling.", "We want to stop the boiling.")},
    "computer": {"on": ("There is work to do.", "We want to start working."),
                 "off": ("The work is done.", "We want to save power.")},
    "printer": {"on": ("A page needs printing.", "We want to print the page."),
                "off": ("The printing is finished.", "We want to save power.")},
    "speaker": {"on": ("The room is quiet.", "We want to play some music."),
                "off": ("The music is too loud.", "We want some quiet.")},
    "projector": {"on": ("The screen is blank.", "We want to show the slides."),
                  "off": ("The slides are finished.", "We want a dark screen.")},
    "stove": {"on": ("The food is raw.", "We want to cook the food."),
              "off": ("The food is cooked.", "We want to stop cooking.")},
}


CONTRASTIVE_OUTPUT = "docs/experiments/means-ends-prolepsis/step_b_contrastive_pairs.json"

# Goal-only minimal contrastive pairs for activation patching. Each device pairs
# an on-goal (clean) and an off-goal (corrupt) single-token antonym in one fixed
# frame, so the two prompts differ at EXACTLY the goal word (token-aligned). No
# state clause: the goal alone determines the action, isolating the goal signal.
# (device, entity, on_goal, off_goal)
CONTRASTIVE_DEVICES = [
    ("lamp", "room", "bright", "dark"),
    ("heater", "room", "warm", "cool"),
    ("oven", "oven", "hot", "cold"),
    ("stove", "stove", "hot", "cold"),
    ("kettle", "water", "hot", "cold"),
    ("radio", "room", "loud", "quiet"),
    ("speaker", "room", "loud", "quiet"),
    ("projector", "screen", "bright", "dark"),
]

CONTRASTIVE_FRAME = "We want the {entity} to be {goal}. Turn the {device}"


VERSIFIED_OUTPUT = "docs/experiments/means-ends-prolepsis/versified_couplets.json"

# Versified means-ends: a rhyming couplet whose final word must satisfy BOTH the
# rhyme (set by line 1's last word) AND the goal (the action/state). `W` is the
# dual rhyme+goal target; `Wp` is the goal-WRONG antonym (scored as the
# `alternative`). line2 ends at a cue with NO trailing space and NO occurrence of
# `W`, so the model's next token is the completion. Each tuple is (line1, line2).
VERSIFIED_FAMILIES = [
    {"family": "glow_dim", "W": "glow", "Wp": "dim", "couplets": [
        ("The room is dim, the lamps are low,", "we want it bright, so make it"),
        ("The fire has faded, embers low,", "we want more light, so make it"),
        ("The screen is dark, its backlight low,", "we want to see, so make it"),
        ("The porch is dark, the wattage low,", "we want it cheery, so make it"),
        ("The lantern's faint, its wick set low,", "we want a beacon, so make it"),
        ("The streetlamp flickers, dull and low,", "we want it shining, so make it"),
        ("The hallway dims, the voltage low,", "we want it brilliant, so make it"),
        ("The bulb is weak, its current low,", "we want it radiant, so make it"),
    ]},
    {"family": "bright_dark", "W": "bright", "Wp": "dark", "couplets": [
        ("The bulb is dead, the room is night,", "we want to read, so make it"),
        ("We grope around, deprived of light,", "we need to work, so make it"),
        ("The dusk has fallen, dim the light,", "we want to sew, so make it"),
        ("The cellar's black as starless night,", "we want to search, so make it"),
        ("The study's gloomy, poor of light,", "we want to focus, so make it"),
        ("The lamp burns weak, a dying light,", "we want it dazzling, so make it"),
        ("The workshop dims at fall of night,", "we want precision, so make it"),
        ("The garret's grey, devoid of light,", "we want to paint, so make it"),
    ]},
    {"family": "warm_cool", "W": "warm", "Wp": "cool", "couplets": [
        ("Outside there rages a winter storm,", "we're chilled inside, so make it"),
        ("The cabin's frigid after the storm,", "we want some heat, so make it"),
        ("The wind howls cold in a dreadful storm,", "we want to thaw, so make it"),
        ("Snow buries the road in a morning storm,", "we seek some comfort, so make it"),
        ("The barn runs cold, below its norm,", "the calves are shivering, so make it"),
        ("The tent is icy through the storm,", "we want to rest, so make it"),
        ("The office froze beneath the storm,", "the staff are stamping, so make it"),
    ]},
    {"family": "cool_warm", "W": "cool", "Wp": "warm", "couplets": [
        ("The afternoon is hot beside the pool,", "we want relief, so make it"),
        ("The classroom's stifling, like a school,", "we want fresh air, so make it"),
        ("The engine's redlined, breaking every rule,", "before it fails, so make it"),
        ("The greenhouse roasts, a sweltering pool,", "to save the ferns, so make it"),
        ("The dorm is baking, hot as a school,", "we want to sleep, so make it"),
        ("The server room is far too warm to cool", "with one small fan, so make it"),
        ("The kiln still glows, too hot to spool,", "we want to handle it, so make it"),
    ]},
    {"family": "loud_soft", "W": "loud", "Wp": "soft", "couplets": [
        ("The hall is hushed before the crowd,", "we want to hear, so make it"),
        ("The speaker's muffled, no sound allowed,", "we want some music, so make it"),
        ("The whisper's lost amid the crowd,", "we want to reach them, so make it"),
        ("The set is silent, no cheer allowed,", "the fans are waiting, so make it"),
        ("The band plays faint, the room not proud,", "we want a party, so make it"),
        ("The intercom is barely loud", "above the din, so make it"),
        ("The alarm is soft, its chime not loud", "across the floor, so make it"),
    ]},
    {"family": "hot_cold", "W": "hot", "Wp": "cold", "couplets": [
        ("The kettle waits in its usual spot,", "we want some tea, so make it"),
        ("The stew's gone lukewarm in the pot,", "we want to serve it, so make it"),
        ("The bath has cooled, no longer hot,", "we want to soak, so make it"),
        ("The iron's tepid, pressing not,", "to smooth the shirt, so make it"),
        ("The grill's barely warm, a feeble spot,", "we want to sear, so make it"),
        ("The coffee's cold, forgotten in its pot,", "we want a sip, so make it"),
        ("The sauna's cooled, no steam, no spot of heat,", "we want to sweat, so make it"),
    ]},
    {"family": "cold_hot", "W": "cold", "Wp": "hot", "couplets": [
        ("The lemonade is flat and warm and old,", "we want refreshment, so make it"),
        ("The soda's tepid, sitting where it's told,", "we want a chill, so make it"),
        ("The fridge gave out, the milk no longer cold,", "we want it fresh, so make it"),
        ("The cellar's mild where wine should keep its cold,", "we want it crisp, so make it"),
        ("The plunge pool's heated, brave and bold,", "we want a shock, so make it"),
        ("The compress warmed, its ice all sold,", "to ease the swelling, so make it"),
        ("The brew is room-warm, going old,", "we want it bracing, so make it"),
    ]},
    {"family": "down_up", "W": "down", "Wp": "up", "couplets": [
        ("The banner hangs high above the town,", "the fete is over, so bring it"),
        ("The flag flies proud across the town,", "the day is done, so take it"),
        ("The blind is raised, the glare its crown,", "we want to nap, so pull it"),
        ("The drone hangs high above the town,", "the flight is finished, so set it"),
        ("The drawbridge towers over the town,", "to let us cross, so lower it"),
        ("The shade is lifted, the room a-frown,", "the sun is harsh, so draw it"),
        ("The mast stands tall above the town,", "the storm approaches, so haul it"),
    ]},
    {"family": "high_low", "W": "high", "Wp": "low", "couplets": [
        ("The kite hangs limp against the sky,", "we want it soaring, so raise it"),
        ("The volume's faint, a feeble sigh,", "we want to rock, so crank it"),
        ("The drone sits grounded, longing to fly,", "we want a view, so send it"),
        ("The flame burns small, it will not fly,", "we want a blaze, so turn it"),
        ("The thermostat reads cold and dry,", "we want some heat, so set it"),
        ("The swing hangs still beneath the sky,", "the child is eager, so push it"),
        ("The balloon won't rise, however we try,", "we want it up, so let it"),
    ]},
    {"family": "shut_open", "W": "shut", "Wp": "open", "couplets": [
        ("A cold wind whistles through the cut,", "we want it sealed, so keep it"),
        ("The gate swings wide, the latch undone, a rut,", "the dog might bolt, so pull it"),
        ("Mosquitoes drift in past the strut,", "we want them gone, so slam it"),
        ("The vault stands open, bare its gut,", "the guard is nervous, so swing it"),
        ("The lid is loose atop the butt,", "to keep it fresh, so press it"),
        ("The door bangs open in the rut,", "the draft is freezing, so hold it"),
        ("The window gapes above the hut,", "the rain is coming, so push it"),
    ]},
    {"family": "wide_narrow", "W": "wide", "Wp": "narrow", "couplets": [
        ("The harbor opens, deep and wide,", "let the ships in, so throw it"),
        ("The curtains part to show the tide,", "we want the view, so fling them"),
        ("The aperture is cramped inside,", "we want more light, so set it"),
        ("The gateway's pinched, too tight to ride,", "the truck must pass, so spread it"),
        ("The lens is stopped, the field denied,", "we want it sweeping, so open it"),
        ("The valve is choked, the flow belied,", "we want full pressure, so crank it"),
        ("The path is narrow by the riverside,", "the crowd is surging, so make it"),
    ]},
]


def build_versified():
    """Build ~100 versified means-ends couplets.

    Each prompt is a couplet ending at a cue (no trailing space); the model's
    next token should be `W`, the word that satisfies BOTH the rhyme (with line
    1's last word, the `anchor`) AND the goal. `Wp` is the goal-wrong antonym
    (the `alternative`). The Rust scorer (`means_ends_prolepsis`) reports the
    greedy top-1 and `P(W)` vs `P(Wp)`; rhyme is checked downstream from the
    `anchor` via CMUdict.
    """
    items = []
    idx = 0
    for fam in VERSIFIED_FAMILIES:
        for line1, line2 in fam["couplets"]:
            anchor = line1.rstrip(",.;:!?").split()[-1].lower()
            items.append({
                "id": idx,
                "family": fam["family"],
                "prompt": f"{line1}\n{line2}",
                "correct": fam["W"],
                "alternative": fam["Wp"],
                "anchor": anchor,
                "target": fam["W"],
            })
            idx += 1
    return items


VERSIFIED2_OUTPUT = "docs/experiments/means-ends-prolepsis/versified2_couplets.json"

# v2 — FORCED-NON-DEFAULT versified means-ends. In v1 most families set `W` = the
# DEFAULT goal word and rigged line 1 to rhyme with it, so rhyme was incidental. Here
# EVERY family's `W` is a NON-default synonym that the goal alone would NOT produce
# (the default `D` is what line 2 primes — "we want it bright" pulls `bright`, but the
# rhyme forces `glow`). So a dual-hit (`C == W`) can only happen by planning the
# rhyme-and-goal word over the primed default. `Wp` = goal-wrong antonym.
VERSIFIED2_FAMILIES = [
    {"family": "brighten_glow", "W": "glow", "Wp": "dim", "couplets": [
        ("The room is dim, the lamps are low,", "we want it bright, so make it"),
        ("The hearth is cold, the embers low,", "we want it bright, so make it"),
        ("The screen has dimmed, its setting low,", "we want it bright, so make it"),
        ("Dusk settles in, the sunlight low,", "we want some brightness, so make it"),
        ("The lantern's wick is burning low,", "we want it bright, so make it"),
        ("The porch is dark, the bulb hung low,", "we want it bright, so make it"),
        ("The fire dies down, its flicker low,", "we want it bright, so make it"),
    ]},
    {"family": "darken_dim", "W": "dim", "Wp": "bright", "couplets": [
        ("The glare is harsh, the outlook grim,", "we want it dark, so make it"),
        ("The spotlight's bright around the rim,", "we want it dark, so make it"),
        ("The studio is lit to the brim,", "we want it dark, so make it"),
        ("The film needs shadow, mood-lit, slim,", "we want it darker, so make it"),
        ("The hall is dazzling, edge and rim,", "we want it dark, so make it"),
        ("The pool reflects the sun, each whim,", "we want it shaded, so make it"),
        ("The lamp is fierce, no shadow, prim,", "we want it dark, so make it"),
    ]},
    {"family": "quieten_still", "W": "still", "Wp": "loud", "couplets": [
        ("The engine roars atop the hill,", "we want it quiet, so make it"),
        ("The crowd is restless, voices shrill,", "we want it quiet, so make it"),
        ("The workshop clatters with the drill,", "we want some quiet, so make it"),
        ("The brook runs noisy past the mill,", "we want it silent, so make it"),
        ("The traffic blares beneath the hill,", "we want it hushed, so make it"),
        ("The fans are chanting, loud and shrill,", "we want it quiet, so make it"),
        ("The downpipe gurgles, fit to spill,", "we want it calm, so make it"),
    ]},
    {"family": "speedup_quick", "W": "quick", "Wp": "slow", "couplets": [
        ("The queue is long, the traffic thick,", "we're running late, so make it"),
        ("The mud is deep, the going slick,", "we want it fast, so make it"),
        ("The clock is ticking, every click,", "we want it fast, so make it"),
        ("The download crawls, the network thick,", "we want it fast, so make it"),
        ("He fumbles slowly with the stick,", "we want it fast, so make it"),
        ("The path is icy, treacherous, slick,", "we want it rapid, so make it"),
        ("The candle gutters on its wick,", "we want it brisk, so make it"),
    ]},
    {"family": "tidy_neat", "W": "neat", "Wp": "messy", "couplets": [
        ("The papers sprawl across the seat,", "the guests arrive, so make it"),
        ("Mud tracks the floor from dirty feet,", "we want it clean, so make it"),
        ("The stall's a mess along the street,", "we want it tidy, so make it"),
        ("Crumbs scatter on the rumpled sheet,", "we want it clean, so make it"),
        ("The desk is cluttered, far from sweet,", "we want it tidy, so make it"),
        ("The booth is chaos where we meet,", "we want it clean, so make it"),
        ("Leaves choke the gutter down the street,", "we want it spotless, so make it"),
    ]},
    {"family": "enlarge_vast", "W": "vast", "Wp": "small", "couplets": [
        ("The map is cramped, the scale held fast,", "we want it big, so make it"),
        ("The tiny stage from ages past,", "we want it large, so make it"),
        ("The model's small, the first we cast,", "we want it huge, so make it"),
        ("The signal's range is fading fast,", "we want it wider, so make it"),
        ("The garden's narrow by the mast,", "we want it sprawling, so make it"),
        ("The screen is little, holding fast,", "we want it big, so make it"),
        ("The courtyard shrank from what was last,", "we want it open, so make it"),
    ]},
    {"family": "cheer_glad", "W": "glad", "Wp": "sad", "couplets": [
        ("The news was grim, the mood turned sad,", "we want her happy, so make her"),
        ("The boy is sulking, feeling bad,", "we want him cheerful, so make him"),
        ("The team has lost, the fans are mad,", "we want them happy, so make them"),
        ("The day went wrong for the weary lad,", "we want him joyful, so make him"),
        ("Her spirits sank, her face was sad,", "we want her happy, so make her"),
        ("The crowd is sombre, dully clad,", "we want them merry, so make them"),
        ("The toddler cries, his morning bad,", "we want him happy, so make him"),
    ]},
    {"family": "moisten_damp", "W": "damp", "Wp": "dry", "couplets": [
        ("The soil is dust beside the lamp,", "we want it wet, so make it"),
        ("The tent is parched at summer camp,", "we want it wet, so make it"),
        ("The sponge is dry upon the ramp,", "we want it wet, so make it"),
        ("The cloth is crisp, it gives a cramp,", "we want it moist, so make it"),
        ("The envelope won't hold its stamp,", "we want it wet, so make it"),
        ("The wick is dry inside the lamp,", "we want it moist, so make it"),
        ("The towel's stiff beside the ramp,", "we want it wet, so make it"),
    ]},
    {"family": "crackopen_ajar", "W": "ajar", "Wp": "shut", "couplets": [
        ("The room is stuffy, the window far,", "we want some air, so leave it"),
        ("Fumes linger thick inside the car,", "we want it open, so leave it"),
        ("The cellar's stale behind the bar,", "we want some air, so leave it"),
        ("The closet reeks of old cigar,", "we want it aired, so leave it"),
        ("The bedroom's hot, the moon a star,", "we want a breeze, so leave it"),
        ("The kitchen smokes, the door's a bar,", "we want it open, so leave it"),
        ("The loft is airless, faintly char,", "we want it vented, so leave it"),
    ]},
    {"family": "sharpen_keen", "W": "keen", "Wp": "blunt", "couplets": [
        ("The blade is dull, the dullest seen,", "we want it sharp, so make it"),
        ("The scissors blunt, the edge not clean,", "we want it sharp, so make it"),
        ("The knife won't cut, its grind too lean,", "we want it sharp, so make it"),
        ("The chisel's worn, as you have seen,", "we want it sharp, so make it"),
        ("The sword hangs dull above the queen,", "we want it sharp, so make it"),
        ("The plane blade drags across the green,", "we want it sharp, so make it"),
        ("The razor pulls, its bevel mean,", "we want it sharp, so make it"),
    ]},
    {"family": "lighten_pale", "W": "pale", "Wp": "dark", "couplets": [
        ("The wall is navy, dark as a jail,", "we want it light, so make it"),
        ("The paint's too deep along the rail,", "we want it light, so make it"),
        ("The fabric's inky, like a snail's trail,", "we want it light, so make it"),
        ("The portrait's dim, its colours frail,", "we want it lighter, so make it"),
        ("The icing's brown along the pail,", "we want it light, so make it"),
        ("The sky-blue faded down the trail,", "we want it lighter, so make it"),
        ("The curtain's crimson, stiff as a sail,", "we want it light, so make it"),
    ]},
    {"family": "cozy_snug", "W": "snug", "Wp": "cold", "couplets": [
        ("The night is freezing, on the rug,", "we want it cozy, so make it"),
        ("The cabin's drafty, by the mug,", "we want it warm, so make it"),
        ("The baby's cold beneath the rug,", "we want it cozy, so make it"),
        ("The window leaks beside the jug,", "we want it warm, so make it"),
        ("The tent is chilly where we dug,", "we want it cozy, so make it"),
        ("The attic's cold, we'll pull the plug,", "we want it warm, so make it"),
        ("The blanket's thin upon the rug,", "we want it cozy, so make it"),
    ]},
]


def build_versified2():
    """Build the FORCED-NON-DEFAULT versified couplets (v2).

    Same schema as `build_versified`, but every `W` is a NON-default synonym that
    the goal (which line 2 states via the default word `D`) would not produce on
    its own — only the rhyme forces it. So `C == W` (dual-hit) is a genuine
    planning signal in every family, not the incidental rhyme of v1.
    """
    items = []
    idx = 0
    for fam in VERSIFIED2_FAMILIES:
        for line1, line2 in fam["couplets"]:
            anchor = line1.rstrip(",.;:!?").split()[-1].lower()
            items.append({
                "id": idx,
                "family": fam["family"],
                "prompt": f"{line1}\n{line2}",
                "correct": fam["W"],
                "alternative": fam["Wp"],
                "anchor": anchor,
                "target": fam["W"],
            })
            idx += 1
    return items


def build_contrastive():
    """Build token-aligned clean/corrupt goal-flip pairs (on-goal vs off-goal).

    Each pair shares the frame and differs at exactly the goal word, so patching
    a residual at position p is well defined. The Rust side re-validates
    token-alignment and competence and prunes any pair that fails.
    """
    items = []
    for idx, (device, entity, on_goal, off_goal) in enumerate(CONTRASTIVE_DEVICES):
        clean = CONTRASTIVE_FRAME.format(entity=entity, goal=on_goal, device=device)
        corrupt = CONTRASTIVE_FRAME.format(entity=entity, goal=off_goal, device=device)
        items.append({
            "id": idx,
            "family": "on_off",
            "device": device,
            "entity": entity,
            "clean_prompt": clean,
            "corrupt_prompt": corrupt,
            "clean_action": "on",
            "corrupt_action": "off",
            "goal_clean": on_goal,
            "goal_corrupt": off_goal,
        })
    return items


def build_controlled():
    """Build the ~48-item Step-B on_off set: each device x direction x order,
    device-once, segment-annotated, prompt ending at the planning site."""
    items = []
    idx = 0
    for device, directions in CONTROLLED_ON_OFF.items():
        stem = f"Turn the {device}"
        for direction in ("on", "off"):
            state, goal = directions[direction]
            alternative = "off" if direction == "on" else "on"
            for order in ("initial_goal", "goal_initial"):
                prompt = (
                    f"{state} {goal} {stem}"
                    if order == "initial_goal"
                    else f"{goal} {state} {stem}"
                )
                items.append({
                    "id": idx,
                    "family": "on_off",
                    "order": order,
                    "device": device,
                    "correct": direction,
                    "alternative": alternative,
                    "segments": {"initial": state, "goal": goal, "stem": stem},
                    "prompt": prompt,
                })
                idx += 1
    return items


def render(device, state_token, goal, stem, template_idx):
    """Render one prompt; ends at the planning site (no trailing space)."""
    state = f"The {device} is {state_token}."
    stem_text = stem.format(device=device)
    if template_idx == 0:
        return f"{state} {goal}. {stem_text}"
    if template_idx == 1:
        return f"{goal}. {state} {stem_text}"
    return f"Right now, the {device} is {state_token}. {goal}. {stem_text}"


def build_all():
    """Build every candidate item, grouped by (family, correct_token)."""
    by_group = {}
    for family, spec in FAMILIES.items():
        tok_a, tok_b = spec["tokens"]
        stem = spec["stem"]
        for device, (goal_a, goal_b) in spec["devices"].items():
            # (correct, alternative, opposite-state token, goal text)
            for correct, alternative, goal in (
                (tok_a, tok_b, goal_a),
                (tok_b, tok_a, goal_b),
            ):
                for template_idx in range(3):
                    prompt = render(device, alternative, goal, stem, template_idx)
                    item = {
                        "prompt": prompt,
                        "correct": correct,
                        "alternative": alternative,
                        "family": family,
                    }
                    by_group.setdefault((family, correct), []).append(item)
    return by_group


def balanced_sample(by_group, num_instances, rng):
    """Draw a (family, token)-balanced sample, remainder distributed round-robin."""
    groups = sorted(by_group)
    base, remainder = divmod(num_instances, len(groups))
    chosen = []
    for i, key in enumerate(groups):
        want = base + (1 if i < remainder else 0)
        pool = by_group[key]
        if want > len(pool):
            raise ValueError(
                f"group {key} has only {len(pool)} candidates but {want} requested "
                f"(add devices/templates or lower --num-instances)"
            )
        chosen.extend(rng.sample(pool, want))
    rng.shuffle(chosen)
    return chosen


def write_items(items, output):
    """Write the item list to `output` (pretty JSON, trailing newline)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
        f.write("\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--controlled",
        action="store_true",
        help="emit the Step-B controlled on_off set (device-once, order-tagged, "
        "segment-annotated) instead of the balanced Step-A set",
    )
    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="emit goal-only token-aligned clean/corrupt pairs for activation "
        "patching (bright/dark goal flip; no state clause)",
    )
    parser.add_argument(
        "--versified",
        action="store_true",
        help="emit ~100 versified means-ends couplets (final word must satisfy "
        "both rhyme and goal); scored via means_ends_prolepsis + a CMUdict rhyme check",
    )
    parser.add_argument(
        "--versified-v2",
        dest="versified_v2",
        action="store_true",
        help="emit the FORCED-NON-DEFAULT versified couplets (every family's W is a "
        "non-default rhyming synonym of the goal); the real planning test at scale",
    )
    args = parser.parse_args()

    if args.versified:
        output = args.output or Path(VERSIFIED_OUTPUT)
        items = build_versified()
        write_items(items, output)
        fam_counts = Counter(i["family"] for i in items)
        print(f"Wrote {len(items)} versified couplets to {output}", file=sys.stderr)
        print(f"Per-family: {dict(sorted(fam_counts.items()))}", file=sys.stderr)
        return

    if args.versified_v2:
        output = args.output or Path(VERSIFIED2_OUTPUT)
        items = build_versified2()
        write_items(items, output)
        fam_counts = Counter(i["family"] for i in items)
        print(f"Wrote {len(items)} forced-non-default couplets to {output}", file=sys.stderr)
        print(f"Per-family: {dict(sorted(fam_counts.items()))}", file=sys.stderr)
        return

    if args.contrastive:
        output = args.output or Path(CONTRASTIVE_OUTPUT)
        items = build_contrastive()
        write_items(items, output)
        print(f"Wrote {len(items)} contrastive pairs to {output}", file=sys.stderr)
        for it in items:
            print(f"  {it['device']:<10} clean={it['goal_clean']:<6} corrupt={it['goal_corrupt']}", file=sys.stderr)
        return

    if args.controlled:
        output = args.output or Path(CONTROLLED_OUTPUT)
        items = build_controlled()
        write_items(items, output)
        order_counts = Counter(i["order"] for i in items)
        tok_counts = Counter((i["order"], i["correct"]) for i in items)
        print(f"Wrote {len(items)} controlled on_off items to {output}", file=sys.stderr)
        print(f"Per-order: {dict(sorted(order_counts.items()))}", file=sys.stderr)
        summary = ", ".join(f"{o}:{t}={n}" for (o, t), n in sorted(tok_counts.items()))
        print(f"Per-(order,token): {summary}", file=sys.stderr)
        return

    if args.num_instances < 1:
        parser.error("--num-instances must be >= 1")

    rng = random.Random(args.seed)
    by_group = build_all()
    items = balanced_sample(by_group, args.num_instances, rng)

    output = args.output or Path(DEFAULT_OUTPUT)
    write_items(items, output)

    fam_counts = Counter(i["family"] for i in items)
    tok_counts = Counter((i["family"], i["correct"]) for i in items)
    print(f"Wrote {len(items)} items (seed {args.seed}) to {output}", file=sys.stderr)
    print(f"Per-family: {dict(sorted(fam_counts.items()))}", file=sys.stderr)
    summary = ", ".join(f"{fam}:{tok}={n}" for (fam, tok), n in sorted(tok_counts.items()))
    print(f"Per-(family,token): {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
