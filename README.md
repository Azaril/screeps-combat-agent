# screeps-combat-agent

> The sim side of the tactical seam — run the bot's *real* combat brain over an engine world, with no tactics fork.

`screeps-combat-agent` is the third layer of the Screeps combat family. It bridges the two halves of a headless self-play harness: a `SimView` builds the JS-free [`CombatView`](https://github.com/Azaril/screeps-combat-decision) (from [`screeps-combat-decision`](https://github.com/Azaril/screeps-combat-decision)) out of a [`CombatWorld`](https://github.com/Azaril/screeps-combat-engine) (from [`screeps-combat-engine`](https://github.com/Azaril/screeps-combat-engine)), and `IbexAgent` runs the bot's real decision code (`decide_combat` + `decide_movement`) over that view. Movement goals are planned through [`screeps-rover`](https://github.com/Azaril/screeps-rover)'s headless pathfinder, so sim and live route over the same system.

The point is *one implementation*: because the agent calls the bot's actual tactics rather than a sim-only copy, self-play is `IbexAgent` vs `IbexAgent` (or vs a scripted opponent) with nothing to drift or overfit. It is a component extracted from the [screeps-ibex](https://github.com/Azaril/screeps-ibex) workspace.

## Installation

This is a library. Add it as a git dependency:

```toml
[dependencies]
screeps-combat-agent = { git = "https://github.com/Azaril/screeps-combat-agent" }
```

Beyond the shared `screeps-game-api` base types, it depends only on the other pure value-type combat crates — `screeps-combat-engine`, `screeps-combat-decision`, and `screeps-rover` (the latter without its `screeps` feature, i.e. the headless core) — not the bot, so it builds host-side with no game runtime.

## Usage

The core loop: build a `SimView` from a `CombatWorld` for one side, run a `TacticalAgent` over it to produce engine `Intents`, and hand those to the engine's `resolve_tick` (the authoritative "server").

```rust
use screeps_combat_agent::{agent_intents, IbexAgent, SimView};
use screeps_combat_engine::{resolve_tick, CombatWorld, PlayerId};

fn run(mut world: CombatWorld, me_owner: PlayerId, center: screeps::Position, room: screeps::RoomName) {
    let mut agent = IbexAgent;
    for _ in 0..30 {
        // Build this side's view of the world, run the bot's real brain over each creep,
        // and translate the emitted intents into engine actions + pathfound moves.
        let sim = SimView::from_world(&world, me_owner, center, room);
        let intents = agent_intents(&world, &sim, &mut agent);
        // The engine resolves the tick (combat, movement, deaths) — the "server".
        resolve_tick(&mut world, &intents);
    }
}
```

Key pieces:

- **`SimView::from_world(world, me_owner, center, room)`** — builds one side's DTO view. Living creeps owned by `me_owner` are `friends`, all others are `hostiles`; structures and towers are classified mine / hostile / neutral. Sim creeps have no game `ObjectId`, so the view mints a stable synthetic `RawObjectId` per creep and keeps the reverse map; `creep_for` / `structure_for` resolve an emitted intent's target back to an engine id.
- **`IbexAgent`** — a `TacticalAgent` wrapping the bot's real per-tick decision: `decide_combat` (attack + heal) plus `decide_movement` (kite / engage / flee / heal-follow).
- **`agent_intents(world, sim, agent)`** — runs `agent` over each friendly creep and assembles the engine `Intents`: combat intents become engine `CombatAction`s (`to_engine_action`), movement intents become a step `Direction` planned through rover (`pathing::resolve_move_direction`).
- **`to_engine_action(intent, view)`** — translates a single `CombatIntent` into a `CombatAction`. Creep-targeted intents resolve by synthetic id; structure-targeted intents (addressed by position) resolve by position, with the *shield* (rampart, then wall) winning on a shared tile.
- **`HoldAgent`** — a trivial scripted agent that always idles, proving the `TacticalAgent` seam is swappable.

### Squads

Two squad drivers exercise squad-level movement and cohesion (`module squad`):

- **`SimSquad`** — an anchor mover (`rover::AnchorPath`) plus ordered members holding a formation layout (`anchor + offset`). `step(world)` measures cohesion, advances the anchor toward the objective only when a quorum of members is in formation, routes the squad's bounding-box footprint around walls, relaxes to single-file in a 1-wide corridor, and re-forms the box on open terrain. Returns `(Intents, AnchorOutcome)` — `AnchorOutcome::Blocked` surfaces a path failure. A blob (N > 4) advances loosely on centroid proximity.
- **`ManagedSimSquad`** — the *manager-fielded*, anchorless path that mirrors the live `SquadManager`. `step(world)` builds a `SquadView`, runs the pure `decide_squad_with_pathing` (shared focus + heal assignment + the cohesive pathfinding-scored kite goal), then per-creep `decide_combat` + `decide_movement` with that shared directive. `ManagedSimSquad::new(owner, members, objective)`; `with_tactics(SquadTacticParams)` overrides the position-scoring weights for empirical tuning.

```rust
use screeps_combat_agent::squad::ManagedSimSquad;
use screeps_combat_engine::resolve_tick;

let mut squad = ManagedSimSquad::new(0, vec![1, 2], objective);
for _ in 0..50 {
    let intents = squad.step(&world);
    resolve_tick(&mut world, &intents);
}
```

### Scripted opponents and engagements

`module opponents` supplies fixed, deterministic adversaries to validate `IbexAgent` against, plus a head-to-head runner:

- **Scripted `TacticalAgent`s** — `RushAgent` (melee bruiser, closes and attacks), `KiteAgent` (ranged skirmisher, holds range 3), `TurtleAgent` (stand-and-heal, focus-fire must out-DPS), `DrainAgent` (pure tower-bait tank that bleeds a tower's energy).
- **`run_engagement(world, room, a_owner, a_center, agent_a, b_owner, b_center, agent_b, max_ticks)`** — runs two agents head-to-head through the engine until one side is fully gone (no creeps, towers, or owned structures) or `max_ticks` elapses, with each side's towers fired by the scripted `tower_intents`. Returns an `EngagementOutcome` (ticks, per-side alive counts, per-side worst cohesion, per-side tower energy, and a `CombatRecording`).
- **`self_play(...)`** — convenience wrapper that runs `IbexAgent` on both sides.
- **`Unit` / `world_from_units(a_owner, a_units, b_owner, b_units)`** — compose a two-sided `CombatWorld` from `(body, positions)` specs with auto-numbered creep ids.

### Scenarios

`module scenario` is a fluent `ScenarioBuilder` over a single-room `CombatWorld` for building richer rooms than the trivial open-field case — terrain, passive structures, and firing towers:

```rust
use screeps_combat_agent::scenario::ScenarioBuilder;

let world = ScenarioBuilder::empty(room)
    .perimeter(1, 20, 20, 30, 30, 300_000) // constructed-wall ring
    .spawn(1, 25, 25)
    .rampart(1, 25, 24, 1_000_000)
    .tower_nest(1, 26, 26, 3, 1000)
    .build();
```

Chaining helpers include `wall` / `wall_column` / `wall_row` / `swamp_rect`, `cwall` / `rampart` / `spawn` / `perimeter`, `tower` / `tower_nest`, and `safe_mode`. `from_units(...)` seeds creeps first; `world_mut()` is an escape hatch; `build()` returns the `CombatWorld`. All synthesized coordinates are clamped to `0..=49`, and structure ids start at `1_000_000` so they never collide with creep ids.

### Replay

`module replay` renders a `CombatRecording` (captured by `record_tick` during an engagement) as an SVG filmstrip — one mini-map per tick, scrubbed left to right:

```rust
use screeps_combat_agent::replay;

let svg = replay::to_svg(&outcome.recording);
std::fs::write("engagement.svg", svg).unwrap();
```

Creeps render as owner-coloured circles whose radius tracks HP fraction; structures render as grey tiles. (Towers aren't in the frame model, so they don't draw.)

## Example

`examples/replay_demo.rs` runs `IbexAgent` against the scripted opponent roster across four adversarial scenarios (kiter vs rush, focus-fire vs turtle, quad vs strong turtle, drain vs tower), captures each recording, and writes SVG filmstrips to `target/replays/`:

```bash
cargo run --example replay_demo -p screeps-combat-agent
```

It writes both a full filmstrip and a compact (≤ 6-frame) version per scenario, and prints a one-line summary (ticks, survivors, cohesion, tower energy) for each.

## How it works

A `CombatWorld` is the engine's ground-truth model of a room (creeps, structures, towers, terrain). `SimView` projects it into the decision layer's read-only DTOs — minting a synthetic `RawObjectId` per creep, computing the shared squad focus once (`select_focus_target`), and building a position → engine-`StructureId` map so a by-position structure intent can be routed back to a concrete wall / rampart / tower / spawn. On a shared tile the shield wins, so a breach must break the rampart before the structure it covers (the engine applies single-target structure damage with no auto-redirect).

`IbexAgent` then emits `CombatIntent`s, which `agent_intents` splits two ways: combat intents become `CombatAction`s via `to_engine_action`, and movement goals (`MoveTo` / `Flee`) are pathfound to a single-step `Direction` through rover's headless `LocalPathfinder`, backed by a `CostMatrixDataSource` over the world (walls / structures / towers / hostile creeps block; friendly creeps do not, matching the live bot). The resulting `Intents` go to the engine's `resolve_tick`, which validates and applies them. Because the same decision code runs in the sim and live, the harness can run self-play and adversarial scenarios with one implementation and no fork.

## Related crates

- [screeps-combat-engine](https://github.com/Azaril/screeps-combat-engine) — the combat mechanism: the `CombatWorld` model and the authoritative `resolve_tick` resolver.
- [screeps-combat-decision](https://github.com/Azaril/screeps-combat-decision) — the tactics: the pure `decide_combat` / `decide_movement` / `decide_squad_with_pathing` brain over a `CombatView`.
- [screeps-combat-eval](https://github.com/Azaril/screeps-combat-eval) — the policy / scoring layer: aggregated metrics, stalemate adjudication, and the richer replay scrubber built on top of this crate's engagements.
