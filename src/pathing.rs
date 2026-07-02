//! The sim's movement-planning bridge (P2.M-bridge): turns a tactical movement **goal**
//! ([`CombatIntent::MoveTo`] / [`CombatIntent::Flee`]) into the next-step [`Direction`] by routing
//! through **rover** — a [`CombatWorld`]-backed [`CostMatrixDataSource`] feeds rover's cost-matrix
//! builder, and rover's headless [`LocalPathfinder`] does the multi-step, room-aware search. The
//! caller hands the resulting `Direction` to the engine's `resolve_moves` (the authoritative
//! "server"), so live and sim plan paths through the same system and the engine validates the move
//! (ADR 0006 §B.2). Real pathfinding, not a greedy stepper: a kiter routes *around* obstacles.

use screeps::local::LocalCostMatrix;
use screeps::{Direction, Position, RoomName};
use screeps_combat_decision::kite::{KiteThreat, KiteTower, ThreatField, ThreatKind};
use screeps_combat_decision::CombatIntent;
use screeps_combat_engine::{CombatWorld, CreepId, PlayerId, SimBodyCombat};
use screeps_rover::{
    ConstructionSiteCostMatrixCache, CostMatrixCache, CostMatrixDataSource, CostMatrixOptions,
    CostMatrixSystem, CostMatrixWrite, CreepCostMatrixCache, LinearCostMatrix, LocalPathfinder,
    MovementPriority, PathfindingProvider, StuctureCostMatrixCache,
};
// The rover DRIVER (resolve_moves_via_system + its request/cache/handle/external types) now lives in
// the kernel (`screeps-sim-core`, ADR 0033 M1); this crate wraps it with the combat cost source below.
use screeps_sim_core::{MoverConfig, SimMoveCache, SimMoveGoal, SimMoveRequest};
use std::collections::{HashMap, HashSet};

/// Search budget — the room is 2500 tiles; this comfortably covers a single-room plan.
const MAX_OPS: u32 = 2000;
/// Swamp tile cost baked into the matrix (matches rover's `CostMatrixOptions::default().swamp_cost`).
const SWAMP_COST: u8 = 10;
/// Plains tile cost (matches rover's `CostMatrixOptions::default().plains_cost`) — the base a threat
/// stamp is added on top of for a non-swamp tile.
const PLAINS_COST: u8 = 2;
/// ADR 0024 threat-weighted path cost (the "safest route" — don't get picked off en route). Scales the
/// threat field's incoming hits/tick into an ADDITIVE per-tile traversal cost: `add = (raw / DIV) cap
/// CAP`, kept small + HARD-CAPPED so a threatened tile is *preferred against* but always cheaply
/// PASSABLE (never a wall) — a fully-threatened approach must stay traversable or the squad can never
/// close. Seed values; the EXP-*/SquadTacticParams sweep is the sanctioned tuner.
const THREAT_PATH_DIV: i32 = 150;
const THREAT_PATH_CAP: i32 = 8;

/// The in-room incoming-hits field from `me_owner`'s hostiles + towers (the same [`ThreatField`] the
/// kite scorer uses) — the source of the threat-weighted path cost. Only `attack_power`/`ranged_power`/
/// tower position feed the stamp, so `kind`/`reach`/`step_ticks` are dummies here.
fn room_threat_field(world: &CombatWorld, room: RoomName, me_owner: PlayerId) -> ThreatField {
    let threats: Vec<KiteThreat> = world
        .movement
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner != me_owner && c.pos.room_name() == room)
        .map(|c| KiteThreat {
            pos: c.pos,
            kind: ThreatKind::MeleeOnly,
            reach: 0,
            step_ticks: None,
            attack_power: c.body.attack_power(),
            ranged_power: c.body.ranged_attack_power(),
        })
        .collect();
    let towers: Vec<KiteTower> = world
        .towers
        .iter()
        .filter(|t| t.is_alive() && t.pos.room_name() == room)
        .map(|t| KiteTower { pos: t.pos })
        .collect();
    ThreatField::build(&threats, &towers)
}

/// Per-tile additive-applied threat cost for `room`: for every tile the field covers (and that isn't a
/// wall), the final matrix value `(base terrain + scaled threat)` capped passable. Returns a SPARSE
/// list (only threatened tiles) — empty when there are no threats, so the matrix is byte-identical to
/// the threat-free build. `walls` are excluded (they stay impassable).
fn threat_cost_tiles(
    tf: &ThreatField,
    room: RoomName,
    swamps: &std::collections::HashSet<(u8, u8)>,
    walls: &std::collections::HashSet<(u8, u8)>,
) -> Vec<(u8, u8, u8)> {
    let mut out = Vec::new();
    for x in 0..50u8 {
        for y in 0..50u8 {
            if walls.contains(&(x, y)) {
                continue;
            }
            let xy = match (
                screeps::RoomCoordinate::new(x),
                screeps::RoomCoordinate::new(y),
            ) {
                (Ok(cx), Ok(cy)) => Position::new(cx, cy, room),
                _ => continue,
            };
            let raw = tf.raw_at(xy);
            if raw <= 0 {
                continue;
            }
            let base = if swamps.contains(&(x, y)) {
                SWAMP_COST as i32
            } else {
                PLAINS_COST as i32
            };
            let add = (raw / THREAT_PATH_DIV).min(THREAT_PATH_CAP);
            out.push((x, y, (base + add).min(254) as u8));
        }
    }
    out
}

/// A [`CostMatrixDataSource`] over a `CombatWorld` snapshot. It owns its data (no borrow of the
/// world), satisfying the `'static` bound `CostMatrixSystem` places on its boxed data source. Every
/// obstacle — walls, structures, towers, and **hostile** creeps — is impassable (255); swamps cost
/// [`SWAMP_COST`]. **Friendly creeps (same owner as the pather) are NOT obstacles** — they are
/// moving with you, and treating a teammate's tile as a wall would stall tight formations (a member
/// could never path into a slot a teammate is vacating). This matches the live bot's default
/// (`friendly_creeps: false`). The pather's own tile being blocked is harmless anyway (the search
/// starts there and never re-enters it).
struct CombatCostSource {
    room: RoomName,
    walls: Vec<(u8, u8)>,
    swamps: Vec<(u8, u8)>,
    blockers: Vec<(u8, u8)>,
    hostiles: Vec<(u8, u8)>,
    /// ADR 0024 threat-weighted path cost: `(x, y, final-cost)` for threatened non-wall tiles (sparse;
    /// empty ⇒ byte-identical to the threat-free matrix).
    threat: Vec<(u8, u8, u8)>,
}

impl CombatCostSource {
    fn from_world(world: &CombatWorld, room: RoomName, me_owner: PlayerId) -> Self {
        // Room-scoped (S3): only obstacles IN `room` populate `room`'s matrix — a structure/creep at
        // (x,y) in a *different* room must not block (x,y) here, and terrain reads the room's own
        // override (`terrain_for`). Without this, the multi-room search saw every room's obstacles
        // overlaid at the same (x,y), so it could never route across a border.
        let mut blockers = Vec::new();
        for s in world
            .structures
            .iter()
            .filter(|s| s.is_alive() && s.pos.room_name() == room)
        {
            blockers.push((s.pos.x().u8(), s.pos.y().u8()));
        }
        for t in world
            .towers
            .iter()
            .filter(|t| t.is_alive() && t.pos.room_name() == room)
        {
            blockers.push((t.pos.x().u8(), t.pos.y().u8()));
        }
        let terrain = world.terrain_for(room);
        let walls: Vec<(u8, u8)> = terrain.walls.iter().copied().collect();
        let swamps: Vec<(u8, u8)> = terrain.swamps.iter().copied().collect();
        let wall_set: std::collections::HashSet<(u8, u8)> =
            walls.iter().chain(&blockers).copied().collect();
        let swamp_set: std::collections::HashSet<(u8, u8)> = swamps.iter().copied().collect();
        let threat = threat_cost_tiles(
            &room_threat_field(world, room, me_owner),
            room,
            &swamp_set,
            &wall_set,
        );
        Self {
            room,
            walls,
            swamps,
            blockers,
            hostiles: world
                .movement
                .creeps
                .iter()
                .filter(|c| c.is_alive() && c.owner != me_owner && c.pos.room_name() == room)
                .map(|c| (c.pos.x().u8(), c.pos.y().u8()))
                .collect(),
            threat,
        }
    }
}

impl CostMatrixDataSource for CombatCostSource {
    fn get_structure_costs(&self, room_name: RoomName) -> Option<StuctureCostMatrixCache> {
        if room_name != self.room {
            return None;
        }
        let mut other = LinearCostMatrix::new();
        // Swamps, then threat (base+scaled, overwrites swamp/plains), then impassables — later `set`s win
        // on a tile (apply order), so walls always beat a threat stamp and stay impassable.
        for &(x, y) in &self.swamps {
            other.set(x, y, SWAMP_COST);
        }
        for &(x, y, cost) in &self.threat {
            other.set(x, y, cost);
        }
        for &(x, y) in self.walls.iter().chain(&self.blockers) {
            other.set(x, y, u8::MAX);
        }
        Some(StuctureCostMatrixCache {
            roads: LinearCostMatrix::new(),
            other,
        })
    }

    fn get_construction_site_costs(
        &self,
        _room: RoomName,
    ) -> Option<ConstructionSiteCostMatrixCache> {
        None
    }

    fn get_creep_costs(&self, room_name: RoomName) -> Option<CreepCostMatrixCache> {
        if room_name != self.room {
            return None;
        }
        let mut hostile_creeps = LinearCostMatrix::new();
        for &(x, y) in &self.hostiles {
            hostile_creeps.set(x, y, u8::MAX);
        }
        Some(CreepCostMatrixCache {
            friendly_creeps: LinearCostMatrix::new(), // friendlies intentionally NOT avoided
            hostile_creeps,
            source_keeper_agro: LinearCostMatrix::new(),
        })
    }
}

/// Build the combat cost matrix for `room` from `me_owner`'s perspective via rover's cost-matrix
/// builder (the same pipeline live uses, with a `CombatWorld` data source). Hostiles + structures +
/// walls block; friendlies do not. Shared by per-creep movement and the squad anchor mover so both
/// path over identical costs. `None` for a room the world doesn't model.
pub fn build_combat_matrix(
    world: &CombatWorld,
    room: RoomName,
    me_owner: PlayerId,
) -> Option<LocalCostMatrix> {
    let mut cache = CostMatrixCache::default();
    let mut system = CostMatrixSystem::new(
        &mut cache,
        Box::new(CombatCostSource::from_world(world, room, me_owner)),
    );
    system
        .build_local_cost_matrix(room, &CostMatrixOptions::default())
        .ok()
}

/// Resolve a movement goal to the next-step [`Direction`] from `from` (owned by `me_owner`), via
/// rover's pathfinder over the `CombatWorld`. Returns `None` for non-movement intents, when already
/// satisfied (empty path), or when no route exists. Combat intents (`Attack`/`Heal`/…) and `Idle`
/// yield `None` here. **`MoveTo` routes directly to a (possibly cross-room) target** — rover's search
/// is multi-room, so no MoveToRoom projection is needed; the engine's edge-exit carries the cross.
pub fn resolve_move_direction(
    world: &CombatWorld,
    from: Position,
    me_owner: PlayerId,
    intent: &CombatIntent,
) -> Option<Direction> {
    let opts = CostMatrixOptions::default();
    let mut room_cb = |r: RoomName| build_combat_matrix(world, r, me_owner);
    let mut pf = LocalPathfinder;

    let result = match intent {
        CombatIntent::MoveTo { target, range } => {
            // The multi-room search routes directly to a (possibly cross-room) target.
            pf.search(
                from,
                *target,
                *range as u32,
                &mut room_cb,
                MAX_OPS,
                opts.plains_cost,
                opts.swamp_cost,
            )
        }
        CombatIntent::Flee {
            from: threats,
            range,
        } => {
            let goals: Vec<(Position, u32)> = threats.iter().map(|p| (*p, *range as u32)).collect();
            pf.search_many(
                from,
                &goals,
                true,
                &mut room_cb,
                MAX_OPS,
                opts.plains_cost,
                opts.swamp_cost,
            )
        }
        _ => return None,
    };

    result
        .path
        .first()
        .and_then(|next| from.get_direction_to(*next))
}

// ── Unified mover: route through rover's MovementSystem + resolver (P-MOVE+ / task #30) ──────────
//
// The live bot moves creeps through rover's `MovementSystem` (cached paths, multi-room `find_route`,
// and the `resolver`'s traffic management — swaps / shoves / local-avoidance / stuck-escalation),
// then the game server applies the moves. The sim mirrors this: `resolve_moves_via_system` runs the
// SAME `MovementSystem` over a `CombatWorld` and hands the resolved directions to `resolve_tick` (the
// authoritative "server"). This is the unified replacement for the per-creep `resolve_move_direction`
// shim, so sim ≡ live. The per-creep `CreepMovementData` cache is the CALLER's (held across ticks) so
// path reuse + the stuck-escalation (avoid-friendlies → shove) actually accumulate.

/// Build a [`SimMoveRequest`] from a movement [`CombatIntent`] (`MoveTo` / `Flee`); `None` for a
/// non-movement intent (`Attack`/`Heal`/`Idle`/…) — so a caller can drive the mover straight from the
/// decision. Lives here (not on the kernel's `SimMoveRequest`, which is combat-agnostic) because it
/// depends on the combat-decision `CombatIntent`.
pub fn move_request_from_intent(creep: CreepId, intent: &CombatIntent) -> Option<SimMoveRequest> {
    match intent {
        CombatIntent::MoveTo { target, range } => {
            Some(SimMoveRequest::move_to(creep, *target, *range as u32))
        }
        CombatIntent::Flee { from, range } => Some(SimMoveRequest {
            creep,
            goal: SimMoveGoal::Flee {
                threats: from.clone(),
                range: *range as u32,
            },
            priority: MovementPriority::Normal,
            // Kernel defaults (byte-identical): no numeric-priority bid, run-config ladder.
            priority_value: None,
            shove: true,
            anchor: None,
            stuck_thresholds: None,
        }),
        _ => None,
    }
}

#[derive(Default)]
struct RoomObstacles {
    walls: Vec<(u8, u8)>,
    swamps: Vec<(u8, u8)>,
    blockers: Vec<(u8, u8)>,
    hostiles: Vec<(u8, u8)>,
    /// ADR 0024 threat-weighted path cost per tile (sparse; empty ⇒ byte-identical to threat-free).
    threat: Vec<(u8, u8, u8)>,
}

/// Multi-room [`CostMatrixDataSource`] over a whole `CombatWorld` snapshot (vs the single-room
/// `CombatCostSource` the per-creep path uses) — so the `MovementSystem` can build a cost matrix for
/// ANY room it routes through. Owns its data (`'static`): per room, walls + swamps (terrain),
/// structures/towers + hostile creeps (impassable); friendlies are not avoided. Rooms with no content
/// are all-plain (passable).
struct CombatWorldCostSource {
    rooms: HashMap<RoomName, RoomObstacles>,
}

impl CombatWorldCostSource {
    fn from_world(world: &CombatWorld, me_owner: PlayerId) -> Self {
        let mut rooms: HashMap<RoomName, RoomObstacles> = HashMap::new();
        for s in world.structures.iter().filter(|s| s.is_alive()) {
            rooms
                .entry(s.pos.room_name())
                .or_default()
                .blockers
                .push((s.pos.x().u8(), s.pos.y().u8()));
        }
        for t in world.towers.iter().filter(|t| t.is_alive()) {
            rooms
                .entry(t.pos.room_name())
                .or_default()
                .blockers
                .push((t.pos.x().u8(), t.pos.y().u8()));
        }
        for c in world
            .movement
            .creeps
            .iter()
            .filter(|c| c.is_alive() && c.owner != me_owner)
        {
            rooms
                .entry(c.pos.room_name())
                .or_default()
                .hostiles
                .push((c.pos.x().u8(), c.pos.y().u8()));
        }
        // Terrain for every room that matters — any structure/tower/hostile room (above), any explicit
        // per-room override, AND every (friendly or hostile) creep's room: a room with only friendly
        // creeps + walls must still carry its terrain, or the movers would path straight through it.
        let names: Vec<RoomName> = rooms
            .keys()
            .copied()
            .chain(world.movement.rooms.keys().copied())
            .chain(
                world
                    .movement
                    .creeps
                    .iter()
                    .filter(|c| c.is_alive())
                    .map(|c| c.pos.room_name()),
            )
            .collect();
        for name in names {
            let terrain = world.terrain_for(name);
            let entry = rooms.entry(name).or_default();
            entry.walls.extend(terrain.walls.iter().copied());
            entry.swamps.extend(terrain.swamps.iter().copied());
        }
        // ADR 0024 threat-weighted path cost, per room (from that room's hostiles + towers).
        let room_names: Vec<RoomName> = rooms.keys().copied().collect();
        for name in room_names {
            let (swamp_set, wall_set) = {
                let o = rooms.get(&name).expect("just iterated keys");
                let s: HashSet<(u8, u8)> = o.swamps.iter().copied().collect();
                let w: HashSet<(u8, u8)> = o.walls.iter().chain(&o.blockers).copied().collect();
                (s, w)
            };
            let threat = threat_cost_tiles(
                &room_threat_field(world, name, me_owner),
                name,
                &swamp_set,
                &wall_set,
            );
            rooms.get_mut(&name).expect("just iterated keys").threat = threat;
        }
        Self { rooms }
    }
}

impl CostMatrixDataSource for CombatWorldCostSource {
    fn get_structure_costs(&self, room_name: RoomName) -> Option<StuctureCostMatrixCache> {
        let mut other = LinearCostMatrix::new();
        if let Some(o) = self.rooms.get(&room_name) {
            for &(x, y) in &o.swamps {
                other.set(x, y, SWAMP_COST);
            }
            for &(x, y, cost) in &o.threat {
                other.set(x, y, cost);
            }
            for &(x, y) in o.walls.iter().chain(&o.blockers) {
                other.set(x, y, u8::MAX);
            }
        }
        Some(StuctureCostMatrixCache {
            roads: LinearCostMatrix::new(),
            other,
        })
    }

    fn get_construction_site_costs(
        &self,
        _room: RoomName,
    ) -> Option<ConstructionSiteCostMatrixCache> {
        None
    }

    fn get_creep_costs(&self, room_name: RoomName) -> Option<CreepCostMatrixCache> {
        let mut hostile_creeps = LinearCostMatrix::new();
        if let Some(o) = self.rooms.get(&room_name) {
            for &(x, y) in &o.hostiles {
                hostile_creeps.set(x, y, u8::MAX);
            }
        }
        Some(CreepCostMatrixCache {
            friendly_creeps: LinearCostMatrix::new(),
            hostile_creeps,
            source_keeper_agro: LinearCostMatrix::new(),
        })
    }
}

/// The COMBAT-domain mover defaults — the SINGLE combat-config point. Now **identical to the
/// kernel default ([`MoverConfig::default`]), `register_idle_creeps` ON included**: the split
/// default (registration OFF, 2026-07-01) existed because an unrequested COMBAT creep was
/// usually HOLDING — an in-position shooter/tank that decided to attack, not move — and
/// registering holders as shoveable Low idles let squadmates displace them out of formation and
/// dance-avoid their held tiles (measured: drain-soak `RosterWiped`, >10% period-2 oscillation).
/// That drain-wipe mechanism is CLOSED by holding-as-a-request (ADR 0033 end-state item (1),
/// `squad::push_hold_requests`): every living squad member now claims its held tile explicitly
/// (`move_to` own tile, range 0, `Immovable` — never displaceable, enum-checked in the
/// resolver), so no unrequested creep exists INSIDE a squad and registration only catches
/// genuinely ownerless idlers — same-owner creeps outside any squad's request set — which are
/// parked junk in combat exactly as in the economy domain. Kept as a fn (not an alias) so combat
/// keeps ONE opt-out point if the domains ever diverge again.
pub fn combat_mover_config() -> MoverConfig {
    MoverConfig::default()
}

/// Run rover's `MovementSystem` (resolver included) over `world` for `owner`'s `requests`, returning
/// the resolved per-creep directions to hand to `resolve_tick`. This is the **combat wrapper** around
/// the kernel driver [`screeps_sim_core::resolve_moves_via_system`] (ADR 0033 M1): it injects the
/// combat routing policy — [`CombatWorldCostSource`] (towers / ADR-0024 threat field / structures as
/// obstacles) — and forwards the pure `MovementState`. `cache` persists per-creep path reuse +
/// stuck-escalation across ticks. Traffic-managed, unified analogue of routing each creep alone.
/// Defaults to [`combat_mover_config`] (currently ≡ the kernel default — holding-as-a-request
/// closed the registration split; the fn stays the single combat-config point).
pub fn resolve_moves_via_system(
    world: &CombatWorld,
    owner: PlayerId,
    requests: &[SimMoveRequest],
    cache: &mut SimMoveCache,
) -> HashMap<CreepId, Direction> {
    resolve_moves_via_system_with(world, owner, requests, cache, &combat_mover_config())
}

/// [`resolve_moves_via_system`] with explicit rover tunables — the combat analogue of the kernel's
/// [`screeps_sim_core::resolve_moves_via_system_with`] (ADR 0033 M5), so a COMBAT-corpus parameter
/// tournament can sweep one [`MoverConfig`] per evaluated point over real combat scenarios (the
/// rover-eval `tuning.rs` idiom, adjudicating haul-tuned candidates like `ladder(8)` on combat
/// outcomes). Same cost source, same `MovementState`; `MoverConfig::default()` ⇒ byte-identical to
/// the plain fn (`Default` mirrors the live tuning exactly, rover_driver.rs).
pub fn resolve_moves_via_system_with(
    world: &CombatWorld,
    owner: PlayerId,
    requests: &[SimMoveRequest],
    cache: &mut SimMoveCache,
    config: &MoverConfig,
) -> HashMap<CreepId, Direction> {
    screeps_sim_core::resolve_moves_via_system_with(
        &world.movement,
        requests,
        cache,
        CombatWorldCostSource::from_world(world, owner),
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Part, RoomCoordinate};
    use screeps_combat_engine::{
        resolve_tick, CombatWorld, Intents, MovementState, SimBody, SimCreep,
    };

    fn room() -> RoomName {
        "W1N1".parse().unwrap()
    }
    fn pos(x: u8, y: u8) -> Position {
        Position::new(
            RoomCoordinate::new(x).unwrap(),
            RoomCoordinate::new(y).unwrap(),
            room(),
        )
    }
    /// The room one step east of W1N1, derived via `checked_add` (no hardcoded W0/W2).
    fn east_room() -> RoomName {
        pos(49, 25).checked_add((1, 0)).unwrap().room_name()
    }
    fn pos_in(r: RoomName, x: u8, y: u8) -> Position {
        Position::new(
            RoomCoordinate::new(x).unwrap(),
            RoomCoordinate::new(y).unwrap(),
            r,
        )
    }
    fn creep(id: u32, x: u8, y: u8) -> SimCreep {
        SimCreep {
            id,
            owner: 0,
            pos: pos(x, y),
            body: SimBody::unboosted(&[Part::Move, Part::Attack]),
            fatigue: 0,
            carry_used: 0,
        }
    }

    #[test]
    fn moves_toward_an_open_goal() {
        let world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 5, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        let dir = resolve_move_direction(
            &world,
            pos(5, 25),
            0,
            &CombatIntent::MoveTo {
                target: pos(15, 25),
                range: 0,
            },
        );
        // 8-directional + uniform cost ⇒ Right / TopRight / BottomRight are all equally-optimal
        // first steps toward an eastern goal; assert we head east, not the exact diagonal.
        assert!(
            matches!(
                dir,
                Some(Direction::Right | Direction::TopRight | Direction::BottomRight)
            ),
            "open room → step east toward the goal, got {:?}",
            dir
        );
    }

    #[test]
    fn threat_field_raises_cost_near_a_tower_but_stays_passable() {
        // ADR 0024 Stage 1: a hostile tower stamps an ADDITIVE per-tile threat cost around it — higher
        // than plains so the rover routes around exposure, but capped well below impassable so a
        // threatened approach is always traversable. The tower's own tile stays a hard blocker.
        use screeps_combat_engine::SimTower;
        let world = CombatWorld {
            towers: vec![SimTower {
                id: 1,
                owner: 1,
                pos: pos(10, 10),
                energy: 1000,
                hits: 3000,
                hits_max: 3000,
            }],
            ..Default::default()
        };
        let m = build_combat_matrix(&world, room(), 0).expect("matrix");
        let near = m.get(pos(10, 12).xy()); // range 2 of the tower (full ~600 damage)
        let far = m.get(pos(45, 45).xy()); // range 35 (a tower hits the WHOLE room at its ~150 floor)
        assert!(
            near > far,
            "closer to the tower costs more (the routing gradient): near={near} far={far}"
        );
        assert!(
            near < u8::MAX && far < u8::MAX,
            "threatened tiles stay PASSABLE (never a wall): near={near} far={far}"
        );
        assert!(
            far > PLAINS_COST,
            "a tower threatens the whole room (min damage everywhere): far={far}"
        );
        assert_eq!(
            m.get(pos(10, 10).xy()),
            u8::MAX,
            "the tower's own tile stays impassable"
        );
    }

    #[test]
    fn no_threats_matrix_is_byte_identical_plains() {
        // No creeps/towers → empty threat field → plains tiles stay 0 (byte-identical to pre-Stage-1).
        let world = CombatWorld::default();
        let m = build_combat_matrix(&world, room(), 0).expect("matrix");
        assert_eq!(
            m.get(pos(25, 25).xy()),
            0,
            "plains tile unchanged with no threats"
        );
        assert_eq!(m.get(pos(5, 5).xy()), 0);
    }

    #[test]
    fn detours_around_a_wall() {
        // Wall column at x=6, y=23..=27, goal directly east behind it. The first step must not be
        // straight into the wall at (6,25) — it routes around (a diagonal toward a gap).
        let mut world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 5, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        for y in 23..=27 {
            world.movement.terrain.walls.insert((6, y));
        }
        let dir = resolve_move_direction(
            &world,
            pos(5, 25),
            0,
            &CombatIntent::MoveTo {
                target: pos(10, 25),
                range: 0,
            },
        )
        .expect("a route around exists");
        // Stepping Right would enter the wall at (6,25); the planner must pick a detour.
        assert_ne!(dir, Direction::Right, "does not walk into the wall");
        assert!(
            matches!(
                dir,
                Direction::TopRight | Direction::BottomRight | Direction::Top | Direction::Bottom
            ),
            "heads around the wall, got {:?}",
            dir
        );
    }

    #[test]
    fn already_in_range_yields_no_move() {
        let world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 5, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        let dir = resolve_move_direction(
            &world,
            pos(5, 25),
            0,
            &CombatIntent::MoveTo {
                target: pos(7, 25),
                range: 3,
            },
        );
        assert_eq!(dir, None, "distance 2 already within range 3 → hold");
    }

    #[test]
    fn flees_away_from_a_threat() {
        let world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 30, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        let dir = resolve_move_direction(
            &world,
            pos(30, 25),
            0,
            &CombatIntent::Flee {
                from: vec![pos(25, 25)],
                range: 5,
            },
        )
        .expect("can flee in an open room");
        // Threat is to the west (x=25); fleeing should move east (away), increasing x.
        assert!(
            matches!(
                dir,
                Direction::Right | Direction::TopRight | Direction::BottomRight
            ),
            "flees away from the threat (eastward), got {:?}",
            dir
        );
    }

    #[test]
    fn non_movement_intent_is_none() {
        let world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 5, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            resolve_move_direction(&world, pos(5, 25), 0, &CombatIntent::Idle),
            None
        );
    }

    // ── Cross-room direction production (ADR 0023 S3) ─────────────────────────────────────────────

    #[test]
    fn aims_toward_the_exit_for_a_cross_room_target() {
        // A target in the adjacent (east) room → the next step heads east toward the exit. The rover's
        // multi-room search routes across once the cost source is room-scoped (S3); previously every
        // room's obstacles overlaid at the same (x,y) so it could not route across a border.
        let east = pos_in(east_room(), 25, 25);
        let world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 25, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        let dir = resolve_move_direction(
            &world,
            pos(25, 25),
            0,
            &CombatIntent::MoveTo {
                target: east,
                range: 0,
            },
        )
        .expect("a cross-room route exists");
        assert!(
            matches!(
                dir,
                Direction::Right | Direction::TopRight | Direction::BottomRight
            ),
            "heads east toward the exit, got {:?}",
            dir
        );
    }

    #[test]
    fn paths_a_creep_across_a_room_boundary() {
        // End-to-end: resolve_move_direction picks each next step toward a target in the EAST room;
        // resolve_tick applies the move AND the engine's edge-exit relocation. The creep crosses the
        // border and arrives — the full S3 cross-room movement path, validated against the engine.
        let target = pos_in(east_room(), 5, 25);
        let mut world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 45, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut reached = false;
        for _ in 0..40 {
            let from = world.movement.creeps[0].pos;
            if from == target {
                reached = true;
                break;
            }
            let mut i = Intents::new();
            if let Some(dir) =
                resolve_move_direction(&world, from, 0, &CombatIntent::MoveTo { target, range: 0 })
            {
                i.set_move(1, dir);
            }
            resolve_tick(&mut world, &i);
        }
        assert!(
            reached,
            "the creep pathed across the border into the east room and reached the target"
        );
    }

    #[test]
    fn system_mover_routes_a_creep_across_a_room_boundary() {
        // The UNIFIED mover (rover MovementSystem + resolver) routes a creep across a border to a
        // target — the sim now runs the SAME traffic-managed mover as live, not just the per-creep
        // shim. The CreepMovementData cache persists across ticks (path reuse + stuck-escalation).
        let target = pos_in(east_room(), 5, 25);
        let mut world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 45, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut cache = SimMoveCache::new();
        let mut reached = false;
        for _ in 0..60 {
            let from = world.movement.creeps[0].pos;
            if from == target {
                reached = true;
                break;
            }
            let reqs = [SimMoveRequest::move_to(1, target, 0)];
            let moves = resolve_moves_via_system(&world, 0, &reqs, &mut cache);
            let mut i = Intents::new();
            if let Some(&dir) = moves.get(&1) {
                i.set_move(1, dir);
            }
            resolve_tick(&mut world, &i);
        }
        assert!(
            reached,
            "the MovementSystem-routed creep crossed the border and reached the target"
        );
    }

    #[test]
    fn system_mover_deconflicts_two_creeps() {
        // Two creeps moving head-on toward each other's tile in the same row. The unified mover runs
        // BOTH through one resolver pass (the traffic manager), so they pass each other and both reach
        // their targets — no deadlock. (Exercises the multi-creep resolver path the per-creep shim
        // could not coordinate.)
        let mut world = CombatWorld {
            movement: MovementState {
                creeps: vec![creep(1, 10, 25), creep(2, 16, 25)],
                ..Default::default()
            },
            ..Default::default()
        };
        let (ta, tb) = (pos(16, 25), pos(10, 25));
        let mut cache = SimMoveCache::new();
        let mut both_arrived = false;
        for _ in 0..30 {
            let a = world
                .movement
                .creeps
                .iter()
                .find(|c| c.id == 1)
                .map(|c| c.pos);
            let b = world
                .movement
                .creeps
                .iter()
                .find(|c| c.id == 2)
                .map(|c| c.pos);
            if a == Some(ta) && b == Some(tb) {
                both_arrived = true;
                break;
            }
            let reqs = [
                SimMoveRequest::move_to(1, ta, 0),
                SimMoveRequest::move_to(2, tb, 0),
            ];
            let moves = resolve_moves_via_system(&world, 0, &reqs, &mut cache);
            let mut i = Intents::new();
            for (&id, &dir) in &moves {
                i.set_move(id, dir);
            }
            resolve_tick(&mut world, &i);
        }
        assert!(
            both_arrived,
            "the resolver let both creeps pass each other and reach their targets"
        );
    }
}
