//! The sim's movement-planning bridge (P2.M-bridge): turns a tactical movement **goal**
//! ([`CombatIntent::MoveTo`] / [`CombatIntent::Flee`]) into the next-step [`Direction`] by routing
//! through **rover** — a [`CombatWorld`]-backed [`CostMatrixDataSource`] feeds rover's cost-matrix
//! builder, and rover's headless [`LocalPathfinder`] does the multi-step, room-aware search. The
//! caller hands the resulting `Direction` to the engine's `resolve_moves` (the authoritative
//! "server"), so live and sim plan paths through the same system and the engine validates the move
//! (ADR 0006 §B.2). Real pathfinding, not a greedy stepper: a kiter routes *around* obstacles.

use screeps::local::LocalCostMatrix;
use screeps::{Direction, Position, RoomName};
use screeps_combat_decision::CombatIntent;
use screeps_combat_engine::{CombatWorld, CreepId, PlayerId};
use screeps_rover::traits::CreepHandle;
use screeps_rover::{
    ConstructionSiteCostMatrixCache, CostMatrixCache, CostMatrixDataSource, CostMatrixOptions, CostMatrixSystem,
    CostMatrixWrite, CreepCostMatrixCache, CreepMovementData, FleeTarget, LinearCostMatrix, LocalPathfinder,
    MovementData, MovementError, MovementPriority, MovementSystem, MovementSystemExternal, PathfindingProvider,
    StuctureCostMatrixCache,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Search budget — the room is 2500 tiles; this comfortably covers a single-room plan.
const MAX_OPS: u32 = 2000;
/// Swamp tile cost baked into the matrix (matches rover's `CostMatrixOptions::default().swamp_cost`).
const SWAMP_COST: u8 = 10;

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
}

impl CombatCostSource {
    fn from_world(world: &CombatWorld, room: RoomName, me_owner: PlayerId) -> Self {
        // Room-scoped (S3): only obstacles IN `room` populate `room`'s matrix — a structure/creep at
        // (x,y) in a *different* room must not block (x,y) here, and terrain reads the room's own
        // override (`terrain_for`). Without this, the multi-room search saw every room's obstacles
        // overlaid at the same (x,y), so it could never route across a border.
        let mut blockers = Vec::new();
        for s in world.structures.iter().filter(|s| s.is_alive() && s.pos.room_name() == room) {
            blockers.push((s.pos.x().u8(), s.pos.y().u8()));
        }
        for t in world.towers.iter().filter(|t| t.is_alive() && t.pos.room_name() == room) {
            blockers.push((t.pos.x().u8(), t.pos.y().u8()));
        }
        let terrain = world.terrain_for(room);
        Self {
            room,
            walls: terrain.walls.iter().copied().collect(),
            swamps: terrain.swamps.iter().copied().collect(),
            blockers,
            hostiles: world
                .creeps
                .iter()
                .filter(|c| c.is_alive() && c.owner != me_owner && c.pos.room_name() == room)
                .map(|c| (c.pos.x().u8(), c.pos.y().u8()))
                .collect(),
        }
    }
}

impl CostMatrixDataSource for CombatCostSource {
    fn get_structure_costs(&self, room_name: RoomName) -> Option<StuctureCostMatrixCache> {
        if room_name != self.room {
            return None;
        }
        let mut other = LinearCostMatrix::new();
        // Swamps first, then impassables — later `set`s win on a tile (apply order = push order).
        for &(x, y) in &self.swamps {
            other.set(x, y, SWAMP_COST);
        }
        for &(x, y) in self.walls.iter().chain(&self.blockers) {
            other.set(x, y, u8::MAX);
        }
        Some(StuctureCostMatrixCache { roads: LinearCostMatrix::new(), other })
    }

    fn get_construction_site_costs(&self, _room: RoomName) -> Option<ConstructionSiteCostMatrixCache> {
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
pub fn build_combat_matrix(world: &CombatWorld, room: RoomName, me_owner: PlayerId) -> Option<LocalCostMatrix> {
    let mut cache = CostMatrixCache::default();
    let mut system = CostMatrixSystem::new(&mut cache, Box::new(CombatCostSource::from_world(world, room, me_owner)));
    system.build_local_cost_matrix(room, &CostMatrixOptions::default()).ok()
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
            pf.search(from, *target, *range as u32, &mut room_cb, MAX_OPS, opts.plains_cost, opts.swamp_cost)
        }
        CombatIntent::Flee { from: threats, range } => {
            let goals: Vec<(Position, u32)> = threats.iter().map(|p| (*p, *range as u32)).collect();
            pf.search_many(from, &goals, true, &mut room_cb, MAX_OPS, opts.plains_cost, opts.swamp_cost)
        }
        _ => return None,
    };

    result.path.first().and_then(|next| from.get_direction_to(*next))
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

/// Default shove-chain depth for the sim mover.
const DEFAULT_SHOVE_DEPTH: u32 = 3;

/// Per-creep movement state (cached path + stuck tracking), persisted across ticks by the caller.
pub type SimMoveCache = HashMap<CreepId, CreepMovementData>;

/// A movement goal for [`resolve_moves_via_system`] — mirrors the movement [`CombatIntent`]s.
pub enum SimMoveGoal {
    /// Reach `target` within `range`.
    To { target: Position, range: u32 },
    /// Flee to outside `range` of every threat.
    Flee { threats: Vec<Position>, range: u32 },
}

/// A per-creep movement request for [`resolve_moves_via_system`]. `priority` decides who wins a
/// contested tile (the resolver orders by priority before any tie-break) — e.g. a squad's combat
/// creep takes `High` so it claims the forward kite/shooting spot over a support creep.
pub struct SimMoveRequest {
    pub creep: CreepId,
    pub goal: SimMoveGoal,
    pub priority: MovementPriority,
}

impl SimMoveRequest {
    /// A `move_to` request (default priority): reach `target` within `range`.
    pub fn move_to(creep: CreepId, target: Position, range: u32) -> Self {
        SimMoveRequest { creep, goal: SimMoveGoal::To { target, range }, priority: MovementPriority::Normal }
    }

    /// Build a request from a movement [`CombatIntent`] (`MoveTo` / `Flee`); `None` for non-movement
    /// intents (`Attack`/`Heal`/`Idle`/…) — so a caller can drive the mover straight from the decision.
    pub fn from_intent(creep: CreepId, intent: &CombatIntent) -> Option<Self> {
        match intent {
            CombatIntent::MoveTo { target, range } => {
                Some(SimMoveRequest { creep, goal: SimMoveGoal::To { target: *target, range: *range as u32 }, priority: MovementPriority::Normal })
            }
            CombatIntent::Flee { from, range } => {
                Some(SimMoveRequest { creep, goal: SimMoveGoal::Flee { threats: from.clone(), range: *range as u32 }, priority: MovementPriority::Normal })
            }
            _ => None,
        }
    }

    /// Set the contention priority (e.g. `High` for a combat creep that must win the shooting tile).
    pub fn with_priority(mut self, priority: MovementPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Shared sink the creep handles write their resolved direction into (`move_direction` is `&self`,
/// mirroring the live `creep.move()`, so it needs interior mutability).
type MoveSink = Rc<RefCell<HashMap<CreepId, Direction>>>;

/// A [`CreepHandle`] over a `SimCreep` snapshot; `move_direction` records into the shared sink (the
/// sim's analogue of issuing `creep.move(dir)` to the server).
struct CombatCreepHandle {
    id: CreepId,
    pos: Position,
    fatigue: u32,
    sink: MoveSink,
}

impl CreepHandle for CombatCreepHandle {
    fn pos(&self) -> Position {
        self.pos
    }
    fn fatigue(&self) -> u32 {
        self.fatigue
    }
    fn spawning(&self) -> bool {
        false
    }
    fn move_direction(&self, dir: Direction) -> Result<(), String> {
        self.sink.borrow_mut().insert(self.id, dir);
        Ok(())
    }
    fn pull(&self, _other: &Self) -> Result<(), String> {
        Ok(()) // pull chains: a sim follow-up (the engine supports Intents.pulls); no-op for now.
    }
    fn move_pulled_by(&self, _other: &Self) -> Result<(), String> {
        Ok(())
    }
}

/// `CombatWorld`-backed [`MovementSystemExternal`] — the headless analogue of the live
/// `MovementSystemExternalProvider`. Owns the move sink + borrows the world + the caller's cache.
struct CombatMovementExternal<'w, 'c> {
    world: &'w CombatWorld,
    sink: MoveSink,
    cache: &'c mut SimMoveCache,
}

impl MovementSystemExternal<CreepId> for CombatMovementExternal<'_, '_> {
    type Creep = CombatCreepHandle;

    fn get_creep(&self, entity: CreepId) -> Result<CombatCreepHandle, MovementError> {
        let c = self
            .world
            .creeps
            .iter()
            .find(|c| c.id == entity && c.is_alive())
            .ok_or_else(|| "creep not found".to_owned())?;
        Ok(CombatCreepHandle { id: entity, pos: c.pos, fatigue: c.fatigue, sink: self.sink.clone() })
    }

    fn get_creep_movement_data(&mut self, entity: CreepId) -> Result<&mut CreepMovementData, MovementError> {
        Ok(self.cache.entry(entity).or_default())
    }

    fn get_entity_position(&self, entity: CreepId) -> Option<Position> {
        self.world.creeps.iter().find(|c| c.id == entity && c.is_alive()).map(|c| c.pos)
    }
}

#[derive(Default)]
struct RoomObstacles {
    walls: Vec<(u8, u8)>,
    swamps: Vec<(u8, u8)>,
    blockers: Vec<(u8, u8)>,
    hostiles: Vec<(u8, u8)>,
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
            rooms.entry(s.pos.room_name()).or_default().blockers.push((s.pos.x().u8(), s.pos.y().u8()));
        }
        for t in world.towers.iter().filter(|t| t.is_alive()) {
            rooms.entry(t.pos.room_name()).or_default().blockers.push((t.pos.x().u8(), t.pos.y().u8()));
        }
        for c in world.creeps.iter().filter(|c| c.is_alive() && c.owner != me_owner) {
            rooms.entry(c.pos.room_name()).or_default().hostiles.push((c.pos.x().u8(), c.pos.y().u8()));
        }
        // Terrain for every room that matters — any structure/tower/hostile room (above), any explicit
        // per-room override, AND every (friendly or hostile) creep's room: a room with only friendly
        // creeps + walls must still carry its terrain, or the movers would path straight through it.
        let names: Vec<RoomName> = rooms
            .keys()
            .copied()
            .chain(world.rooms.keys().copied())
            .chain(world.creeps.iter().filter(|c| c.is_alive()).map(|c| c.pos.room_name()))
            .collect();
        for name in names {
            let terrain = world.terrain_for(name);
            let entry = rooms.entry(name).or_default();
            entry.walls.extend(terrain.walls.iter().copied());
            entry.swamps.extend(terrain.swamps.iter().copied());
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
            for &(x, y) in o.walls.iter().chain(&o.blockers) {
                other.set(x, y, u8::MAX);
            }
        }
        Some(StuctureCostMatrixCache { roads: LinearCostMatrix::new(), other })
    }

    fn get_construction_site_costs(&self, _room: RoomName) -> Option<ConstructionSiteCostMatrixCache> {
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

/// Run rover's `MovementSystem` (resolver included) over `world` for `owner`'s `requests`, returning
/// the resolved per-creep directions to hand to `resolve_tick`. `cache` is the caller's persisted
/// per-creep movement state (path reuse + stuck-escalation accumulate across ticks). This is the
/// traffic-managed, unified analogue of calling [`resolve_move_direction`] per creep.
pub fn resolve_moves_via_system(
    world: &CombatWorld,
    owner: PlayerId,
    requests: &[SimMoveRequest],
    cache: &mut SimMoveCache,
) -> HashMap<CreepId, Direction> {
    let sink: MoveSink = Rc::new(RefCell::new(HashMap::new()));
    let mut external = CombatMovementExternal { world, sink: sink.clone(), cache };

    let mut cm_cache = CostMatrixCache::default();
    let mut cms = CostMatrixSystem::new(&mut cm_cache, Box::new(CombatWorldCostSource::from_world(world, owner)));
    let mut pf = LocalPathfinder;
    let mut system = MovementSystem::new(&mut cms, &mut pf, None);
    system.set_max_shove_depth(DEFAULT_SHOVE_DEPTH);

    // The MovementSystem routes to the (possibly cross-room) target directly — the rover search is
    // now multi-room, so no MoveToRoom pre-projection is needed.
    let mut data = MovementData::new();
    for req in requests {
        match &req.goal {
            SimMoveGoal::To { target, range } => {
                data.move_to(req.creep, *target).range(*range).allow_shove(true).allow_swap(true).priority(req.priority);
            }
            SimMoveGoal::Flee { threats, range } => {
                let targets: Vec<FleeTarget> = threats.iter().map(|p| FleeTarget { pos: *p, range: *range }).collect();
                data.flee(req.creep, targets).allow_shove(true).allow_swap(true).priority(req.priority);
            }
        }
    }
    let _ = system.process(&mut external, data);

    drop(external);
    Rc::try_unwrap(sink).map(|c| c.into_inner()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Part, RoomCoordinate};
    use screeps_combat_engine::{resolve_tick, CombatWorld, Intents, SimBody, SimCreep};

    fn room() -> RoomName {
        "W1N1".parse().unwrap()
    }
    fn pos(x: u8, y: u8) -> Position {
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room())
    }
    /// The room one step east of W1N1, derived via `checked_add` (no hardcoded W0/W2).
    fn east_room() -> RoomName {
        pos(49, 25).checked_add((1, 0)).unwrap().room_name()
    }
    fn pos_in(r: RoomName, x: u8, y: u8) -> Position {
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), r)
    }
    fn creep(id: u32, x: u8, y: u8) -> SimCreep {
        SimCreep {
            id,
            owner: 0,
            pos: pos(x, y),
            body: SimBody::unboosted(&[Part::Move, Part::Attack]),
            fatigue: 0,
        }
    }

    #[test]
    fn moves_toward_an_open_goal() {
        let world = CombatWorld { creeps: vec![creep(1, 5, 25)], ..Default::default() };
        let dir = resolve_move_direction(&world, pos(5, 25), 0, &CombatIntent::MoveTo { target: pos(15, 25), range: 0 });
        // 8-directional + uniform cost ⇒ Right / TopRight / BottomRight are all equally-optimal
        // first steps toward an eastern goal; assert we head east, not the exact diagonal.
        assert!(
            matches!(dir, Some(Direction::Right | Direction::TopRight | Direction::BottomRight)),
            "open room → step east toward the goal, got {:?}",
            dir
        );
    }

    #[test]
    fn detours_around_a_wall() {
        // Wall column at x=6, y=23..=27, goal directly east behind it. The first step must not be
        // straight into the wall at (6,25) — it routes around (a diagonal toward a gap).
        let mut world = CombatWorld { creeps: vec![creep(1, 5, 25)], ..Default::default() };
        for y in 23..=27 {
            world.terrain.walls.insert((6, y));
        }
        let dir = resolve_move_direction(&world, pos(5, 25), 0, &CombatIntent::MoveTo { target: pos(10, 25), range: 0 })
            .expect("a route around exists");
        // Stepping Right would enter the wall at (6,25); the planner must pick a detour.
        assert_ne!(dir, Direction::Right, "does not walk into the wall");
        assert!(
            matches!(dir, Direction::TopRight | Direction::BottomRight | Direction::Top | Direction::Bottom),
            "heads around the wall, got {:?}",
            dir
        );
    }

    #[test]
    fn already_in_range_yields_no_move() {
        let world = CombatWorld { creeps: vec![creep(1, 5, 25)], ..Default::default() };
        let dir = resolve_move_direction(&world, pos(5, 25), 0, &CombatIntent::MoveTo { target: pos(7, 25), range: 3 });
        assert_eq!(dir, None, "distance 2 already within range 3 → hold");
    }

    #[test]
    fn flees_away_from_a_threat() {
        let world = CombatWorld { creeps: vec![creep(1, 30, 25)], ..Default::default() };
        let dir = resolve_move_direction(&world, pos(30, 25), 0, &CombatIntent::Flee { from: vec![pos(25, 25)], range: 5 })
            .expect("can flee in an open room");
        // Threat is to the west (x=25); fleeing should move east (away), increasing x.
        assert!(
            matches!(dir, Direction::Right | Direction::TopRight | Direction::BottomRight),
            "flees away from the threat (eastward), got {:?}",
            dir
        );
    }

    #[test]
    fn non_movement_intent_is_none() {
        let world = CombatWorld { creeps: vec![creep(1, 5, 25)], ..Default::default() };
        assert_eq!(resolve_move_direction(&world, pos(5, 25), 0, &CombatIntent::Idle), None);
    }

    // ── Cross-room direction production (ADR 0023 S3) ─────────────────────────────────────────────

    #[test]
    fn aims_toward_the_exit_for_a_cross_room_target() {
        // A target in the adjacent (east) room → the next step heads east toward the exit. The rover's
        // multi-room search routes across once the cost source is room-scoped (S3); previously every
        // room's obstacles overlaid at the same (x,y) so it could not route across a border.
        let east = pos_in(east_room(), 25, 25);
        let world = CombatWorld { creeps: vec![creep(1, 25, 25)], ..Default::default() };
        let dir = resolve_move_direction(&world, pos(25, 25), 0, &CombatIntent::MoveTo { target: east, range: 0 })
            .expect("a cross-room route exists");
        assert!(
            matches!(dir, Direction::Right | Direction::TopRight | Direction::BottomRight),
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
        let mut world = CombatWorld { creeps: vec![creep(1, 45, 25)], ..Default::default() };
        let mut reached = false;
        for _ in 0..40 {
            let from = world.creeps[0].pos;
            if from == target {
                reached = true;
                break;
            }
            let mut i = Intents::new();
            if let Some(dir) = resolve_move_direction(&world, from, 0, &CombatIntent::MoveTo { target, range: 0 }) {
                i.set_move(1, dir);
            }
            resolve_tick(&mut world, &i);
        }
        assert!(reached, "the creep pathed across the border into the east room and reached the target");
    }

    #[test]
    fn system_mover_routes_a_creep_across_a_room_boundary() {
        // The UNIFIED mover (rover MovementSystem + resolver) routes a creep across a border to a
        // target — the sim now runs the SAME traffic-managed mover as live, not just the per-creep
        // shim. The CreepMovementData cache persists across ticks (path reuse + stuck-escalation).
        let target = pos_in(east_room(), 5, 25);
        let mut world = CombatWorld { creeps: vec![creep(1, 45, 25)], ..Default::default() };
        let mut cache = SimMoveCache::new();
        let mut reached = false;
        for _ in 0..60 {
            let from = world.creeps[0].pos;
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
        assert!(reached, "the MovementSystem-routed creep crossed the border and reached the target");
    }

    #[test]
    fn system_mover_deconflicts_two_creeps() {
        // Two creeps moving head-on toward each other's tile in the same row. The unified mover runs
        // BOTH through one resolver pass (the traffic manager), so they pass each other and both reach
        // their targets — no deadlock. (Exercises the multi-creep resolver path the per-creep shim
        // could not coordinate.)
        let mut world = CombatWorld {
            creeps: vec![creep(1, 10, 25), creep(2, 16, 25)],
            ..Default::default()
        };
        let (ta, tb) = (pos(16, 25), pos(10, 25));
        let mut cache = SimMoveCache::new();
        let mut both_arrived = false;
        for _ in 0..30 {
            let a = world.creeps.iter().find(|c| c.id == 1).map(|c| c.pos);
            let b = world.creeps.iter().find(|c| c.id == 2).map(|c| c.pos);
            if a == Some(ta) && b == Some(tb) {
                both_arrived = true;
                break;
            }
            let reqs = [SimMoveRequest::move_to(1, ta, 0), SimMoveRequest::move_to(2, tb, 0)];
            let moves = resolve_moves_via_system(&world, 0, &reqs, &mut cache);
            let mut i = Intents::new();
            for (&id, &dir) in &moves {
                i.set_move(id, dir);
            }
            resolve_tick(&mut world, &i);
        }
        assert!(both_arrived, "the resolver let both creeps pass each other and reach their targets");
    }
}
