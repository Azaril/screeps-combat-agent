//! The sim's movement-planning bridge (P2.M-bridge): turns a tactical movement **goal**
//! ([`CombatIntent::MoveTo`] / [`CombatIntent::Flee`]) into the next-step [`Direction`] by routing
//! through **rover** — a [`CombatWorld`]-backed [`CostMatrixDataSource`] feeds rover's cost-matrix
//! builder, and rover's headless [`LocalPathfinder`] does the multi-step, room-aware search. The
//! caller hands the resulting `Direction` to the engine's `resolve_moves` (the authoritative
//! "server"), so live and sim plan paths through the same system and the engine validates the move
//! (ADR 0006 §B.2). Real pathfinding, not a greedy stepper: a kiter routes *around* obstacles.

use screeps::local::LocalCostMatrix;
use screeps::{Direction, Position, RoomCoordinate, RoomName};
use screeps_combat_decision::CombatIntent;
use screeps_combat_engine::{CombatWorld, PlayerId};
use screeps_rover::{
    ConstructionSiteCostMatrixCache, CostMatrixCache, CostMatrixDataSource, CostMatrixOptions, CostMatrixSystem,
    CostMatrixWrite, CreepCostMatrixCache, LinearCostMatrix, LocalPathfinder, PathfindingProvider,
    StuctureCostMatrixCache,
};

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

/// Project a (possibly cross-room) `target` onto `from`'s current room (ADR 0023 S3 "MoveToRoom"):
/// for a same-room target, the target itself; for a cross-room target, the target's world position
/// **clamped to the current room** — i.e. the room-edge tile that points toward the target's room.
/// The headless [`LocalPathfinder`] is single-room (cross-room travel is a separate MoveToRoom phase,
/// like the live bot's `find_route` + move-to-exit), so the creep routes to that exit tile and the
/// engine's edge-exit relocation (`resolve_tick` Phase D) carries it across; the next room re-projects.
fn in_room_goal(from: Position, target: Position) -> Position {
    if from.room_name() == target.room_name() {
        return target;
    }
    let (fwx, fwy) = from.world_coords();
    let (twx, twy) = target.world_coords();
    let origin_x = fwx - from.x().u8() as i32; // the current room's world-coord origin
    let origin_y = fwy - from.y().u8() as i32;
    let lx = (twx - origin_x).clamp(0, 49) as u8;
    let ly = (twy - origin_y).clamp(0, 49) as u8;
    Position::new(
        RoomCoordinate::new(lx).expect("clamped 0..=49"),
        RoomCoordinate::new(ly).expect("clamped 0..=49"),
        from.room_name(),
    )
}

/// Resolve a movement goal to the next-step [`Direction`] from `from` (owned by `me_owner`), via
/// rover's pathfinder over the `CombatWorld`. Returns `None` for non-movement intents, when already
/// satisfied (empty path), or when no route exists. Combat intents (`Attack`/`Heal`/…) and `Idle`
/// yield `None` here. **Cross-room `MoveTo`** routes to the room-edge tile toward the target (see
/// [`in_room_goal`]); the engine's edge-exit carries the creep across, then the next room re-projects.
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
            // Single-room search to the in-room goal: the target if same-room, else the edge tile
            // toward the target's room (range 0 — reach the exit exactly so the edge-exit fires).
            let goal = in_room_goal(from, *target);
            let goal_range = if goal == *target { *range as u32 } else { 0 };
            pf.search(from, goal, goal_range, &mut room_cb, MAX_OPS, opts.plains_cost, opts.swamp_cost)
        }
        CombatIntent::Flee { from: threats, range } => {
            let goals: Vec<(Position, u32)> = threats.iter().map(|p| (*p, *range as u32)).collect();
            pf.search_many(from, &goals, true, &mut room_cb, MAX_OPS, opts.plains_cost, opts.swamp_cost)
        }
        _ => return None,
    };

    result.path.first().and_then(|next| from.get_direction_to(*next))
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
}
