//! Squad support for the sim (P2.M2 validation): a squad whose footprint-aware anchor
//! ([`rover::AnchorPath`](screeps_rover::AnchorPath)) advances toward an objective while its
//! members hold formation (`anchor + offset`) and fight via the seam. This lets the sim exercise
//! **squad-level movement + cohesion** — members stay in formation across obstacles instead of
//! scattering (the squad-scatter fix), measured with [`cohesion`](screeps_combat_decision::cohesion),
//! the same instrument H3 uses. The anchor advance is **cohesion-gated** (hold if members lag) and
//! routes the squad's W×H box around walls; a [`AnchorOutcome::Blocked`] anchor surfaces a path
//! failure for the owner to respond to.

use crate::pathing::{build_combat_matrix, resolve_moves_via_system, SimMoveCache, SimMoveRequest};
use crate::{to_engine_action, SimView};
use screeps::{Part, Position, RoomCoordinate};
use screeps_combat_decision::{
    cohesion, decide_combat, decide_movement, decide_squad_with_pathing,
    kite::{SquadTacticParams, MAX_KITE_OPS},
    CombatIntent, CreepOrders, EngageObjective, FocusTarget, SquadMemberView, SquadMovement, SquadOrderState, SquadStateDto, SquadView,
};
use screeps_combat_engine::{CombatWorld, CreepId, Intents, PlayerId};
use screeps_rover::{AnchorOutcome, AnchorPath, LocalPathfinder, MovementPriority};

/// Members within this Chebyshev distance of their slot count as "in formation".
const COHESION_TOL: u32 = 1;
/// Advance the anchor only when at least this fraction of members are in formation. ALIASES the shared
/// `rally::GATHER_QUORUM_RATIO` so the sim's assault-advance gate and the live bot's gather quorum use ONE
/// constant (the movement-stall fix — they can't drift).
const ADVANCE_QUORUM: f32 = screeps_combat_decision::rally::GATHER_QUORUM_RATIO;
/// Loose-mode (blob / corridor) cohesion radius — members within this of the anchor are gathered. ALIASES
/// the shared `rally::RALLY_GATHER_RADIUS` (same reason).
const LOOSE_RADIUS: u32 = screeps_combat_decision::rally::RALLY_GATHER_RADIUS;

/// `anchor + (dx,dy)`, with an off-room offset **folded** back into the room (mirrored). Near a room
/// edge a formation's far slots would land off-map; folding keeps them as DISTINCT in-room tiles so
/// members spread out instead of stacking on the anchor — the sim's engine `movement.check` has no
/// shoving (that traffic management is the live rover resolver), so converging on one tile piles up.
fn offset_pos(anchor: Position, (dx, dy): (i32, i32)) -> Position {
    let fold = |c: i32, o: i32| {
        let v = c + o;
        if (0..50).contains(&v) {
            v
        } else {
            (c - o).clamp(0, 49) // off-map → mirror the offset back inside
        }
    };
    let x = fold(anchor.x().u8() as i32, dx);
    let y = fold(anchor.y().u8() as i32, dy);
    Position::new(
        RoomCoordinate::new(x as u8).expect("0..=49"),
        RoomCoordinate::new(y as u8).expect("0..=49"),
        anchor.room_name(),
    )
}

/// A squad in the sim: an anchor mover + ordered members (member `i` holds `layout[i]`).
pub struct SimSquad {
    pub owner: PlayerId,
    /// Members in slot order (member `i` ↔ `layout[i]`).
    pub members: Vec<CreepId>,
    /// Formation slot offsets relative to the anchor.
    pub layout: Vec<(i32, i32)>,
    pub anchor: AnchorPath,
    pub objective: Position,
    /// Persisted corridor/loose state: once the box can't fit (a corridor), the squad relaxes to
    /// single-file and stays loose (gated on centroid, not box formation) until it re-gathers into
    /// the box on open terrain. A blob (N>4) is always loose regardless.
    pub loose: bool,
    /// Per-creep movement state (cached path + stuck tracking) for the rover `MovementSystem`,
    /// persisted across ticks so path reuse + the resolver's stuck-escalation accumulate (matches live).
    move_cache: SimMoveCache,
}

impl SimSquad {
    /// The squad's bounding-box footprint `(w,h)` from its layout — the size the anchor path must
    /// fit (so the block routes as a unit, never threading a gap narrower than itself).
    pub fn footprint(&self) -> (u8, u8) {
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (0i32, 0i32, 0i32, 0i32);
        for &(dx, dy) in &self.layout {
            min_x = min_x.min(dx);
            max_x = max_x.max(dx);
            min_y = min_y.min(dy);
            max_y = max_y.max(dy);
        }
        (((max_x - min_x + 1).max(1)) as u8, ((max_y - min_y + 1).max(1)) as u8)
    }

    /// Member positions (living members only), in slot order — for cohesion measurement.
    fn member_positions(&self, sim: &SimView) -> Vec<Position> {
        self.members
            .iter()
            .filter_map(|&id| sim.friend_index(id).map(|i| sim.friends()[i].pos))
            .collect()
    }

    /// Living member positions read from the WHOLE world (any room) — used for the cross-edge gate,
    /// where members straddle the border and the anchor's single-room [`SimView`] can't see them all.
    fn all_member_positions(&self, world: &CombatWorld) -> Vec<Position> {
        self.members
            .iter()
            .filter_map(|&id| world.creeps.iter().find(|c| c.id == id && c.is_alive()).map(|c| c.pos))
            .collect()
    }

    /// Advance the squad one tick. Measures cohesion against the current anchor; advances the
    /// anchor toward the objective only if a quorum is in formation (else holds for stragglers);
    /// then moves each member toward its formation slot and emits seam combat. Returns the engine
    /// [`Intents`] for the squad's creeps plus the anchor [`AnchorOutcome`] (`Blocked` = the path
    /// failed; the owner should respond).
    pub fn step(&mut self, world: &CombatWorld) -> (Intents, AnchorOutcome) {
        let room = self.anchor.virtual_pos.room_name();
        let sim = SimView::from_world(world, self.owner, self.anchor.virtual_pos, room);

        // Cohesion gate: only advance the anchor when the squad is gathered (members near slots).
        let positions = self.member_positions(&sim);
        let anchor_pos = self.anchor.virtual_pos;
        let n = positions.len().max(1) as f32;

        // Mode (P2.M3): a blob (N>4) is always **loose** (centroid cohesion, single-tile footprint).
        // A small squad tries to move as a **box**; if the box can't fit (a corridor) it relaxes to
        // single-file (`self.loose`) and is gated on centroid proximity (a strung-out line is never
        // "in box formation"). The instant the box footprint can advance again — group pathfinding
        // works — it clears `self.loose`, transitioning back to a tight box as soon as possible; the
        // members then re-gather into their slots.
        let blob = self.members.len() > 4;
        let box_rate = cohesion::measure(&positions, Some((anchor_pos, &self.layout)), COHESION_TOL).in_formation_rate;
        // Gathered-near-anchor count via the SHARED rally kernel (`members_gathered_at`) — the SAME
        // instrument the live bot's gather quorum uses, so the sim's assault-advance gate and the bot's
        // can't drift (the movement-stall root cause). `LOOSE_RADIUS == rally::RALLY_GATHER_RADIUS` and
        // `ADVANCE_QUORUM == rally::GATHER_QUORUM_RATIO`, so this is byte-equivalent to the prior inline
        // count/ratio — the drain agent-sim test stays unchanged.
        let anchor_opts: Vec<Option<Position>> = positions.iter().map(|p| Some(*p)).collect();
        let near_anchor =
            screeps_combat_decision::rally::members_gathered_at(&anchor_opts, anchor_pos, LOOSE_RADIUS) as f32 / n;

        // Gate: a strung-out (loose) squad or a blob advances on centroid proximity (so it can keep
        // threading / re-gathering); a formed box advances only when actually in box formation.
        // Cross-edge cohesion (P-MOVE): before the anchor steps across a room border, HOLD (pre-group)
        // until a quorum has clustered at the exit (members fold into distinct in-room slots there),
        // then advance to cross as a bloc — so they aren't strung out across the boundary and picked
        // off one-by-one. Otherwise the normal gate (loose/blob on centroid, a box on box formation).
        let cohesive = if self.anchor.next_step_crosses_room() || blob || self.loose {
            near_anchor >= ADVANCE_QUORUM
        } else {
            box_rate >= ADVANCE_QUORUM
        };

        let mut pf = LocalPathfinder;
        let mut outcome = AnchorOutcome::Advanced;
        if cohesive {
            if blob {
                outcome = self.anchor.advance(self.objective, (1, 1), &mut pf, &mut |r| build_combat_matrix(world, r, self.owner));
            } else {
                // Always attempt to move as a box. Blocked ⇒ a corridor: relax to single-file and
                // mark loose. Not blocked ⇒ the box fits (open terrain): clear loose to re-form.
                outcome = self.anchor.advance(self.objective, self.footprint(), &mut pf, &mut |r| build_combat_matrix(world, r, self.owner));
                if outcome == AnchorOutcome::Blocked {
                    self.loose = true;
                    outcome = self.anchor.advance(self.objective, (1, 1), &mut pf, &mut |r| build_combat_matrix(world, r, self.owner));
                } else {
                    self.loose = false;
                }
            }
        }
        let loose = blob || self.loose;

        // Move members: box → exact slot; loose (blob / corridor) → clump near the anchor (they
        // queue single-file through a 1-wide corridor). Fight via the seam regardless.
        let anchor = self.anchor.virtual_pos; // post-advance (may have crossed a border)
        // While crossing/straddling a border, members CONVERGE on the anchor (range 1) instead of
        // holding box slots: the box slots straddle the boundary (some clamp off-room), so converging
        // gathers the bloc at the exit (pre-group) and pulls stragglers across after it (bulk-cross).
        let straddling = self
            .all_member_positions(world)
            .iter()
            .any(|p| p.room_name() != anchor.room_name());
        // Crossing/straddling a border → hold DISTINCT folded slots (NOT converge on the anchor: that
        // piles up, the sim has no shoving). A genuine 1-wide corridor still single-files behind it.
        let crossing = self.anchor.next_step_crosses_room() || straddling;
        let mut intents = Intents::new();
        let mut move_reqs: Vec<SimMoveRequest> = Vec::new();
        for (slot, &member_id) in self.members.iter().enumerate() {
            // Skip dead/gone members entirely (no combat, no move request).
            if !world.creeps.iter().any(|c| c.id == member_id && c.is_alive()) {
                continue;
            }

            // Combat decision needs the local view → only for members in the anchor's room.
            if let Some(fi) = sim.friend_index(member_id) {
                let actions: Vec<_> = decide_combat(&sim.view_for(fi))
                    .iter()
                    .filter_map(|ci| to_engine_action(ci, &sim))
                    .collect();
                if !actions.is_empty() {
                    intents.set(member_id, actions);
                }
            }

            // Member target by mode: a true 1-wide corridor single-files behind the anchor; a box, a
            // blob, or a border-crossing squad holds DISTINCT (folded) formation slots so members
            // spread to separate tiles (range 1 when loose/crossing, exact when a tight box).
            let (target, range) = if loose && !blob && !crossing {
                (anchor, 1)
            } else {
                let offset = self.layout.get(slot).copied().unwrap_or((0, 0));
                (offset_pos(anchor, offset), if loose || crossing { 1 } else { 0 })
            };
            move_reqs.push(SimMoveRequest::move_to(member_id, target, range));
        }
        // ONE traffic-managed pass: route every member through rover's `MovementSystem` + resolver
        // (swaps / shoves / stuck-escalation), the same mover the live bot uses, then apply the
        // resolved directions. The folded slots above give a good (distinct) target geometry; the
        // resolver deconflicts whatever collisions remain — sim ≡ live.
        for (id, dir) in resolve_moves_via_system(world, self.owner, &move_reqs, &mut self.move_cache) {
            intents.set_move(id, dir);
        }
        (intents, outcome)
    }
}

/// A **manager-fielded** squad in the sim (P2.G3-tail Step 8): anchorless, driven by the pure
/// `decide_squad_with_pathing` (shared focus + heal assignment + the cohesive, pathfinding-scored
/// kite goal) and the per-creep `decide_movement` — exactly the live `SquadManager` + `SquadCombatJob`
/// path (no anchor mover). This is the self-play vehicle that validates cohesive kiting + focus-fire
/// + heal against the engine (no fork: the SAME decision code runs live).
pub struct ManagedSimSquad {
    pub owner: PlayerId,
    /// Members in slot order (the decision indexes the *living* subset of these each tick).
    pub members: Vec<CreepId>,
    /// Where the squad is fighting (the centroid fallback + the room).
    pub objective: Position,
    pub retreat_threshold: f32,
    state: SquadOrderState,
    /// Position-scoring weights (ADR 0019 Stage 4 tuning seam). Defaults to the shipped presets; the
    /// EXP sweep sets custom vectors via [`Self::with_tactics`] to tune them empirically.
    tactics: SquadTacticParams,
    /// Per-creep movement state for the rover `MovementSystem`, persisted across ticks (path reuse +
    /// stuck-escalation), matching live.
    move_cache: SimMoveCache,
    /// What the squad intends toward the enemy: `Destroy` (close + finish; default) vs `Hold` (pin at
    /// standoff). Fed to the engage gate (close-to-kill + stalemate disengage).
    intent: EngageObjective,
    /// Stalemate tracking: the previous tick's total ENEMY hits + how many consecutive ticks we've made
    /// no headway on it. Past [`STALL_LIMIT`] the squad reports `enemy_stalled` (disengage under Destroy).
    prev_enemy_hits: Option<u32>,
    stall_ticks: u32,
    /// Whether the resolver may shove/swap others to reach a tile (the rover default). Off = A/B the
    /// effect of shoving on positioning (the investigated control).
    shove_enabled: bool,
    /// ADR 0031 #39 — field this squad in a TOWER-DRAIN stance: the decision holds the falloff standoff
    /// while the finite towers bleed dry (the unwinnable veto's drain exception), then advances + breaches.
    /// Set by the drain-tactic proving test ([`Self::with_drain_stance`]); default `false` (every other
    /// squad takes the byte-unchanged breach/engage path). Drain comps do NOT reach the live bot at P1.
    drain_stance: bool,
}

/// Consecutive no-enemy-HP-progress ticks before a Destroy squad treats the fight as a stalemate and
/// disengages (don't burn `CREEP_LIFE_TIME` on an un-closable standoff).
const STALL_LIMIT: u32 = 40;

impl ManagedSimSquad {
    pub fn new(owner: PlayerId, members: Vec<CreepId>, objective: Position) -> Self {
        Self {
            owner,
            members,
            objective,
            retreat_threshold: 0.3,
            state: SquadOrderState::Forming,
            tactics: SquadTacticParams::default(),
            move_cache: SimMoveCache::default(),
            intent: EngageObjective::Destroy,
            prev_enemy_hits: None,
            stall_ticks: 0,
            shove_enabled: true,
            drain_stance: false,
        }
    }

    /// Enable/disable shoving for this squad's moves (the investigated control — A/B shoving's effect on
    /// positioning). Default on (the rover default).
    pub fn with_shove(mut self, shove: bool) -> Self {
        self.shove_enabled = shove;
        self
    }

    /// ADR 0031 #39 — field this squad in a tower-DRAIN stance (the drain-tactic proving vehicle): the
    /// decision holds the falloff standoff while the finite towers bleed dry, then advances + breaches.
    pub fn with_drain_stance(mut self, drain: bool) -> Self {
        self.drain_stance = drain;
        self
    }

    /// Override the position-scoring weights (the EXP-* sweep loop, ADR 0019 Stage 4).
    pub fn with_tactics(mut self, tactics: SquadTacticParams) -> Self {
        self.tactics = tactics;
        self
    }

    /// Set the squad's engage intent — `Destroy` (close + finish, the default) vs `Hold` (pin at
    /// standoff). Drives the close-to-kill gradient + the stalemate disengage.
    pub fn with_intent(mut self, intent: EngageObjective) -> Self {
        self.intent = intent;
        self
    }

    /// Advance one tick: build the `SquadView` from living members, run `decide_squad_with_pathing`
    /// (the squad's ONE bounded kite search), then run the per-creep `decide_combat` + `decide_movement`
    /// with the shared directive, returning the engine [`Intents`].
    pub fn step(&mut self, world: &CombatWorld) -> Intents {
        let room = self.objective.room_name();

        // TRAVEL phase (cross-room): the in-room `SimView` below is scoped to the objective room, so a
        // squad whose members are still in another room would be invisible to it (no intents → no
        // movement). Until the whole living squad has crossed into the objective room, path it there via
        // the per-creep rover (which crosses borders); the in-room combat brain runs only once arrived.
        let living_ids: Vec<CreepId> = self.members.iter().copied().filter(|&id| world.creeps.iter().any(|c| c.id == id && c.is_alive())).collect();
        if living_ids.is_empty() {
            return Intents::new();
        }
        let in_objective_room = |id: CreepId| world.creeps.iter().any(|c| c.id == id && c.is_alive() && c.pos.room_name() == room);
        if !living_ids.iter().copied().all(in_objective_room) {
            let mut intents = Intents::new();
            let goal = CombatIntent::MoveTo { target: self.objective, range: 1 };
            let reqs: Vec<SimMoveRequest> = living_ids.iter().filter_map(|&id| SimMoveRequest::from_intent(id, &goal)).collect();
            for (id, dir) in resolve_moves_via_system(world, self.owner, &reqs, &mut self.move_cache) {
                intents.set_move(id, dir);
            }
            return intents;
        }

        let sim = SimView::from_world(world, self.owner, self.objective, room);

        // Living members in slot order — `member_views` and the decision index by THIS list.
        let living: Vec<(CreepId, usize)> = self.members.iter().filter_map(|&id| sim.friend_index(id).map(|fi| (id, fi))).collect();
        if living.is_empty() {
            return Intents::new();
        }
        let member_views: Vec<SquadMemberView> = living
            .iter()
            .map(|&(_, fi)| {
                let f = &sim.friends()[fi];
                SquadMemberView {
                    hits: f.hits,
                    hits_max: f.hits_max,
                    heal_power: f.working_parts(Part::Heal) as u32,
                    pos: Some(f.pos),
                    has_ranged: f.has_working(Part::RangedAttack),
                    // Per-tick attack output for the engage DMG reward (ADR 0019 focus_damage richness).
                    melee_power: f.working_parts(Part::Attack) as u32 * screeps_combat_engine::constants::ATTACK_POWER,
                    ranged_power: f.working_parts(Part::RangedAttack) as u32 * screeps_combat_engine::constants::RANGED_ATTACK_POWER,
                    damage_taken_last_tick: 0,
                    // ADR 0025: the synthetic id (so the kernel's heal intent resolves this ally) + the
                    // structure-damage/declaim capabilities the kernel's action menu prices.
                    id: f.id,
                    dismantle_power: f.working_parts(Part::Work) as u32 * screeps_combat_engine::constants::DISMANTLE_POWER,
                    claim_power: f.working_parts(Part::Claim) as u32 * screeps_combat_engine::constants::CONTROLLER_ATTACK_PER_PART,
                }
            })
            .collect();

        // Stalemate tracking: total alive ENEMY hits this tick; no decrease for STALL_LIMIT ticks ⇒ a
        // standoff we're not closing → report enemy_stalled (the Destroy disengage; Hold ignores it).
        let enemy_hits: u32 = sim.hostiles().iter().filter(|h| h.hits > 0).map(|h| h.hits).sum();
        match self.prev_enemy_hits {
            Some(prev) if enemy_hits >= prev => self.stall_ticks = self.stall_ticks.saturating_add(1),
            _ => self.stall_ticks = 0,
        }
        self.prev_enemy_hits = Some(enemy_hits);
        let enemy_stalled = self.stall_ticks >= STALL_LIMIT;

        let view = SquadView {
            members: &member_views,
            hostiles: sim.hostiles(),
            structures: sim.structures(),
            retreat_threshold: self.retreat_threshold,
            current_state: self.state,
            // Enemy safe mode nullifies all our combat in the room (engage-veto, ADR 0020 §8).
            enemy_safe_mode: world.safe_mode_owner.is_some_and(|o| o != self.owner),
            engage_objective: self.intent,
            enemy_stalled,
            drain_stance: self.drain_stance,
        };
        let decision = decide_squad_with_pathing(&view, None, self.tactics, &mut |r| build_combat_matrix(world, r, self.owner), MAX_KITE_OPS);
        self.state = decision.state;

        let squad_dto = SquadStateDto {
            center: decision.center.unwrap_or(self.objective),
            room,
            movement: decision.movement,
            cohesion_radius: decision.cohesion_radius,
        };

        let mut intents = Intents::new();
        let mut move_reqs: Vec<SimMoveRequest> = Vec::new();
        for (idx, &(member_id, fi)) in living.iter().enumerate() {
            let heal_target = decision.heal_assignments.iter().find(|a| a.healer_idx == idx).and_then(|a| {
                let &(_, tfi) = living.get(a.target_idx)?;
                let tf = &sim.friends()[tfi];
                Some(FocusTarget { pos: tf.pos, id: tf.id })
            });
            // Per-member focus with damage spill (ADR 0020 §4.2); `None` ⇒ the shared focus. `idx`
            // aligns with `decision.focus_assignments` (member_views were built from `living` in order).
            let focus = decision.focus_assignments.get(idx).copied().flatten().or(decision.focus);
            let orders = CreepOrders { focus, heal_target };
            // ADR 0019 §8 heal-coverage positioning: a pure-support healer gets its OWN tile goal
            // (member_goals) instead of the shared block directive — the live SquadManager applies it the
            // same way (squad_manager.rs). Without this the sim drops §8 and healers can drift out of heal
            // range (the operator-flagged cohesion gap). Stamp it as this member's Advance{range:0}.
            let mut member_dto = squad_dto.clone();
            if let Some(goal) = decision.member_goals.get(idx).copied().flatten() {
                member_dto.movement = SquadMovement::Advance { goal, range: 0 };
            }
            let view_i = sim.view_for_with(fi, &member_dto, orders);

            // ADR 0025: when the kernel ran (Engaged, non-kiting) it already chose this member's ACTION
            // jointly with its position — emit `member_intents` directly. Otherwise (kite/retreat/solo)
            // fall back to the per-creep `decide_combat`.
            let combat_intents = match decision.member_intents.get(idx) {
                Some(ks) if !ks.is_empty() => ks.clone(),
                _ => decide_combat(&view_i),
            };
            let actions: Vec<_> = combat_intents.iter().filter_map(|ci| to_engine_action(ci, &sim)).collect();
            if !actions.is_empty() {
                intents.set(member_id, actions);
            }
            // The squad's movement is the highest-priority movement intent (`decide_movement` returns
            // a priority list; the executor used to take the first with a path). Route it through the
            // shared resolver mover with everyone else's so the manager squad gets traffic management.
            // Combat creeps take HIGH priority so they win the forward (shooting) tile over support —
            // otherwise the resolver's neutral tie-break can park the shooter one tile out of range.
            if let Some(mut req) = decide_movement(&view_i).iter().find_map(|mv| SimMoveRequest::from_intent(member_id, mv)) {
                let f = &sim.friends()[fi];
                let combat = f.has_working(Part::RangedAttack) || f.working_parts(Part::Attack) > 0;
                if combat {
                    req = req.with_priority(MovementPriority::High);
                }
                req = req.with_shove(self.shove_enabled);
                // Anti-scatter anchor: while Engaged + cohesive, confine each member's shoves/swaps to
                // within the cohesion radius of the centroid so the resolver can't push the block off its
                // scored tiles (the investigated managed-squad anchoring gap).
                if matches!(decision.state, SquadOrderState::Engaged) && decision.cohesion_radius > 0 {
                    if let Some(center) = decision.center {
                        req = req.with_anchor(center, decision.cohesion_radius);
                    }
                }
                move_reqs.push(req);
            }
        }
        // ONE traffic-managed pass for the whole squad (rover MovementSystem + resolver), like live.
        for (id, dir) in resolve_moves_via_system(world, self.owner, &move_reqs, &mut self.move_cache) {
            intents.set_move(id, dir);
        }
        intents
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::RoomName;
    use screeps_combat_engine::{resolve_tick, SimBody, SimCreep};

    fn room() -> RoomName {
        "W1N1".parse().unwrap()
    }
    fn pos(x: u8, y: u8) -> Position {
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room())
    }
    fn creep(id: CreepId, x: u8, y: u8) -> SimCreep {
        SimCreep {
            id,
            owner: 0,
            pos: pos(x, y),
            // balanced body so it clears fatigue and moves every tick on plains.
            body: SimBody::unboosted(&[Part::Attack, Part::Move]),
            fatigue: 0,
        }
    }

    const QUAD: [(i32, i32); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

    fn quad_squad(anchor: Position, objective: Position) -> SimSquad {
        SimSquad {
            owner: 0,
            members: vec![1, 2, 3, 4],
            layout: QUAD.to_vec(),
            anchor: AnchorPath::new(anchor, objective),
            objective,
            loose: false,
            move_cache: SimMoveCache::default(),
        }
    }

    #[test]
    fn managed_squad_travels_across_a_room_border() {
        // Two ranged movers near the WEST edge of W1N1; the objective is just across the border in the
        // west neighbour W2N1. The travel mode must path the managed squad across (the room-scoped view
        // alone can't — the operator-flagged "no cross-room movement").
        let w2: RoomName = "W2N1".parse().unwrap();
        let p2 = |x: u8, y: u8| Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), w2);
        let mut world = CombatWorld {
            creeps: vec![
                SimCreep { id: 1, owner: 0, pos: pos(3, 25), body: SimBody::unboosted(&[Part::RangedAttack, Part::Move]), fatigue: 0 },
                SimCreep { id: 2, owner: 0, pos: pos(3, 26), body: SimBody::unboosted(&[Part::RangedAttack, Part::Move]), fatigue: 0 },
            ],
            ..Default::default()
        };
        let mut squad = ManagedSimSquad::new(0, vec![1, 2], p2(40, 25));
        for _ in 0..150 {
            let i = squad.step(&world);
            resolve_tick(&mut world, &i);
            if world.creeps.iter().all(|c| c.pos.room_name() == w2) {
                break;
            }
        }
        assert!(world.creeps.iter().any(|c| c.pos.room_name() == w2), "the managed squad crossed W1N1 → W2N1 (travel mode)");
    }

    #[test]
    fn a_quad_crosses_an_open_room_staying_in_formation() {
        // Start a 2×2 quad formed at (5,25), objective (40,25). It should arrive cohesively.
        let mut world = CombatWorld {
            creeps: vec![creep(1, 5, 25), creep(2, 6, 25), creep(3, 5, 26), creep(4, 6, 26)],
            ..Default::default()
        };
        let mut squad = quad_squad(pos(5, 25), pos(40, 25));
        let mut worst_in_formation = 1.0f32;
        for _ in 0..80 {
            let (intents, _) = squad.step(&world);
            resolve_tick(&mut world, &intents);
            let sim = SimView::from_world(&world, 0, squad.anchor.virtual_pos, room());
            let s = cohesion::measure(&squad.member_positions(&sim), Some((squad.anchor.virtual_pos, &QUAD)), 1);
            worst_in_formation = worst_in_formation.min(s.in_formation_rate);
            if squad.anchor.virtual_pos == pos(40, 25) {
                break;
            }
        }
        // The anchor reached the objective and the squad never fell apart.
        assert!(squad.anchor.virtual_pos.x().u8() >= 38, "squad advanced to the objective");
        assert!(worst_in_formation >= 0.75, "stayed cohesive throughout (worst {})", worst_in_formation);
    }

    #[test]
    fn a_quad_crosses_a_room_border_as_a_bloc() {
        // P-MOVE cross-edge cohesion: a 2×2 quad near the east border of W1N1, objective in the room
        // to the east. It pre-groups at the exit and crosses as a bloc — all four end in the east
        // room and the pairwise spread stays bounded through the crossing (not strung out one-by-one
        // across the border, which is where a scattered squad gets picked off).
        let east = pos(49, 25).checked_add((1, 0)).unwrap().room_name();
        let objective = Position::new(RoomCoordinate::new(20).unwrap(), RoomCoordinate::new(25).unwrap(), east);
        let mut world = CombatWorld {
            creeps: vec![creep(1, 44, 24), creep(2, 45, 24), creep(3, 44, 25), creep(4, 45, 25)],
            ..Default::default()
        };
        let mut squad = quad_squad(pos(44, 24), objective);
        let mut worst_spread = 0u32;
        let mut all_in_east = false;
        for _ in 0..100 {
            let (intents, _) = squad.step(&world);
            resolve_tick(&mut world, &intents);
            let positions = squad.all_member_positions(&world);
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    worst_spread = worst_spread.max(positions[i].get_range_to(positions[j]));
                }
            }
            if positions.len() == 4 && positions.iter().all(|p| p.room_name() == east) {
                all_in_east = true;
                break;
            }
        }
        assert!(all_in_east, "the whole quad crossed the border into the east room");
        assert!(worst_spread <= 8, "the quad crossed as a bloc, never strung out (worst spread {})", worst_spread);
    }

    #[test]
    fn a_quad_routes_its_footprint_around_a_wall() {
        // A wall band with a 3-wide gap; a 2×2 quad must route through the gap (fits) and not clip.
        let mut world = CombatWorld {
            creeps: vec![creep(1, 5, 25), creep(2, 6, 25), creep(3, 5, 26), creep(4, 6, 26)],
            ..Default::default()
        };
        for y in 0..=49u8 {
            if !(24..=26).contains(&y) {
                world.terrain.walls.insert((20, y)); // wall column with a gap at y=24..=26
            }
        }
        let mut squad = quad_squad(pos(5, 25), pos(35, 25));
        let mut blocked = false;
        for _ in 0..120 {
            let (intents, outcome) = squad.step(&world);
            if outcome == AnchorOutcome::Blocked {
                blocked = true;
            }
            resolve_tick(&mut world, &intents);
            if squad.anchor.virtual_pos.x().u8() >= 33 {
                break;
            }
        }
        assert!(!blocked, "the 2×2 fits the 3-wide gap → never Blocked");
        assert!(squad.anchor.virtual_pos.x().u8() >= 33, "squad threaded the gap to the far side");
    }

    #[test]
    fn a_quad_threads_a_one_wide_corridor_single_file() {
        // A 1-wide gap a 2×2 box can't fit → M3 relaxes to single-file (footprint 1×1, members
        // clump) and threads it, re-forming on the far side.
        let mut world = CombatWorld {
            creeps: vec![creep(1, 5, 25), creep(2, 6, 25), creep(3, 5, 26), creep(4, 6, 26)],
            ..Default::default()
        };
        for y in 0..=49u8 {
            if y != 25 {
                world.terrain.walls.insert((20, y)); // single-tile gap at y=25
            }
        }
        let mut squad = quad_squad(pos(15, 25), pos(35, 25));
        for _ in 0..150 {
            let (intents, _) = squad.step(&world);
            resolve_tick(&mut world, &intents);
            if squad.anchor.virtual_pos.x().u8() >= 33 {
                break;
            }
        }
        assert!(squad.anchor.virtual_pos.x().u8() >= 33, "relaxed to single-file and threaded the 1-wide corridor");
    }

    #[test]
    fn re_forms_a_tight_box_after_a_corridor() {
        // Thread a 1-wide corridor (forces loose/single-file), then verify the squad transitions
        // back to a TIGHT box as soon as the box footprint can path again on the open far side.
        let mut world = CombatWorld {
            creeps: vec![creep(1, 5, 25), creep(2, 6, 25), creep(3, 5, 26), creep(4, 6, 26)],
            ..Default::default()
        };
        for y in 0..=49u8 {
            if y != 25 {
                world.terrain.walls.insert((20, y)); // single-tile gap at y=25
            }
        }
        let mut squad = quad_squad(pos(15, 25), pos(45, 25));
        let mut went_loose = false;
        for _ in 0..300 {
            let (intents, _) = squad.step(&world);
            resolve_tick(&mut world, &intents);
            went_loose |= squad.loose; // must have relaxed to pass the corridor
            if squad.anchor.virtual_pos.x().u8() >= 40 {
                break;
            }
        }
        // Let the members finish re-gathering into the box on the open side.
        for _ in 0..20 {
            let (intents, _) = squad.step(&world);
            resolve_tick(&mut world, &intents);
        }
        assert!(went_loose, "the squad relaxed to single-file in the corridor");
        assert!(!squad.loose, "re-formed: back in tight box mode once group pathfinding worked again");
        let sim = SimView::from_world(&world, 0, squad.anchor.virtual_pos, room());
        let s = cohesion::measure(&squad.member_positions(&sim), Some((squad.anchor.virtual_pos, &QUAD)), 1);
        assert!(s.in_formation_rate >= 0.75, "members re-gathered into the box (in-formation {})", s.in_formation_rate);
        assert!(s.max_pairwise <= 3, "tight again (diameter {})", s.max_pairwise);
    }

    #[test]
    fn reports_blocked_when_fully_sealed() {
        // No gap at all → even the single-file relax fails → Blocked, anchor holds on the near side.
        let mut world = CombatWorld {
            creeps: vec![creep(1, 5, 25), creep(2, 6, 25), creep(3, 5, 26), creep(4, 6, 26)],
            ..Default::default()
        };
        for y in 0..=49u8 {
            world.terrain.walls.insert((20, y)); // fully sealed
        }
        let mut squad = quad_squad(pos(15, 25), pos(35, 25));
        let mut saw_blocked = false;
        for _ in 0..30 {
            let (intents, outcome) = squad.step(&world);
            saw_blocked |= outcome == AnchorOutcome::Blocked;
            resolve_tick(&mut world, &intents);
        }
        assert!(saw_blocked, "fully sealed → Blocked surfaced (even single-file can't pass)");
        assert!(squad.anchor.virtual_pos.x().u8() < 20, "anchor held on the near side, never clipped through");
    }

    // ── EXP-SQUAD-KITE-1: managed cohesive kiting + focus-fire + survival (P2.G3-tail Step 8) ──
    #[test]
    fn exp_squad_kite_1_managed_duo_kites_cohesively_and_focus_fires() {
        // A high-HP melee keeper + a ranged attacker + a healer, driven by the manager path
        // (decide_squad_with_pathing → per-creep decide_movement). The squad should advance to its
        // pathfinding-scored kite goal (shooting range, clear of the keeper's melee reach), stay
        // cohesive (ONE shared goal → the block doesn't separate), and chip the keeper while surviving.
        let keeper_body: Vec<Part> = std::iter::repeat_n(Part::Attack, 5)
            .chain(std::iter::repeat_n(Part::Move, 5))
            .chain(std::iter::repeat_n(Part::Tough, 10))
            .collect();
        let keeper = SimCreep { id: 99, owner: 1, pos: pos(25, 25), body: SimBody::unboosted(&keeper_body), fatigue: 0 };
        let ra_body = [Part::RangedAttack, Part::RangedAttack, Part::RangedAttack, Part::RangedAttack, Part::RangedAttack, Part::Move, Part::Move, Part::Move, Part::Move, Part::Move];
        let attacker = SimCreep { id: 1, owner: 0, pos: pos(20, 25), body: SimBody::unboosted(&ra_body), fatigue: 0 };
        let heal_body = [Part::Heal, Part::Heal, Part::Heal, Part::Move, Part::Move, Part::Move];
        let healer = SimCreep { id: 2, owner: 0, pos: pos(20, 26), body: SimBody::unboosted(&heal_body), fatigue: 0 };

        let mut world = CombatWorld { creeps: vec![keeper, attacker, healer], ..Default::default() };
        let keeper_hits_0 = world.creeps.iter().find(|c| c.id == 99).unwrap().body.hits;

        let mut squad = ManagedSimSquad::new(0, vec![1, 2], pos(25, 25));
        let mut worst_pairwise = 0u32;
        for _ in 0..50 {
            let intents = squad.step(&world);
            resolve_tick(&mut world, &intents);
            let positions: Vec<Position> = world.creeps.iter().filter(|c| c.owner == 0 && c.is_alive()).map(|c| c.pos).collect();
            if positions.len() >= 2 {
                worst_pairwise = worst_pairwise.max(cohesion::measure(&positions, None, 0).max_pairwise);
            }
        }

        let keeper_hits_1 = world.creeps.iter().find(|c| c.id == 99).map(|c| if c.is_alive() { c.body.hits } else { 0 }).unwrap_or(0);
        let duo_alive = world.creeps.iter().filter(|c| c.owner == 0 && c.is_alive()).count();

        assert!(keeper_hits_1 < keeper_hits_0, "the squad focus-fired the keeper ({keeper_hits_0} -> {keeper_hits_1})");
        assert_eq!(duo_alive, 2, "the duo kited to shooting range + survived (took no melee)");
        assert!(worst_pairwise <= 4, "the duo stayed cohesive throughout (worst pairwise {worst_pairwise})");
    }

    #[test]
    fn a_blob_of_five_advances_loosely() {
        // N>4 → loose-centroid mode: the blob advances to the objective staying near the anchor.
        let mut world = CombatWorld {
            creeps: vec![creep(1, 5, 25), creep(2, 6, 25), creep(3, 5, 26), creep(4, 6, 26), creep(5, 5, 24)],
            ..Default::default()
        };
        let mut squad = SimSquad {
            owner: 0,
            members: vec![1, 2, 3, 4, 5],
            layout: QUAD.to_vec(), // ignored in loose mode (N>4)
            anchor: AnchorPath::new(pos(5, 25), pos(30, 25)),
            objective: pos(30, 25),
            loose: false,
            move_cache: SimMoveCache::default(),
        };
        for _ in 0..90 {
            let (intents, _) = squad.step(&world);
            resolve_tick(&mut world, &intents);
            if squad.anchor.virtual_pos.x().u8() >= 28 {
                break;
            }
        }
        assert!(squad.anchor.virtual_pos.x().u8() >= 28, "the 5-blob advanced to the objective");
        let sim = SimView::from_world(&world, 0, squad.anchor.virtual_pos, room());
        let near = squad.member_positions(&sim).iter().filter(|p| p.get_range_to(squad.anchor.virtual_pos) <= LOOSE_RADIUS).count();
        assert!(near >= 4, "blob stayed loosely gathered near the anchor ({} of 5 within {})", near, LOOSE_RADIUS);
    }

    #[test]
    fn a_winnable_melee_heal_siege_closes_and_dismantles_under_tower_fire() {
        // Operator-flagged "melee+heal sitting outside of range": a melee+heal squad facing a
        // tower-defended structure must CLOSE to range 1 and dismantle when the fight is winnable (the
        // squad out-sustains the tower). (The unwinnable case correctly retreats — that's the gate, not
        // this.) A 6-strong TOUGH/ATTACK/HEAL squad out-heals one close tower → it should reach + raze.
        use crate::scenario::ScenarioBuilder;
        use screeps_combat_engine::StructureKind;
        let mut b = ScenarioBuilder::empty(room());
        let spawn_id = b.structure(StructureKind::Spawn, Some(1), 25, 25, 50_000, 50_000);
        b.tower(1, 24, 16, 100_000);
        let mut world = b.build();
        let body = [
            Part::Tough, Part::Tough, Part::Attack, Part::Attack, Part::Attack,
            Part::Heal, Part::Heal, Part::Heal, Part::Heal, Part::Heal, Part::Heal, Part::Heal, Part::Heal, Part::Heal, Part::Heal, Part::Heal,
            Part::Move, Part::Move, Part::Move, Part::Move, Part::Move, Part::Move,
        ];
        for (i, y) in [23u8, 24, 25, 26, 27, 28].into_iter().enumerate() {
            world.creeps.push(SimCreep { id: 1 + i as u32, owner: 0, pos: pos(20, y), body: SimBody::unboosted(&body), fatigue: 0 });
        }
        let hits_0 = world.structures.iter().find(|s| s.id == spawn_id).unwrap().hits;
        let mut squad = ManagedSimSquad::new(0, vec![1, 2, 3, 4, 5, 6], pos(25, 25));
        let mut min_range = 99u32;
        for _ in 0..60 {
            let intents = squad.step(&world);
            resolve_tick(&mut world, &intents);
            for c in world.creeps.iter().filter(|c| c.owner == 0 && c.is_alive()) {
                min_range = min_range.min(c.pos.get_range_to(pos(25, 25)));
            }
        }
        let hits_1 = world.structures.iter().find(|s| s.id == spawn_id).map(|s| s.hits).unwrap_or(0);
        assert_eq!(min_range, 1, "the melee+heal squad closed to range 1 of the structure");
        assert!(hits_1 < hits_0, "and dismantled it under tower fire ({hits_0} -> {hits_1})");
    }

    #[test]
    fn a_drain_squad_bleeds_finite_towers_dry_then_breaches() {
        // ADR 0031 #39 P1 — THE MAKE-OR-BREAK: a managed TOUGH+HEAL(+WORK) drain squad in the drain stance
        // vs a FINITE-energy multi-tower nest (driven by the scripted `tower_intents`). It must:
        //   (1) HOLD the falloff standoff (not charge into the point-blank tower dps and die / not retreat
        //       on the unwinnable veto — the drain-scoped exception lets it hold), while
        //   (2) the towers BLEED to 0 energy (10/shot/tick under sustained fire), then
        //   (3) ADVANCE on the dead base and DISMANTLE it (the DRY→ADVANCE transition).
        // This is the runtime drain tactic flowing through the SAME `decide_squad_with_pathing` the live
        // bot runs — proven offline on the bit-deterministic sim. If this can't be made to work it's
        // reported honestly, not faked.
        use crate::opponents::tower_intents;
        use crate::scenario::ScenarioBuilder;
        use screeps_combat_engine::StructureKind;

        let mut b = ScenarioBuilder::empty(room());
        let spawn_id = b.structure(StructureKind::Spawn, Some(1), 25, 25, 30_000, 30_000);
        // TWO finite-energy towers flanking the core. 800 energy each ⇒ 80 shots to dry. Point-blank (≤5)
        // they deal 2×600 = 1200/tick — un-out-healable by a single squad (a breach is vetoed); but they're
        // FINITE, so the drain standoff works. At the falloff FLOOR (range ≥20) they deal 2×150 = 300/tick,
        // which the drain tank's 432/tick self-heal beats with the sustain margin → a valid standoff exists.
        b.tower(1, 24, 25, 800);
        b.tower(1, 26, 25, 800);
        let mut world = b.build();

        // The drain squad. A dedicated SOLO tank proves the tactic unambiguously: the towers concentrate on
        // the nearest creep, so the soaking tank must SOLO out-heal the aggregate falloff fire (cross-healing
        // from co-located members is a P2 efficiency win, not part of the tactic proof). The tank carries 36
        // HEAL (×12 = 432/tick self-heal > the 300/tick falloff floor of two towers, with margin), TOUGH for
        // an HP buffer, WORK to dismantle the dead base after the drain, and enough MOVE to hold/reposition.
        let drain_body = {
            let mut v = vec![Part::Tough; 2];
            v.extend(std::iter::repeat_n(Part::Heal, 36));
            v.extend(std::iter::repeat_n(Part::Work, 4));
            v.extend(std::iter::repeat_n(Part::Move, 8));
            v
        };
        // Start it already AT the standoff (range ~20 west of the nest) so the proof is the drain + breach,
        // not the approach transient (the approach is the same Drain directive; the standoff move is tested
        // by the decision unit tests).
        world.creeps.push(SimCreep { id: 1, owner: 0, pos: pos(5, 25), body: SimBody::unboosted(&drain_body), fatigue: 0 });
        let total_tower_energy = |w: &CombatWorld| w.towers.iter().filter(|t| t.owner == 1).map(|t| t.energy).sum::<u32>();
        let start_energy = total_tower_energy(&world);
        assert!(start_energy > 0, "the bed has energized finite towers");
        let core_hits_0 = world.structures.iter().find(|s| s.id == spawn_id).unwrap().hits;

        let mut squad = ManagedSimSquad::new(0, vec![1], pos(25, 25)).with_drain_stance(true);
        let mut towers_dried_at: Option<u32> = None;
        let mut min_range_after_dry = 99u32;
        let mut all_dead = false;
        for tick in 0..600u32 {
            let mut intents = squad.step(&world);
            // The defender's scripted tower AI fires at the nearest attacker (drains energy via can_fire).
            tower_intents(&world, &mut intents);
            resolve_tick(&mut world, &intents);

            if towers_dried_at.is_none() && total_tower_energy(&world) == 0 {
                towers_dried_at = Some(tick);
            }
            // After the drain, track how close the squad gets to the core (the breach advance).
            if towers_dried_at.is_some() {
                for c in world.creeps.iter().filter(|c| c.owner == 0 && c.is_alive()) {
                    min_range_after_dry = min_range_after_dry.min(c.pos.get_range_to(pos(25, 25)));
                }
            }
            if world.creeps.iter().filter(|c| c.owner == 0).all(|c| !c.is_alive()) {
                all_dead = true;
                break;
            }
            if world.structures.iter().find(|s| s.id == spawn_id).is_none_or(|s| !s.is_alive()) {
                break; // core destroyed
            }
        }

        let survivors = world.creeps.iter().filter(|c| c.owner == 0 && c.is_alive()).count();
        let core_hits_1 = world.structures.iter().find(|s| s.id == spawn_id).map(|s| s.hits).unwrap_or(0);

        assert!(!all_dead, "the drain squad survived the soak (it held the standoff + out-healed the falloff fire)");
        assert!(survivors > 0, "at least one drainer lived to breach");
        // (1)+(2) the finite towers BLED to 0.
        assert_eq!(total_tower_energy(&world), 0, "the towers bled to 0 energy (start {start_energy})");
        assert!(towers_dried_at.is_some(), "the drain reached dry towers");
        // (3) DRY→ADVANCE: after the drain the squad closed on the dead base and dismantled it.
        assert!(min_range_after_dry <= 3, "after the drain the squad advanced onto the dead base (min range {min_range_after_dry})");
        assert!(core_hits_1 < core_hits_0, "and dismantled the core ({core_hits_0} -> {core_hits_1})");
    }

    #[test]
    fn the_oracle_decides_drain_then_a_sized_squad_bleeds_the_towers_and_breaches() {
        // ADR 0031 #39 P2+P3 END-TO-END — the drain is now ORACLE-DRIVEN, not a hand-set `with_drain_stance`:
        //   (P2) the force-sizing oracle (`assess`) is fed the bed's defense + a representative single-squad
        //        budget, PICKS `AssaultMode::Drain`, and SIZES a drain comp (`RequiredForce`: HEAL to out-pace
        //        the falloff soak + a TOUGH EHP buffer + WORK to breach), then
        //   (P3) the drain STANCE is derived from the oracle's verdict (`mode == Drain`) — exactly the bit the
        //        live bot threads (war.rs → the objective's runtime `assault_mode` → `SquadView.drain_stance`),
        //        NOT a literal `true` — and the SAME `decide_squad_with_pathing` bleeds the finite towers dry
        //        then breaches.
        use crate::opponents::tower_intents;
        use crate::scenario::ScenarioBuilder;
        use screeps_combat_decision::force_sizing::{assess, AssaultMode, DefenseProfile, ForceBudget, RequiredForce, TowerThreat};
        use screeps_combat_engine::constants::{DISMANTLE_POWER, HEAL_POWER};
        use screeps_combat_engine::StructureKind;

        let mut b = ScenarioBuilder::empty(room());
        let spawn_id = b.structure(StructureKind::Spawn, Some(1), 25, 25, 30_000, 30_000);
        // The finite-energy bed (mirrors `a_drain_squad_bleeds...`): two 800-energy towers flanking the core.
        b.tower(1, 24, 25, 800);
        b.tower(1, 26, 25, 800);
        let mut world = b.build();
        let nest = pos(25, 25);

        // ── P2: feed the oracle the bed's defense + a representative single-squad budget; it must PICK Drain. ──
        // The DefenseProfile mirrors what the bot's `project_defense` builds: each tower at its point-blank
        // range to the nest (the breach assault tile). `assess` evaluates the BREACH at this range (un-out-
        // healable point-blank) but the DRAIN at the falloff standoff (where the heal sustains) — so it picks
        // Drain. The budget is one tank's heal/EHP/dismantle (heal beats the 2×150 falloff floor, big EHP).
        let defense = DefenseProfile {
            towers: world
                .towers
                .iter()
                .filter(|t| t.owner == 1)
                .map(|t| TowerThreat { range_to_assault: t.pos.get_range_to(nest), energy: t.energy })
                .collect(),
            breach_hits: 0,
            objective_hits: world.structures.iter().find(|s| s.id == spawn_id).unwrap().hits,
            repair_per_tick: 0.0,
            safe_mode: false,
        };
        let budget = ForceBudget { max_heal_per_tick: 432.0, max_dismantle_dps: 200.0, tank_effective_hp: 4_400.0, onsite_budget_ticks: 600 };
        // ADR 0031 #41: enemy creep dps is the explicit `assess` arg now (this bed has no defender creeps → 0).
        let assessment = assess(&defense, 0.0, &budget);
        assert!(assessment.winnable, "the oracle finds the finite-tower bed winnable: {}", assessment.reason);
        assert_eq!(assessment.mode, AssaultMode::Drain, "and PICKS the drain (breach can't out-heal point-blank): {}", assessment.reason);

        // The oracle's SIZED drain comp → part counts. HEAL out-paces the falloff soak; TOUGH is the EHP buffer.
        let required = RequiredForce::from_assessment(&assessment);
        assert!(required.heal_parts > 0, "the drain comp is sized with HEAL: {required:?}");
        // The squad fields these parts (+ WORK to breach + MOVE to hold/reposition). One soaking tank proves
        // the tactic; the heal must SOLO out-heal the aggregate falloff, so floor it at the bed's known-good 36.
        let heal_parts = required.heal_parts.max(36);
        let work_parts = required.dismantle_parts.div_ceil(DISMANTLE_POWER).max(4); // ≥ enough to dismantle the dead base
        let tough_parts = required.tough_parts; // the oracle's EHP buffer (may be 0 when heal carries the soak)
        let drain_body = {
            let mut v = vec![Part::Tough; tough_parts as usize];
            v.extend(std::iter::repeat_n(Part::Heal, heal_parts as usize));
            v.extend(std::iter::repeat_n(Part::Work, work_parts as usize));
            v.extend(std::iter::repeat_n(Part::Move, 8));
            v
        };
        // Sanity: the fielded heal out-paces the 2×150 = 300/tick falloff floor (the soak the oracle sized for).
        assert!(heal_parts * HEAL_POWER >= 300, "fielded HEAL out-paces the falloff soak ({heal_parts} parts)");
        world.creeps.push(SimCreep { id: 1, owner: 0, pos: pos(5, 25), body: SimBody::unboosted(&drain_body), fatigue: 0 });

        // ── P3: the drain STANCE is DERIVED from the oracle's verdict (exactly the bot's threading), not a literal. ──
        let drain_stance = assessment.mode == AssaultMode::Drain;
        let mut squad = ManagedSimSquad::new(0, vec![1], nest).with_drain_stance(drain_stance);

        let total_tower_energy = |w: &CombatWorld| w.towers.iter().filter(|t| t.owner == 1).map(|t| t.energy).sum::<u32>();
        let start_energy = total_tower_energy(&world);
        let core_hits_0 = world.structures.iter().find(|s| s.id == spawn_id).unwrap().hits;
        let mut min_range_after_dry = 99u32;
        let mut towers_dried = false;
        for _ in 0..600u32 {
            let mut intents = squad.step(&world);
            tower_intents(&world, &mut intents);
            resolve_tick(&mut world, &intents);
            if !towers_dried && total_tower_energy(&world) == 0 {
                towers_dried = true;
            }
            if towers_dried {
                for c in world.creeps.iter().filter(|c| c.owner == 0 && c.is_alive()) {
                    min_range_after_dry = min_range_after_dry.min(c.pos.get_range_to(nest));
                }
            }
            if world.structures.iter().find(|s| s.id == spawn_id).is_none_or(|s| !s.is_alive()) {
                break;
            }
        }
        let core_hits_1 = world.structures.iter().find(|s| s.id == spawn_id).map(|s| s.hits).unwrap_or(0);
        // The oracle-driven drain bled the finite towers dry, then advanced + dismantled the dead base.
        assert_eq!(total_tower_energy(&world), 0, "the oracle-sized drain bled the towers to 0 (start {start_energy})");
        assert!(min_range_after_dry <= 3, "after the drain the squad advanced onto the dead base (min range {min_range_after_dry})");
        assert!(core_hits_1 < core_hits_0, "and dismantled the core ({core_hits_0} -> {core_hits_1})");
    }
}
