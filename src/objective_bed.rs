//! Objective-bed siege model (ADR 0023 S4): the asymmetric attacker-vs-defender fight the
//! composition auction (ADR 0022 P-AUCTION) scores against. A defended **core** (the win objective)
//! sits behind ramparts; the defending towers **focus-fire the attacker closest to the core** AND
//! **actively maintain the defense** — a spare tower heals the most-damaged defender, else repairs the
//! most-damaged rampart. The attacker must out-DPS the rampart wall *and* the tower's repair to reach
//! and destroy the core, faster than the tower fire wipes it.
//!
//! Defense AI mirrors the engine stronghold (`stronghold.js`): `focusClosest` (all energized towers +
//! in-range defenders hit the hostile closest to the core, L1–3) + `towersMaintenance` (one spare
//! tower heals a damaged on-defense creep, else repairs a damaged rampart). The engine has **no creep
//! repair action**, so active repair is tower-based — which is exactly how strongholds hold (defenders
//! don't repair; towers heal them and the static high-HP ramparts wall the core).
//!
//! The bed world is composed with the existing [`ScenarioBuilder`](crate::scenario::ScenarioBuilder)
//! (core = an owned spawn, plus `.rampart(..)` / `.tower(..)` / defenders). This module owns the
//! defense AI + the siege runner; it does not duplicate the builder.

use crate::pathing::resolve_move_direction;
use screeps::Position;
use screeps_combat_decision::CombatIntent;
use screeps_combat_engine::SimBodyCombat;
use screeps_combat_engine::{
    resolve_tick, CombatAction, CombatWorld, Intents, PlayerId, StructureId, StructureKind,
    TowerAction,
};

/// Who won the siege.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiegeResult {
    /// The attacker destroyed the core — the objective fell.
    CoreBreached,
    /// Every attacker died before breaching.
    AttackersWiped,
    /// The core survived `max_ticks` (the attacker stalled — repair/wall out-held the DPS).
    Held,
}

/// The result of running a siege.
#[derive(Clone, Debug)]
pub struct SiegeOutcome {
    pub result: SiegeResult,
    pub ticks: u32,
    /// The core's remaining hits (0 ⇒ breached).
    pub core_hits: u32,
    pub attackers_alive: usize,
}

/// Produce the defender's tower + defender-creep intents for one tick (engine stronghold AI), adding
/// them to `intents` so the caller can layer the defense over the attacker's intents. `core_owner`
/// defends; `core_pos` is the focus anchor (towers focus the hostile closest to it). No-op offense
/// when there are no hostiles (towers still maintain).
pub fn defense_intents(
    world: &CombatWorld,
    core_owner: PlayerId,
    core_pos: Position,
    intents: &mut Intents,
) {
    // focusClosest: the hostile (non-defender) nearest the core is the focus target.
    let focus = world
        .movement
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner != core_owner)
        .min_by_key(|c| core_pos.get_range_to(c.pos));

    let mut towers: Vec<StructureId> = world
        .towers
        .iter()
        .filter(|t| t.is_alive() && t.owner == core_owner)
        .map(|t| t.id)
        .collect();

    // towersMaintenance: reserve ONE tower to heal the most-damaged defender, else repair the
    // most-damaged owned rampart (the active-repair sustain). Heal takes priority over repair.
    let damaged_defender = world
        .movement
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner == core_owner && c.body.hits < c.body.hits_max())
        .min_by_key(|c| c.body.hits)
        .map(|c| c.id);
    let damaged_rampart = world
        .structures
        .iter()
        .filter(|s| {
            s.is_alive()
                && s.kind == StructureKind::Rampart
                && s.owner == Some(core_owner)
                && s.hits < s.hits_max
        })
        .min_by_key(|s| s.hits)
        .map(|s| s.id);
    if damaged_defender.is_some() || damaged_rampart.is_some() {
        if let Some(maintainer) = towers.pop() {
            if let Some(def) = damaged_defender {
                intents.set_tower(maintainer, TowerAction::Heal(def));
            } else if let Some(ramp) = damaged_rampart {
                intents.set_tower(maintainer, TowerAction::Repair(ramp));
            }
        }
    }

    // Remaining towers attack the focus; in-range defenders attack it too (melee r1 / ranged r≤3).
    if let Some(t) = focus {
        for tid in &towers {
            intents.set_tower(*tid, TowerAction::Attack(t.id));
        }
        for d in world
            .movement
            .creeps
            .iter()
            .filter(|c| c.is_alive() && c.owner == core_owner)
        {
            let r = d.pos.get_range_to(t.pos);
            let action = if d.body.ranged_attack_power() > 0 && r <= 3 {
                Some(if r == 1 {
                    CombatAction::RangedMassAttack
                } else {
                    CombatAction::RangedAttack(t.id)
                })
            } else if d.body.attack_power() > 0 && r == 1 {
                Some(CombatAction::Attack(t.id))
            } else {
                None
            };
            if let Some(a) = action {
                intents.set(d.id, vec![a]);
            }
        }
    }
}

/// A scripted besieger (test + auction default): every attacker dismantles the nearest living rampart
/// it can reach, then the core; otherwise it paths toward whichever it is breaking. Real attacker AIs
/// (the squad / `decide_combat`) plug into [`run_siege`] the same way. `core_pos`/`core_id` identify
/// the objective.
pub fn dismantler_intents(
    world: &CombatWorld,
    attacker_owner: PlayerId,
    core_id: StructureId,
    core_pos: Position,
) -> Intents {
    let mut intents = Intents::new();
    for c in world
        .movement
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner == attacker_owner)
    {
        // The nearest living rampart still walling the core is the gate; once none remain, the core.
        let rampart = world
            .structures
            .iter()
            .filter(|s| s.is_alive() && s.kind == StructureKind::Rampart)
            .min_by_key(|s| c.pos.get_range_to(s.pos));
        let (target_pos, target_id) = match rampart {
            Some(r) => (r.pos, r.id),
            None => (core_pos, core_id),
        };
        if c.pos.get_range_to(target_pos) <= 1 {
            intents.set(c.id, vec![CombatAction::Dismantle(target_id)]);
        } else if let Some(dir) = resolve_move_direction(
            world,
            c.pos,
            attacker_owner,
            &CombatIntent::MoveTo {
                target: target_pos,
                range: 1,
            },
        ) {
            intents.set_move(c.id, dir);
        }
    }
    intents
}

/// Run a siege: `attacker` produces the attacking side's intents each tick; the defender's intents
/// (towers + defenders, [`defense_intents`]) are layered on; `resolve_tick` adjudicates. Stops when
/// the core is destroyed ([`SiegeResult::CoreBreached`]), all attackers die
/// ([`SiegeResult::AttackersWiped`]), or `max_ticks` elapses ([`SiegeResult::Held`]). `core_id` is the
/// objective structure's id; `core_pos` the towers' focus anchor.
pub fn run_siege(
    mut world: CombatWorld,
    core_owner: PlayerId,
    core_id: StructureId,
    core_pos: Position,
    attacker: &mut dyn FnMut(&CombatWorld) -> Intents,
    max_ticks: u32,
) -> SiegeOutcome {
    let attacker_count = |w: &CombatWorld| {
        w.movement
            .creeps
            .iter()
            .filter(|c| c.is_alive() && c.owner != core_owner)
            .count()
    };
    // The core may be a structure or (future) a tower — look in both pools by id.
    let core_hits = |w: &CombatWorld| {
        w.structures
            .iter()
            .find(|s| s.id == core_id)
            .map(|s| s.hits)
            .or_else(|| w.towers.iter().find(|t| t.id == core_id).map(|t| t.hits))
            .unwrap_or(0)
    };
    let mut ticks = 0;
    loop {
        let ch = core_hits(&world);
        let alive = attacker_count(&world);
        if ch == 0 {
            return SiegeOutcome {
                result: SiegeResult::CoreBreached,
                ticks,
                core_hits: 0,
                attackers_alive: alive,
            };
        }
        if alive == 0 {
            return SiegeOutcome {
                result: SiegeResult::AttackersWiped,
                ticks,
                core_hits: ch,
                attackers_alive: 0,
            };
        }
        if ticks >= max_ticks {
            return SiegeOutcome {
                result: SiegeResult::Held,
                ticks,
                core_hits: ch,
                attackers_alive: alive,
            };
        }
        let mut intents = attacker(&world);
        defense_intents(&world, core_owner, core_pos, &mut intents);
        resolve_tick(&mut world, &intents);
        ticks += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::ScenarioBuilder;
    use screeps::{Part, RoomCoordinate, RoomName};
    use screeps_combat_engine::{SimBody, SimCreep};

    const DEFENDER: PlayerId = 1;
    const ATTACKER: PlayerId = 0;

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

    /// Core (owned spawn) at (25,25) behind a rampart at (24,25), a defending tower at (25,27), with
    /// the attacker placed **adjacent** to the rampart at (23,25) — it begins dismantling immediately
    /// (like a sieger that has marched up), so the rampart is damaged before the tower lands much fire.
    /// Returns (world, core_id).
    fn bed(attacker_parts: &[Part], rampart_hits: u32) -> (CombatWorld, StructureId) {
        let mut b = ScenarioBuilder::empty(room());
        let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), 25, 25, 3000, 3000);
        b.tower(DEFENDER, 25, 27, 100_000); // amply energized → a sustained core
        let mut world = b.rampart(DEFENDER, 24, 25, rampart_hits).build();
        world.movement.creeps.push(SimCreep {
            id: 1,
            owner: ATTACKER,
            pos: pos(23, 25),
            body: SimBody::unboosted(attacker_parts),
            fatigue: 0,
            carry_used: 0,
        });
        (world, core_id)
    }

    /// A siege dismantler: TOUGH **first** (the engine destroys body parts from index 0, so the front
    /// soaks tower fire and the WORK behind keeps full dismantle power — the reason real bodies put
    /// TOUGH first), then WORK, then MOVE.
    fn dismantler(n_tough: usize, n_work: usize, n_move: usize) -> Vec<Part> {
        let mut v = vec![Part::Tough; n_tough];
        v.extend(std::iter::repeat_n(Part::Work, n_work));
        v.extend(std::iter::repeat_n(Part::Move, n_move));
        v
    }

    #[test]
    fn under_gunned_attacker_is_repelled() {
        // 5 WORK (dismantle far under one tower's 800 repair): the rampart's active repair holds and
        // the tower fire wears the attacker down. The core never takes a hit.
        let (world, core_id) = bed(&dismantler(0, 5, 5), 4000);
        let outcome = run_siege(
            world,
            DEFENDER,
            core_id,
            pos(25, 25),
            &mut |w| dismantler_intents(w, ATTACKER, core_id, pos(25, 25)),
            300,
        );
        assert_ne!(
            outcome.result,
            SiegeResult::CoreBreached,
            "an under-gunned attacker must not breach"
        );
        assert_eq!(
            outcome.core_hits, 3000,
            "the core took no damage — the rampart held"
        );
    }

    #[test]
    fn sufficient_dps_breaches_the_core() {
        // TOUGH-buffered, 30 WORK (1500 dismantle ≫ 800 repair): the rampart falls despite repair, the
        // attacker reaches the core and destroys it.
        let (world, core_id) = bed(&dismantler(10, 30, 10), 4000);
        let outcome = run_siege(
            world,
            DEFENDER,
            core_id,
            pos(25, 25),
            &mut |w| dismantler_intents(w, ATTACKER, core_id, pos(25, 25)),
            300,
        );
        assert_eq!(
            outcome.result,
            SiegeResult::CoreBreached,
            "sufficient DPS breaches; got {:?}",
            outcome
        );
        assert_eq!(outcome.core_hits, 0);
    }

    #[test]
    fn defense_heals_a_damaged_defender_and_focuses_the_attacker() {
        // Two towers + a damaged defender + an attacker: one tower heals the defender (towersMaintenance),
        // the other attacks the hostile (focusClosest).
        let mut b = ScenarioBuilder::empty(room());
        let _core = b.structure(StructureKind::Spawn, Some(DEFENDER), 25, 25, 3000, 3000);
        b.tower(DEFENDER, 25, 26, 100_000);
        b.tower(DEFENDER, 25, 24, 100_000);
        let mut world = b.build();
        // A damaged defender (lost some hits) + an attacker.
        let mut hurt = SimCreep {
            id: 50,
            owner: DEFENDER,
            pos: pos(26, 25),
            body: SimBody::unboosted(&[Part::Attack, Part::Move]),
            fatigue: 0,
            carry_used: 0,
        };
        hurt.body.hits = 50; // below hits_max → a heal target
        world.movement.creeps.push(hurt);
        world.movement.creeps.push(SimCreep {
            id: 1,
            owner: ATTACKER,
            pos: pos(20, 25),
            body: SimBody::unboosted(&[Part::Attack, Part::Move]),
            fatigue: 0,
            carry_used: 0,
        });

        let mut intents = Intents::new();
        defense_intents(&world, DEFENDER, pos(25, 25), &mut intents);
        let heals = intents
            .towers
            .values()
            .filter(|a| matches!(a, TowerAction::Heal(50)))
            .count();
        let attacks = intents
            .towers
            .values()
            .filter(|a| matches!(a, TowerAction::Attack(1)))
            .count();
        assert_eq!(heals, 1, "one tower heals the damaged defender");
        assert_eq!(attacks, 1, "the other tower attacks the closest hostile");
    }

    /// P-FORCE offline validation (ADR 0022): the heal dimension of force-sizing against the engine's
    /// ground-truth resolve. A squad sized with ENOUGH healers to out-heal the focused tower fire — the
    /// member-count-scaling outcome of `sized_for` (D3): one creep can't carry the heal, so the squad
    /// GROWS healers — HOLDS under sustained tower fire + active rampart repair and breaches the core;
    /// the SAME dismantler with only ONE healer (under the heal threshold) is worn down before it can
    /// break through. Bodies are hand-built here (the bot's `sized_for` lives in a sibling crate the sim
    /// can't depend on); the per-tick numbers mirror what the closed-form oracle (HOLD_MARGIN heal vs
    /// tower DPS) produces. Static positions keep it deterministic; the rampart-breach-through-repair
    /// mechanic is also covered by `sufficient_dps_breaches_the_core` / `attacker_marches_*`.
    #[test]
    fn force_sized_squad_holds_and_breaches_where_underhealed_is_wiped() {
        // The attacker AI: WORK creeps dismantle the nearest living rampart, then the core; HEAL creeps
        // heal the most-damaged ally (adjacent → Heal, ≤3 → RangedHeal). Mirrors the squad's
        // dismantler/healer roles. All units start in range, so no movement is needed (determinism).
        fn sized_squad_intents(
            world: &CombatWorld,
            attacker: PlayerId,
            core_id: StructureId,
            core_pos: Position,
        ) -> Intents {
            let mut intents = Intents::new();
            let wounded = world
                .movement
                .creeps
                .iter()
                .filter(|c| c.is_alive() && c.owner == attacker)
                .min_by_key(|c| c.body.hits)
                .map(|c| (c.id, c.pos));
            for c in world
                .movement
                .creeps
                .iter()
                .filter(|c| c.is_alive() && c.owner == attacker)
            {
                if c.body.dismantle_power() > 0 {
                    let rampart = world
                        .structures
                        .iter()
                        .filter(|s| s.is_alive() && s.kind == StructureKind::Rampart)
                        .min_by_key(|s| c.pos.get_range_to(s.pos));
                    let (tpos, tid) = match rampart {
                        Some(r) => (r.pos, r.id),
                        None => (core_pos, core_id),
                    };
                    if c.pos.get_range_to(tpos) <= 1 {
                        intents.set(c.id, vec![CombatAction::Dismantle(tid)]);
                    }
                } else if c.body.heal_power() > 0 {
                    if let Some((wid, wpos)) = wounded {
                        let r = c.pos.get_range_to(wpos);
                        if r <= 1 {
                            intents.set(c.id, vec![CombatAction::Heal(wid)]);
                        } else if r <= 3 {
                            intents.set(c.id, vec![CombatAction::RangedHeal(wid)]);
                        }
                    }
                }
            }
            intents
        }

        // Bed: a thick rampart (24,25) walls the core (25,25). Three towers at range 16 from the (24,24)
        // assault tile (≈270 attack / ≈360 repair each). Once the rampart is damaged one tower maintains
        // (repairs it), the other two focus the closest attacker (≈540 dps) — survivable by a multi-healer
        // squad, lethal to a single healer before the (30k) rampart falls.
        let build_bed = || {
            let mut b = ScenarioBuilder::empty(room());
            let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), 25, 25, 3000, 3000);
            for tx in [23u8, 24, 25] {
                b.tower(DEFENDER, tx, 8, 100_000); // range 16 from (24,24)
            }
            let world = b.rampart(DEFENDER, 24, 25, 30_000).build();
            (world, core_id)
        };
        // The dismantler sits at (24,24): range 1 to BOTH the rampart (24,25) and the core (25,25), so it
        // breaches and then kills the core without moving. TOUGH front (unboosted = HP buffer), 25 WORK.
        let dismantler: Vec<Part> = std::iter::repeat_n(Part::Tough, 8)
            .chain(std::iter::repeat_n(Part::Work, 25))
            .chain(std::iter::repeat_n(Part::Move, 17))
            .collect();
        let healer: Vec<Part> = std::iter::repeat_n(Part::Heal, 23)
            .chain(std::iter::repeat_n(Part::Move, 10))
            .collect();
        let core_pos = pos(25, 25);

        // ── SIZED: dismantler + THREE healers (≈828 heal/tick ≥ the worst-case ≈810 all-tower burst) ──
        let (mut world, core_id) = build_bed();
        world.movement.creeps.push(SimCreep {
            id: 1,
            owner: ATTACKER,
            pos: pos(24, 24),
            body: SimBody::unboosted(&dismantler),
            fatigue: 0,
            carry_used: 0,
        });
        for (i, (hx, hy)) in [(23u8, 24u8), (23, 23), (24, 23)].into_iter().enumerate() {
            world.movement.creeps.push(SimCreep {
                id: 2 + i as u32,
                owner: ATTACKER,
                pos: pos(hx, hy),
                body: SimBody::unboosted(&healer),
                fatigue: 0,
                carry_used: 0,
            });
        }
        let sized = run_siege(
            world,
            DEFENDER,
            core_id,
            core_pos,
            &mut |w| sized_squad_intents(w, ATTACKER, core_id, core_pos),
            300,
        );
        assert_eq!(
            sized.result,
            SiegeResult::CoreBreached,
            "the force-sized (multi-healer) squad holds under tower fire + active repair and breaches; got {sized:?}"
        );

        // ── UNDER-HEALED: the same dismantler + ONE healer (≈276 heal ≪ the focus dps) → wiped first ──
        let (mut world, core_id) = build_bed();
        world.movement.creeps.push(SimCreep {
            id: 1,
            owner: ATTACKER,
            pos: pos(24, 24),
            body: SimBody::unboosted(&dismantler),
            fatigue: 0,
            carry_used: 0,
        });
        world.movement.creeps.push(SimCreep {
            id: 2,
            owner: ATTACKER,
            pos: pos(23, 24),
            body: SimBody::unboosted(&healer),
            fatigue: 0,
            carry_used: 0,
        });
        let under = run_siege(
            world,
            DEFENDER,
            core_id,
            core_pos,
            &mut |w| sized_squad_intents(w, ATTACKER, core_id, core_pos),
            300,
        );
        assert_ne!(
            under.result,
            SiegeResult::CoreBreached,
            "an under-healed squad is worn down before it can breach (size-to-hold is load-bearing); got {under:?}"
        );
        assert_eq!(
            under.core_hits, 3000,
            "the core never took a hit — the under-healed squad never broke through"
        );
    }
}
