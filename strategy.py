"""
Pure Yahtzee strategy logic — no Streamlit dependencies.
Extracted from yahtzee_simulation.ipynb for use in the coaching app.
"""

import numpy as np
from collections import Counter
from itertools import combinations

CATEGORIES = [
    "ones", "twos", "threes", "fours", "fives", "sixes",
    "three_of_a_kind", "four_of_a_kind", "full_house",
    "small_straight", "large_straight", "yahtzee", "chance",
]

UPPER_CATS  = ["ones", "twos", "threes", "fours", "fives", "sixes"]
LOWER_CATS  = ["three_of_a_kind", "four_of_a_kind", "full_house",
               "small_straight", "large_straight", "yahtzee", "chance"]
FACE_VALUES = {"ones": 1, "twos": 2, "threes": 3, "fours": 4, "fives": 5, "sixes": 6}

EV_SAMPLES     = 200
EV_SUB_SAMPLES = 100

ALL_32_KEEPS = [list(c) for r in range(6) for c in combinations(range(5), r)]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_dice(dice, category):
    counts = Counter(dice)
    face_counts = sorted(counts.values(), reverse=True)
    if category in FACE_VALUES:
        return sum(d for d in dice if d == FACE_VALUES[category])
    if category == "three_of_a_kind":
        return sum(dice) if face_counts[0] >= 3 else 0
    if category == "four_of_a_kind":
        return sum(dice) if face_counts[0] >= 4 else 0
    if category == "full_house":
        return 25 if sorted(face_counts) == [2, 3] else 0
    if category == "small_straight":
        return 30 if any(set(range(i, i+4)).issubset(set(dice)) for i in range(1, 4)) else 0
    if category == "large_straight":
        return 40 if set(dice) in ({1,2,3,4,5}, {2,3,4,5,6}) else 0
    if category == "yahtzee":
        return 50 if face_counts[0] == 5 else 0
    if category == "chance":
        return sum(dice)
    raise ValueError(f"Unknown category: {category}")


def score_dice_joker(dice, category):
    """Score a Joker placement. FH/SS/LS award their fixed values regardless of dice."""
    if category == "full_house":     return 25
    if category == "small_straight": return 30
    if category == "large_straight": return 40
    return score_dice(dice, category)


def upper_bonus(scorecard):
    total  = sum(scorecard[c] for c in UPPER_CATS if scorecard[c] is not None)
    filled = all(scorecard[c] is not None for c in UPPER_CATS)
    return 35 if filled and total >= 63 else 0


def total_score(scorecard):
    base = sum(v for k, v in scorecard.items() if k in CATEGORIES and v is not None)
    return base + upper_bonus(scorecard) + scorecard.get("yahtzee_bonus", 0)


def open_categories(scorecard):
    return [c for c in CATEGORIES if scorecard[c] is None]


def _upper_progress(scorecard):
    return float(sum(scorecard[c] for c in UPPER_CATS if scorecard[c] is not None))


# ── Joker (bonus Yahtzee) rules ───────────────────────────────────────────────

def is_yahtzee_roll(dice):
    return len(set(dice)) == 1


def joker_active(dice, scorecard):
    """True when the current roll triggers the Joker bonus rule."""
    return is_yahtzee_roll(dice) and scorecard.get("yahtzee") == 50


def joker_categories(dice, scorecard):
    """
    Returns valid category choices under the Joker rule, in priority order:
      1. Corresponding upper section category (if open)
      2. Any open lower section category (FH/SS/LS score fixed values)
      3. Any open upper section category (scores 0)
    """
    face       = dice[0]
    upper_cat  = UPPER_CATS[face - 1]
    open_cats  = open_categories(scorecard)

    if upper_cat in open_cats:
        return [upper_cat]

    lower_open = [c for c in LOWER_CATS if c in open_cats]
    if lower_open:
        return lower_open

    return [c for c in UPPER_CATS if c in open_cats]


# ── NumPy vectorized evaluation ───────────────────────────────────────────────

def reroll_batch(dice, keep_indices, n_samples):
    batch = np.tile(np.array(dice, dtype=np.uint8), (n_samples, 1))
    slots = [i for i in range(5) if i not in keep_indices]
    if slots:
        batch[:, slots] = np.random.randint(1, 7, (n_samples, len(slots)))
    return batch


def score_batch(dice_batch, open_cats, upper_prog, return_max=True, yahtzee_box=None):
    """
    Vectorized scoring across a batch of dice rolls.
    dice_batch  : (N, 5) uint8 array, values 1-6
    open_cats   : list of open category names
    upper_prog  : scalar — sum of already-filled upper section scores
    return_max  : True → (N,) best score; False → (N, n_cats) all scores
    yahtzee_box : scorecard["yahtzee"] value (None/0/50) — enables Joker rule
    """
    N = len(dice_batch)
    fc = np.zeros((N, 6), dtype=np.uint8)
    for f in range(1, 7):
        fc[:, f-1] = (dice_batch == f).sum(axis=1)
    sc   = np.sort(fc, axis=1)[:, ::-1]
    top1 = sc[:, 0];  top2 = sc[:, 1]
    hf   = fc > 0
    total = dice_batch.sum(axis=1).astype(np.float32)

    scores   = np.zeros((N, len(open_cats)), dtype=np.float32)
    old_frac = min(upper_prog / 63.0, 1.0)

    for i, cat in enumerate(open_cats):
        if cat in FACE_VALUES:
            f   = FACE_VALUES[cat]
            raw = (dice_batch == f).sum(axis=1).astype(np.float32) * f
            new_frac = np.minimum((upper_prog + raw) / 63.0, 1.0)
            scores[:, i] = raw + (new_frac - old_frac) * 35.0
        elif cat == "three_of_a_kind":
            scores[:, i] = np.where(top1 >= 3, total, 0)
        elif cat == "four_of_a_kind":
            scores[:, i] = np.where(top1 >= 4, total, 0)
        elif cat == "full_house":
            scores[:, i] = np.where((top1 == 3) & (top2 == 2), 25.0, 0)
        elif cat == "small_straight":
            ss = np.zeros(N, dtype=bool)
            for start in range(3):
                ss |= hf[:, start:start+4].all(axis=1)
            scores[:, i] = np.where(ss, 30.0, 0)
        elif cat == "large_straight":
            ls = hf[:, 0:5].all(axis=1) | hf[:, 1:6].all(axis=1)
            scores[:, i] = np.where(ls, 40.0, 0)
        elif cat == "yahtzee":
            scores[:, i] = np.where(top1 == 5, 50.0, 0)
        elif cat == "chance":
            scores[:, i] = total

    # Joker rule: rows that are Yahtzees when yahtzee_box=50 get overrides + 100 bonus
    if yahtzee_box == 50:
        is_y = (top1 == 5)
        if is_y.any():
            # FH/SS/LS always award fixed values for Joker placements
            for i, cat in enumerate(open_cats):
                if cat == "full_house":
                    scores[is_y, i] = 25.0
                elif cat == "small_straight":
                    scores[is_y, i] = 30.0
                elif cat == "large_straight":
                    scores[is_y, i] = 40.0

    if not return_max:
        return scores

    result = scores.max(axis=1)

    # Apply Joker priority and +100 bonus for Yahtzee rows
    if yahtzee_box == 50:
        is_y = (top1 == 5)
        if is_y.any():
            face_vals  = dice_batch[is_y, 0]          # uint8 1-6
            y_score_mat = scores[is_y]                 # (n_y, n_cats)

            upper_idx = {FACE_VALUES[c]: i
                         for i, c in enumerate(open_cats) if c in FACE_VALUES}
            lower_idxs = [i for i, c in enumerate(open_cats)
                          if c in {"three_of_a_kind", "four_of_a_kind", "full_house",
                                   "small_straight", "large_straight", "chance"}]

            y_scores = np.zeros(int(is_y.sum()), dtype=np.float32)
            for j, face in enumerate(face_vals.tolist()):
                face = int(face)
                if face in upper_idx:
                    y_scores[j] = y_score_mat[j, upper_idx[face]]
                elif lower_idxs:
                    y_scores[j] = y_score_mat[j, lower_idxs].max()
                # else 0 — upper all filled (priority 3)

            result[is_y] = y_scores + 100.0

    return result


# ── Keep heuristic ────────────────────────────────────────────────────────────

def smart_keeps(dice, scorecard):
    open_cats  = open_categories(scorecard)
    candidates = {(), tuple(range(5))}
    counts     = Counter(dice)
    face_indices = {f: [i for i, d in enumerate(dice) if d == f] for f in counts}

    for cat, face in FACE_VALUES.items():
        if cat in open_cats and counts[face] > 0:
            idxs = face_indices[face]
            for n in range(1, len(idxs) + 1):
                candidates.add(tuple(idxs[:n]))

    if {"three_of_a_kind", "four_of_a_kind", "yahtzee"} & set(open_cats):
        for face, cnt in counts.items():
            if cnt >= 2:
                idxs = face_indices[face]
                for n in range(2, cnt + 1):
                    candidates.add(tuple(idxs[:n]))

    if "full_house" in open_cats:
        pairs = {f: face_indices[f] for f, c in counts.items() if c >= 2}
        trips = {f: face_indices[f] for f, c in counts.items() if c >= 3}
        for idxs in pairs.values():
            candidates.add(tuple(idxs[:2]))
        for idxs in trips.values():
            candidates.add(tuple(idxs[:3]))
        for pf, pi in pairs.items():
            for tf, ti in trips.items():
                if pf != tf:
                    candidates.add(tuple(sorted(pi[:2] + ti[:3])))

    if {"small_straight", "large_straight"} & set(open_cats):
        unique_vals = sorted(set(dice))
        run = [unique_vals[0]]
        for v in unique_vals[1:]:
            if v == run[-1] + 1:
                run.append(v)
            else:
                if len(run) >= 3:
                    idx = tuple(next(i for i, d in enumerate(dice) if d == v2) for v2 in run)
                    candidates.add(tuple(sorted(idx)))
                run = [v]
        if len(run) >= 3:
            idx = tuple(next(i for i, d in enumerate(dice) if d == v) for v in run)
            candidates.add(tuple(sorted(idx)))

    return [list(k) for k in candidates]


# ── EV estimation ─────────────────────────────────────────────────────────────

def expected_score_vec(keep_indices, dice, scorecard, remaining_rolls, n_samples=EV_SAMPLES):
    open_cats = open_categories(scorecard)
    if not open_cats:
        return 0.0
    upper_prog  = _upper_progress(scorecard)
    yahtzee_box = scorecard.get("yahtzee")
    batch = reroll_batch(dice, keep_indices, n_samples)

    if remaining_rolls == 1:
        return float(score_batch(batch, open_cats, upper_prog,
                                 yahtzee_box=yahtzee_box).mean())

    leaf_buf = np.empty((n_samples, EV_SUB_SAMPLES, 5), dtype=np.uint8)
    best_ev  = np.full(n_samples, -np.inf, dtype=np.float32)
    for sk in ALL_32_KEEPS:
        sk_slots = [i for i in range(5) if i not in set(sk)]
        for slot in sk:
            leaf_buf[:, :, slot] = batch[:, slot, np.newaxis]
        for slot in sk_slots:
            leaf_buf[:, :, slot] = np.random.randint(1, 7, (n_samples, EV_SUB_SAMPLES))
        leaf_scores = score_batch(
            leaf_buf.reshape(-1, 5), open_cats, upper_prog, yahtzee_box=yahtzee_box
        ).reshape(n_samples, EV_SUB_SAMPLES).mean(axis=1)
        np.maximum(best_ev, leaf_scores, out=best_ev)
    return float(best_ev.mean())


# ── High-level helpers used by the app ───────────────────────────────────────

def rank_keeps(dice, scorecard, roll_num):
    """
    Returns (keep_ranks, cat_ranks, is_joker).
      - Joker active: keep_ranks=None, cat_ranks sorted by Joker scoring, is_joker=True
      - Roll 3 normal: keep_ranks=None, cat_ranks sorted by EV, is_joker=False
      - Roll 1/2 normal: keep_ranks sorted by EV, cat_ranks=None, is_joker=False
    """
    open_cats  = open_categories(scorecard)
    upper_prog = _upper_progress(scorecard)

    # Joker: Yahtzee roll when box already has 50
    if joker_active(dice, scorecard):
        valid_cats = joker_categories(dice, scorecard)
        arr = np.array(dice, dtype=np.uint8).reshape(1, 5)
        # score_batch with yahtzee_box=50 returns Joker-corrected scores
        cat_scores = score_batch(arr, valid_cats, upper_prog,
                                 return_max=False, yahtzee_box=50)[0]
        ranked_cats = sorted(
            [(float(cat_scores[i]), cat) for i, cat in enumerate(valid_cats)],
            reverse=True,
        )
        return None, ranked_cats, True

    remaining = 3 - roll_num
    if remaining == 0:
        arr = np.array(dice, dtype=np.uint8).reshape(1, 5)
        cat_scores = score_batch(arr, open_cats, upper_prog, return_max=False)[0]
        ranked_cats = sorted(
            [(float(cat_scores[i]), cat) for i, cat in enumerate(open_cats)],
            reverse=True,
        )
        return None, ranked_cats, False

    keeps   = smart_keeps(dice, scorecard)
    ranked  = sorted(
        [(expected_score_vec(k, dice, scorecard, remaining), k) for k in keeps],
        reverse=True,
    )
    return ranked, None, False


def coaching_score(user_keep, best_ev, dice, scorecard, roll_num):
    """
    EV for the user's chosen keep vs the known best EV.
    Works for any keep, including ones outside smart_keeps.
    Returns (user_ev, gap).
    """
    remaining = 3 - roll_num
    user_ev   = expected_score_vec(user_keep, dice, scorecard, remaining)
    return user_ev, best_ev - user_ev
