import random as _random
import streamlit as st
import numpy as np

from strategy import (
    CATEGORIES, UPPER_CATS,
    score_dice, score_dice_joker, total_score,
    open_categories, _upper_progress,
    is_yahtzee_roll, joker_active, joker_categories,
    rank_keeps, coaching_score, score_batch,
)

st.set_page_config(page_title="Yahtzee Coach", layout="wide")

DICE_FACES = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"}


def valid_scores(category):
    if category == "ones":           return list(range(0, 6))
    if category == "twos":           return [0, 2, 4, 6, 8, 10]
    if category == "threes":         return [0, 3, 6, 9, 12, 15]
    if category == "fours":          return [0, 4, 8, 12, 16, 20]
    if category == "fives":          return [0, 5, 10, 15, 20, 25]
    if category == "sixes":          return [0, 6, 12, 18, 24, 30]
    if category in ("three_of_a_kind", "four_of_a_kind"):
        return [0] + list(range(5, 31))
    if category == "full_house":     return [0, 25]
    if category == "small_straight": return [0, 30]
    if category == "large_straight": return [0, 40]
    if category == "yahtzee":        return [0, 50]
    if category == "chance":         return list(range(5, 31))
    return []


# ── Session state ─────────────────────────────────────────────────────────────

def _new_game():
    sc = {c: None for c in CATEGORIES}
    sc["yahtzee_bonus"] = 0
    st.session_state.scorecard    = sc
    st.session_state.roll_num     = 1
    st.session_state.locked       = []
    st.session_state.keep_ranks   = None
    st.session_state.cat_ranks    = None
    st.session_state.best_ev      = None
    st.session_state.coach_result = None
    st.session_state.dice         = [1, 1, 1, 1, 1]
    for i in range(5):
        st.session_state[f"die_{i}"] = 1

if "scorecard" not in st.session_state:
    _new_game()
if "yahtzee_bonus" not in st.session_state.scorecard:
    st.session_state.scorecard["yahtzee_bonus"] = 0


def _clear_ev():
    st.session_state.keep_ranks   = None
    st.session_state.cat_ranks    = None
    st.session_state.best_ev      = None
    st.session_state.coach_result = None


def _start_new_turn():
    """Unlock all dice and reset roll number — user enters their new physical dice."""
    st.session_state.locked   = []
    st.session_state.roll_num = 1
    _clear_ev()


def _die_changed(idx):
    """on_change callback: keep st.session_state.dice in sync immediately.
    This fires BEFORE the rerun, so the face button reads the correct value."""
    if idx not in st.session_state.locked:
        st.session_state.dice[idx] = st.session_state[f"die_{idx}"]


def current_turn():
    sc = st.session_state.scorecard
    return sum(1 for c in CATEGORIES if sc.get(c) is not None) + 1


# ── Sidebar: editable scorecard ───────────────────────────────────────────────

with st.sidebar:
    st.header("Scorecard")
    sc = st.session_state.scorecard
    draft = {}

    st.subheader("Upper Section")
    for cat in UPPER_CATS:
        options = ["—"] + [str(v) for v in valid_scores(cat)]
        current = sc.get(cat)
        idx = options.index(str(current)) if current is not None and str(current) in options else 0
        chosen = st.selectbox(cat.capitalize(), options, index=idx, key=f"sc_{cat}")
        draft[cat] = None if chosen == "—" else int(chosen)

    upper_sum    = sum(draft[c] for c in UPPER_CATS if draft[c] is not None)
    upper_filled = sum(1 for c in UPPER_CATS if draft[c] is not None)
    draft_bonus  = 35 if upper_filled == 6 and upper_sum >= 63 else 0

    st.markdown(f"**Upper total:** {upper_sum} / 63")
    if draft_bonus:
        st.success("+35 bonus ✓")
    elif upper_filled == 6:
        st.warning("Bonus missed")
    else:
        st.info(f"Need {max(0, 63 - upper_sum)} more for +35 bonus")

    st.divider()
    st.subheader("Lower Section")
    for cat in CATEGORIES:
        if cat in UPPER_CATS:
            continue
        options = ["—"] + [str(v) for v in valid_scores(cat)]
        current = sc.get(cat)
        idx = options.index(str(current)) if current is not None and str(current) in options else 0
        label = cat.replace("_", " ").title()
        chosen = st.selectbox(label, options, index=idx, key=f"sc_{cat}")
        draft[cat] = None if chosen == "—" else int(chosen)

    yb_options = [0, 100, 200, 300, 400, 500, 600]
    yb_current = sc.get("yahtzee_bonus", 0)
    yb_idx     = yb_options.index(yb_current) if yb_current in yb_options else 0
    draft_yb   = st.selectbox(
        "Yahtzee Bonus (+100 each)", yb_options, index=yb_idx, key="sc_yahtzee_bonus"
    )
    draft["yahtzee_bonus"] = draft_yb

    draft_total = (
        sum(v for c, v in draft.items() if c in CATEGORIES and v is not None)
        + draft_bonus + draft_yb
    )
    st.divider()
    st.metric("Total", draft_total)

    # Auto-apply when anything changed
    if draft != dict(sc):
        st.session_state.scorecard = draft
        st.session_state.roll_num  = 1
        st.session_state.dice      = [1, 1, 1, 1, 1]
        st.session_state.locked    = []
        _clear_ev()
        st.rerun()

    if st.button("New Game", use_container_width=True):
        _new_game()
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

sc       = st.session_state.scorecard
roll_num = st.session_state.roll_num
turn     = current_turn()

if turn > 13:
    st.title("Game Over!")
    st.metric("Final Score", total_score(sc))
    if st.button("Play Again"):
        _new_game()
        st.rerun()
    st.stop()

st.title("Yahtzee Coach")
st.markdown(f"**Turn {turn} / 13  ·  Roll {roll_num} / 3**")
st.divider()

# ── Dice input ────────────────────────────────────────────────────────────────

st.subheader("Your Dice")
locked = st.session_state.locked

col_roll, _ = st.columns([1, 4])
with col_roll:
    if st.button("🎲 Roll randomly", help="Randomizes unlocked dice"):
        new_dice = list(st.session_state.dice)
        for i in range(5):
            if i not in locked:
                val = _random.randint(1, 6)
                st.session_state[f"die_{i}"] = val  # updates selectbox
                new_dice[i] = val                    # updates face
        st.session_state.dice = new_dice
        _clear_ev()
        st.rerun()

dice_cols = st.columns(5)
dice = []
for i, col in enumerate(dice_cols):
    with col:
        is_locked = i in locked

        # st.session_state.dice[i] is kept current by _die_changed on_change callback,
        # so the face always reflects the value from the previous rerun (no lag).
        face = DICE_FACES[st.session_state.dice[i]]
        btn_label = f"{face} 🔒" if is_locked else face
        if st.button(btn_label, key=f"toggle_{i}", use_container_width=True,
                     help="Click to keep / release"):
            new_locked = list(st.session_state.locked)
            if is_locked:
                new_locked.remove(i)
            else:
                new_locked.append(i)
            st.session_state.locked = sorted(new_locked)
            _clear_ev()
            st.rerun()

        col.selectbox(
            f"Die {i + 1}",
            options=[1, 2, 3, 4, 5, 6],
            key=f"die_{i}",
            disabled=is_locked,
            label_visibility="collapsed",
            on_change=_die_changed,
            args=(i,),
        )
        dice.append(st.session_state.dice[i])

# session_state.dice is already up-to-date via on_change callbacks

st.divider()

# ── Joker check: intercept bonus Yahtzee before normal flow ──────────────────

if joker_active(dice, sc):
    face       = dice[0]
    face_name  = UPPER_CATS[face - 1].capitalize()
    valid_cats = joker_categories(dice, sc)
    upper_prog = _upper_progress(sc)

    st.error(
        f"**YAHTZEE BONUS!** You rolled five {face_name}s and your Yahtzee box "
        f"is already filled. You earn **+100 bonus points** and must place "
        f"the score using the Joker rule."
    )

    arr = np.array(dice, dtype=np.uint8).reshape(1, 5)
    joker_scores_raw = score_batch(arr, valid_cats, upper_prog,
                                   return_max=False, yahtzee_box=50)[0]

    if len(valid_cats) == 1 and valid_cats[0] in UPPER_CATS:
        st.caption("Priority 1: corresponding upper section (forced)")
    elif all(c not in UPPER_CATS for c in valid_cats):
        st.caption("Priority 2: any lower section (Joker — FH=25, SS=30, LS=40)")
    else:
        st.caption("Priority 3: score 0 in any open upper section")

    best_joker_cat   = valid_cats[int(joker_scores_raw.argmax())]
    best_joker_label = best_joker_cat.replace("_", " ").title()
    st.success(f"**Recommended: {best_joker_label}** — "
               f"{score_dice_joker(dice, best_joker_cat)} pts (+ 100 bonus)")

    st.table([
        {"Category": cat.replace("_", " ").title(),
         "Score (Joker)": int(joker_scores_raw[i])}
        for i, cat in enumerate(valid_cats)
    ])

    cat_labels   = [c.replace("_", " ").title() for c in valid_cats]
    chosen_label = st.selectbox("Joker category to score:", cat_labels,
                                index=cat_labels.index(best_joker_label))
    chosen_joker_cat = valid_cats[cat_labels.index(chosen_label)]
    pts = score_dice_joker(dice, chosen_joker_cat)

    if st.button(f"Score {chosen_label} → {pts} pts (+100 bonus)", type="primary"):
        st.session_state.scorecard[chosen_joker_cat] = pts
        st.session_state.scorecard["yahtzee_bonus"]  = sc.get("yahtzee_bonus", 0) + 100
        _start_new_turn()
        st.rerun()

    st.stop()

# ── Recommendation / coaching (rolls 1 & 2) ──────────────────────────────────

if roll_num < 3:
    st.caption("Lock dice by clicking their faces above, then compute EV or score your keep.")

    col_ev, col_score = st.columns(2)

    with col_ev:
        ev_clicked = st.button("⚡ Compute EV", type="primary",
                               use_container_width=True,
                               help="Find the statistically best keep")
    with col_score:
        score_clicked = st.button("📊 Score my keep", use_container_width=True,
                                  help="Compare your locked dice to the optimal keep")

    if ev_clicked:
        with st.spinner("Computing expected values (~0.5s)…"):
            keep_ranks, _, _ = rank_keeps(dice, sc, roll_num)
        st.session_state.keep_ranks   = keep_ranks
        st.session_state.best_ev      = keep_ranks[0][0] if keep_ranks else None
        st.session_state.cat_ranks    = None
        st.session_state.coach_result = None

    if score_clicked:
        # Compute optimal if not already done
        if st.session_state.best_ev is None:
            with st.spinner("Computing EV…"):
                keep_ranks, _, _ = rank_keeps(dice, sc, roll_num)
            st.session_state.keep_ranks = keep_ranks
            st.session_state.best_ev    = keep_ranks[0][0] if keep_ranks else None
        with st.spinner("Scoring your keep…"):
            user_ev, gap = coaching_score(locked, st.session_state.best_ev,
                                          dice, sc, roll_num)
        st.session_state.coach_result = (user_ev, gap)

    # ── EV results ───────────────────────────────────────────────────────────
    if st.session_state.keep_ranks:
        ranks     = st.session_state.keep_ranks
        best_ev   = st.session_state.best_ev
        best_keep = ranks[0][1]
        best_vals = sorted(dice[i] for i in best_keep)

        st.success(
            f"**Optimal keep: {best_vals if best_vals else '[ ] — reroll all'}**"
            f"  ·  EV {best_ev:.1f} pts"
        )
        with st.expander("All candidates"):
            st.table([
                {
                    "Keep":    str(sorted(dice[i] for i in k)) if k else "[] reroll all",
                    "EV":      f"{ev:.1f}",
                    "vs best": f"{ev - best_ev:+.1f}",
                }
                for ev, k in ranks
            ])

    # ── Coaching result ───────────────────────────────────────────────────────
    if st.session_state.coach_result:
        user_ev, gap = st.session_state.coach_result
        best_ev      = st.session_state.best_ev
        best_keep    = st.session_state.keep_ranks[0][1]
        best_vals    = sorted(dice[i] for i in best_keep)
        user_vals    = sorted(dice[i] for i in locked) if locked else []
        optimal_str  = str(best_vals) if best_vals else "[ ] reroll all"

        if gap < 0.5:
            st.success(
                f"Your keep {user_vals or '[ ] reroll all'}: **EV {user_ev:.1f}** — optimal! "
                f"(optimal was also {optimal_str})"
            )
        elif gap < 3.0:
            st.warning(
                f"Your keep {user_vals or '[ ] reroll all'}: **EV {user_ev:.1f}** — close. "
                f"Optimal was {optimal_str} at {best_ev:.1f} pts (gap: {gap:.1f})"
            )
        else:
            st.error(
                f"Your keep {user_vals or '[ ] reroll all'}: **EV {user_ev:.1f}** — "
                f"left {gap:.1f} pts on the table. "
                f"Optimal was {optimal_str} at {best_ev:.1f} pts"
            )

    st.divider()

    # ── Advance to next roll ─────────────────────────────────────────────────
    if st.button(f"→ Roll {roll_num + 1}", type="primary"):
        st.session_state.roll_num += 1
        _clear_ev()
        st.rerun()

# ── Scoring (roll 3) ──────────────────────────────────────────────────────────

if roll_num == 3:
    st.subheader("Score This Turn")

    if st.session_state.cat_ranks is None:
        with st.spinner("Computing best category…"):
            _, cat_ranks, _ = rank_keeps(dice, sc, 3)
        st.session_state.cat_ranks = cat_ranks

    cat_ranks = st.session_state.cat_ranks
    open_cats = open_categories(sc)
    actual    = {cat: score_dice(dice, cat) for cat in open_cats}

    _, best_cat = cat_ranks[0]
    best_label  = best_cat.replace("_", " ").title()
    st.success(f"**Recommended: {best_label}** — {actual[best_cat]} pts")

    with st.expander("All open categories"):
        st.table([
            {
                "Category":            cat.replace("_", " ").title(),
                "Score":               actual[cat],
                "EV (bonus-adjusted)": f"{ev:.1f}",
            }
            for ev, cat in cat_ranks
        ])

    cat_labels   = [c.replace("_", " ").title() for c in open_cats]
    chosen_label = st.selectbox(
        "Category to score:",
        cat_labels,
        index=cat_labels.index(best_label) if best_label in cat_labels else 0,
    )
    chosen_cat = open_cats[cat_labels.index(chosen_label)]
    pts = actual[chosen_cat]

    if st.button(f"Score {chosen_label} → {pts} pts", type="primary"):
        st.session_state.scorecard[chosen_cat] = pts
        _start_new_turn()
        st.rerun()
