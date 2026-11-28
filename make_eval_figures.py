"""Generate 5 publication-quality 300-DPI figures from tier1_run.json and tier4_run.json using Pillow (2x supersampled)."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval_runs" / "v2.2-preregistered"
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

T1 = json.loads((EVAL_DIR / "tier1_run.json").read_text(encoding="utf-8"))
T4 = json.loads((EVAL_DIR / "tier4_run.json").read_text(encoding="utf-8"))

FONT_REG = "/System/Library/Fonts/Supplemental/Georgia.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
FONT_SANS = "/System/Library/Fonts/Helvetica.ttc"

# Colorblind-safe Okabe-Ito / Academic Palette
C_GREEN = (27, 158, 119)
C_BLUE = (55, 126, 184)
C_ORANGE = (217, 95, 2)
C_PURPLE = (117, 112, 179)
C_RED = (213, 94, 0)
C_GRAY = (120, 128, 134)
C_DARK = (32, 36, 40)
C_GRID = (224, 228, 232)
C_BG = (255, 255, 255)


def fnt(size, bold=False, sans=False):
    path = FONT_SANS if sans else (FONT_BOLD if bold else FONT_REG)
    return ImageFont.truetype(path, size * 2)


def new_canvas(w=2200, h=1250):
    img = Image.new("RGB", (w * 2, h * 2), C_BG)
    return img, ImageDraw.Draw(img), w * 2, h * 2


def save_canvas(img, path, target_w=2200, target_h=1250):
    out = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    out.save(path, dpi=(300, 300))
    print(f"Saved {path}")


def draw_centered_text(draw, xy, text, font, fill=C_DARK):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((xy[0] - tw / 2, xy[1] - th / 2), text, font=font, fill=fill)


def draw_right_text(draw, xy, text, font, fill=C_DARK):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((xy[0] - tw, xy[1] - th / 2), text, font=font, fill=fill)


def make_fig1_tier1_models():
    """Fig 1: Tier 1 Pre-Registered (5 seeds) main_test Ordinary vs Delayed Log Loss & ROC-AUC."""
    img, draw, W, H = new_canvas(2200, 1200)
    pr = T1["pre_registered"]["evaluation"]["main_test"]["summary"]
    beh = T1["pre_registered"]["behavioral_summary"]

    models = [
        ("constant", "Constant"),
        ("task_logistic", "Task Logistic"),
        ("history_logistic", "History Logistic"),
        ("history_boosting", "Hist. Boosting"),
        ("kt_decay_gru", "KT Decay GRU"),
        ("time_aware_transformer", "Time-Aware Transf."),
        ("monotonic_transformer", "Monotonic Transf."),
        ("monotonic_gru", "Monotonic GRU"),
        ("time_input_gru", "Time-Input GRU"),
        ("time_decay_gru", "Time-Decay GRU*"),
        ("oracle", "Oracle (Ceiling)"),
    ]

    draw_centered_text(
        draw, (W // 2, 56),
        "Tier 1 Memory Models on main_test (Pre-Registered, 5 Seeds)",
        fnt(32, bold=True),
    )
    draw_centered_text(
        draw, (W // 2, 134),
        "Left: Log Loss (lower is better)   |   Right: ROC-AUC (higher is better)   |   *time_decay_gru failed 6/6 gate",
        fnt(20, sans=True), fill=C_GRAY,
    )

    # Two side-by-side horizontal bar panels
    panels = [
        ("Log Loss (Lower = Better)", 480, 1980, 0.50, 0.71, "ll"),
        ("ROC-AUC (Higher = Better)", 2460, 4120, 0.48, 0.83, "auc"),
    ]
    y0, row_h = 260, 170

    for title, x0, x1, vmin, vmax, kind in panels:
        draw_centered_text(draw, ((x0 + x1) // 2, 200), title, fnt(25, bold=True))
        # Vertical grid lines
        n_ticks = 5
        for t in range(n_ticks + 1):
            val = vmin + (vmax - vmin) * t / n_ticks
            gx = int(x0 + (x1 - x0) * t / n_ticks)
            draw.line([(gx, y0 - 20), (gx, y0 + len(models) * row_h)], fill=C_GRID, width=3)
            draw_centered_text(draw, (gx, y0 + len(models) * row_h + 38), f"{val:.2f}", fnt(19, sans=True), fill=C_GRAY)
        draw.line([(x0, y0 - 20), (x0, y0 + len(models) * row_h)], fill=C_DARK, width=4)

        for idx, (mkey, label) in enumerate(models):
            cy = y0 + idx * row_h + row_h // 2
            if kind == "ll":
                draw_right_text(draw, (x0 - 24, cy), label, fnt(21, bold=(mkey == "monotonic_gru")), fill=C_DARK)

            if kind == "ll":
                v_ord = pr[mkey]["ordinary"]["log_loss"]["mean"]
                v_del = pr[mkey]["delayed_equal_horizon_log_loss"]["mean"]
            else:
                v_ord = pr[mkey]["ordinary"]["roc_auc"]["mean"]
                v_del = pr[mkey]["assessment"]["roc_auc"]["mean"]

            col_ord = C_GREEN if mkey == "monotonic_gru" else (C_RED if mkey == "time_decay_gru" else C_BLUE)
            col_del = (17, 110, 82) if mkey == "monotonic_gru" else (160, 60, 0) if mkey == "time_decay_gru" else C_ORANGE
            if mkey in ("constant", "oracle"):
                col_ord = (140, 145, 150)
                col_del = (90, 95, 100)

            bh = 44
            for sub, val, col in ((-1, v_ord, col_ord), (1, v_del, col_del)):
                by0 = cy - bh - 4 if sub == -1 else cy + 4
                by1 = cy - 4 if sub == -1 else cy + bh + 4
                bx1 = int(x0 + (x1 - x0) * max(0.0, min(1.0, (val - vmin) / (vmax - vmin))))
                draw.rectangle([x0, by0, bx1, by1], fill=col)
                draw.text((bx1 + 12, (by0 + by1) // 2 - 18), f"{val:.4f}", font=fnt(17, sans=True), fill=C_DARK)

    # Legend at bottom
    lx, ly = 1480, H - 70
    draw.rectangle([lx, ly - 16, lx + 44, ly + 16], fill=C_BLUE)
    draw.text((lx + 60, ly - 22), "Ordinary Practice (Day 0)", font=fnt(20, sans=True), fill=C_DARK)
    draw.rectangle([lx + 640, ly - 16, lx + 684, ly + 16], fill=C_ORANGE)
    draw.text((lx + 700, ly - 22), "Delayed Assessment (1/7/30d)", font=fnt(20, sans=True), fill=C_DARK)

    save_canvas(img, FIG_DIR / "fig1_tier1_models.png", 2200, 1200)


def _compute_seed_averaged_bins(csv_gz_path):
    import csv, gzip
    with gzip.open(csv_gz_path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seed_cols = [c for c in rows[0].keys() if c.startswith("monotonic_gru_seed_")]
    bins = {}
    for slice_name, pred_fn in (
        ("ordinary", lambda r: r["purpose"] == "ordinary" and r["delay_days"] in ("", "0", "0.0")),
        ("assessment", lambda r: r["delay_days"] not in ("", "0", "0.0")),
    ):
        sub = [r for r in rows if pred_fn(r)]
        out = []
        for b in range(10):
            lo, hi = b * 0.1, (b + 1) * 0.1
            preds, ys = [], []
            for r in sub:
                p = sum(float(r[c]) for c in seed_cols) / len(seed_cols)
                if (p >= lo and p < hi) or (b == 9 and p >= lo and p <= 1.0):
                    preds.append(p)
                    ys.append(int(r["label"]))
            if preds:
                out.append({
                    "lower": lo, "upper": hi, "queries": len(preds),
                    "mean_prediction": sum(preds) / len(preds),
                    "observed_rate": sum(ys) / len(ys),
                })
            else:
                out.append({"lower": lo, "upper": hi, "queries": 0, "mean_prediction": None, "observed_rate": None})
        bins[slice_name] = out
    return bins


def make_fig2_tier1_reliability():
    """Fig 2: Tier 1 10-Bin Reliability Diagram (monotonic_gru, main_test Ordinary & Delayed)."""
    img, draw, W, H = new_canvas(2200, 1050)
    cal = _compute_seed_averaged_bins(EVAL_DIR / "tier1" / "main_test_predictions.csv.gz")

    draw_centered_text(
        draw, (W // 2, 54),
        "Reliability Diagram for monotonic_gru (Pre-Registered 5-Seed Mean on main_test)",
        fnt(32, bold=True),
    )
    draw_centered_text(
        draw, (W // 2, 128),
        "Bubble size is proportional to query count in each bin; closer to the dashed diagonal = better calibration",
        fnt(20, sans=True), fill=C_GRAY,
    )

    panels = [
        ("Ordinary Practice (ECE-10 = 0.0108)", cal["ordinary"], 460, 1860, C_GREEN),
        ("Delayed Assessments (ECE-10 = 0.0235)", cal["assessment"], 2420, 3820, C_BLUE),
    ]
    y_top, y_bot = 250, 1780
    size = y_bot - y_top

    for title, bins, x_left, x_right, col in panels:
        draw_centered_text(draw, ((x_left + x_right) // 2, 190), title, fnt(24, bold=True))
        # Grid
        for t in range(6):
            frac = t * 0.2
            gx = int(x_left + (x_right - x_left) * frac)
            gy = int(y_bot - size * frac)
            draw.line([(gx, y_top), (gx, y_bot)], fill=C_GRID, width=3)
            draw.line([(x_left, gy), (x_right, gy)], fill=C_GRID, width=3)
            draw_centered_text(draw, (gx, y_bot + 38), f"{frac:.1f}", fnt(19, sans=True), fill=C_GRAY)
            draw_right_text(draw, (x_left - 22, gy), f"{frac:.1f}", fnt(19, sans=True), fill=C_GRAY)

        # Diagonal reference
        for s in range(0, 40):
            f0, f1 = s / 40.0, (s + 0.5) / 40.0
            draw.line(
                [
                    (int(x_left + (x_right - x_left) * f0), int(y_bot - size * f0)),
                    (int(x_left + (x_right - x_left) * f1), int(y_bot - size * f1)),
                ],
                fill=(130, 130, 130), width=4,
            )

        draw.rectangle([x_left, y_top, x_right, y_bot], outline=C_DARK, width=4)
        draw_centered_text(draw, ((x_left + x_right) // 2, y_bot + 100), "Predicted Pass Probability", fnt(21, sans=True))
        draw_centered_text(draw, (x_left - 130, (y_top + y_bot) // 2), "Actual Pass\nRate", fnt(21, sans=True))

        max_n = max((b["queries"] for b in bins), default=1) or 1
        pts = []
        for b in bins:
            if b["queries"] == 0 or b["mean_prediction"] is None:
                continue
            px = int(x_left + (x_right - x_left) * b["mean_prediction"])
            py = int(y_bot - size * b["observed_rate"])
            r = max(14, int(58 * ((b["queries"] / max_n) ** 0.5)))
            pts.append((px, py, r, b["queries"]))

        if len(pts) >= 2:
            draw.line([(p[0], p[1]) for p in pts], fill=col, width=5)
        for px, py, r, qn in pts:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=C_DARK, width=3)

    save_canvas(img, FIG_DIR / "fig2_tier1_reliability.png", 2200, 1050)


def make_fig3_tier4_policies():
    """Fig 3: Tier 4 Pre-Registered (10 seeds, N=400) Delayed Mean & Day-30 across 3 cohorts."""
    img, draw, W, H = new_canvas(2200, 1150)
    ev = T4["pre_registered"]["evaluation"]
    cohorts = [
        ("main_test", "Main Test\n(1.5d gaps)"),
        ("power_law_test", "Power-Law Test\n(stress forgetting)"),
        ("long_gap_test", "Long-Gap Test\n(4.0d gaps)"),
    ]
    policies = [
        ("delayed_retention_ppo", "Delayed PPO (Ours)", C_GREEN),
        ("threshold_heuristic", "Threshold Heuristic", C_BLUE),
        ("random_valid", "Random Valid", C_GRAY),
        ("fixed_curriculum", "Fixed Curriculum", C_PURPLE),
        ("myopic_rl_bandit", "Myopic RL Bandit", C_ORANGE),
    ]

    draw_centered_text(
        draw, (W // 2, 54),
        "Tier 4 Teaching Policies Across 3 Cohorts (Pre-Registered, 10 Seeds, N=400 Learners, Budget 32.0)",
        fnt(31, bold=True),
    )
    draw_centered_text(
        draw, (W // 2, 130),
        "Left: Mean Delayed Pass Rate (1d/7d/30d average)   |   Right: Day-30 Pass Rate",
        fnt(20, sans=True), fill=C_GRAY,
    )

    panels = [
        ("Mean Delayed Pass Rate (%)", "delayed_mean", 260, 2060),
        ("Day-30 Pass Rate (%)", "day_30", 2340, 4140),
    ]
    y_top, y_bot = 280, 1840
    vmin, vmax = 0.44, 0.66

    for title, mkey, x0, x1 in panels:
        draw_centered_text(draw, ((x0 + x1) // 2, 210), title, fnt(25, bold=True))
        for t in range(6):
            val = vmin + (vmax - vmin) * t / 5
            gy = int(y_bot - (y_bot - y_top) * (val - vmin) / (vmax - vmin))
            draw.line([(x0, gy), (x1, gy)], fill=C_GRID, width=3)
            if x0 < 500:
                draw_right_text(draw, (x0 - 20, gy), f"{val * 100:.0f}%", fnt(19, sans=True), fill=C_GRAY)
        draw.line([(x0, y_top), (x0, y_bot)], fill=C_DARK, width=4)
        draw.line([(x0, y_bot), (x1, y_bot)], fill=C_DARK, width=4)

        grp_w = (x1 - x0) / len(cohorts)
        bar_w = int(grp_w * 0.145)
        for c_idx, (ckey, clab) in enumerate(cohorts):
            gx_center = x0 + grp_w * (c_idx + 0.5)
            draw_centered_text(draw, (gx_center, y_bot + 72), clab, fnt(19, sans=True))
            start_x = gx_center - (len(policies) * bar_w) / 2
            for p_idx, (pkey, _, col) in enumerate(policies):
                val = ev[ckey]["summary"][pkey][mkey]["mean"]
                sd = ev[ckey]["summary"][pkey][mkey].get("sd")
                bx0 = int(start_x + p_idx * bar_w + 6)
                bx1 = int(start_x + (p_idx + 1) * bar_w - 6)
                by1 = y_bot
                by0 = int(y_bot - (y_bot - y_top) * (val - vmin) / (vmax - vmin))
                draw.rectangle([bx0, by0, bx1, by1], fill=col)
                label_y = by0 - 30
                if sd:
                    ey_hi = int(y_bot - (y_bot - y_top) * (val + sd - vmin) / (vmax - vmin))
                    ey_lo = int(y_bot - (y_bot - y_top) * (val - sd - vmin) / (vmax - vmin))
                    cx = (bx0 + bx1) // 2
                    draw.line([(cx, ey_hi), (cx, ey_lo)], fill=C_DARK, width=4)
                    label_y = ey_hi - 36
                if pkey in ("delayed_retention_ppo", "threshold_heuristic"):
                    draw_centered_text(draw, ((bx0 + bx1) // 2, label_y), f"{val*100:.1f}%", fnt(16, bold=True))

    # Legend
    lx = 340
    ly = H - 70
    for pkey, plab, col in policies:
        draw.rectangle([lx, ly - 16, lx + 40, ly + 16], fill=col)
        draw.text((lx + 54, ly - 22), plab, font=fnt(20, sans=True), fill=C_DARK)
        lx += 740

    save_canvas(img, FIG_DIR / "fig3_tier4_policies.png", 2200, 1150)


def make_fig4_ablation3():
    """Fig 4: Ablation 3 — Full 36-dim PPO vs 23-dim Mask/Routing-Ablated PPO."""
    img, draw, W, H = new_canvas(2000, 1050)
    ev = T4["pre_registered"]["evaluation"]
    cohorts = [("main_test", "Main Test"), ("power_law_test", "Power-Law Test"), ("long_gap_test", "Long-Gap Test")]

    draw_centered_text(
        draw, (W // 2, 54),
        "Ablation 3: Full 36-Dim PPO vs. 23-Dim No-Long-Term-State PPO (10 Seeds, N=400)",
        fnt(30, bold=True),
    )
    draw_centered_text(
        draw, (W // 2, 130),
        "Error bars show ±1 SD across 10 training seeds; hiding 30-day memory & floor b_t increases variance ~3x and lowers retention",
        fnt(19, sans=True), fill=C_GRAY,
    )

    panels = [("Delayed Mean Pass Rate (%)", "delayed_mean", 280, 1880), ("Day-30 Pass Rate (%)", "day_30", 2140, 3740)]
    y_top, y_bot = 270, 1740
    vmin, vmax = 0.50, 0.66

    for title, mkey, x0, x1 in panels:
        draw_centered_text(draw, ((x0 + x1) // 2, 205), title, fnt(24, bold=True))
        for t in range(5):
            val = vmin + (vmax - vmin) * t / 4
            gy = int(y_bot - (y_bot - y_top) * (val - vmin) / (vmax - vmin))
            draw.line([(x0, gy), (x1, gy)], fill=C_GRID, width=3)
            if x0 < 500:
                draw_right_text(draw, (x0 - 20, gy), f"{val*100:.0f}%", fnt(19, sans=True), fill=C_GRAY)
        draw.line([(x0, y_top), (x0, y_bot)], fill=C_DARK, width=4)
        draw.line([(x0, y_bot), (x1, y_bot)], fill=C_DARK, width=4)

        grp_w = (x1 - x0) / len(cohorts)
        bar_w = int(grp_w * 0.28)
        for c_idx, (ckey, clab) in enumerate(cohorts):
            gx_center = x0 + grp_w * (c_idx + 0.5)
            draw_centered_text(draw, (gx_center, y_bot + 50), clab, fnt(20, sans=True))
            for p_idx, (pkey, col) in enumerate((("delayed_retention_ppo", C_GREEN), ("ppo_no_longterm_state", C_ORANGE))):
                val = ev[ckey]["summary"][pkey][mkey]["mean"]
                sd = ev[ckey]["summary"][pkey][mkey]["sd"]
                bx0 = int(gx_center + (p_idx - 1) * bar_w + 10)
                bx1 = int(gx_center + p_idx * bar_w - 10)
                by0 = int(y_bot - (y_bot - y_top) * (val - vmin) / (vmax - vmin))
                draw.rectangle([bx0, by0, bx1, y_bot], fill=col)
                ey_hi = int(y_bot - (y_bot - y_top) * (val + sd - vmin) / (vmax - vmin))
                ey_lo = int(y_bot - (y_bot - y_top) * (val - sd - vmin) / (vmax - vmin))
                cx = (bx0 + bx1) // 2
                draw.line([(cx, ey_hi), (cx, ey_lo)], fill=C_DARK, width=5)
                draw.line([(cx - 18, ey_hi), (cx + 18, ey_hi)], fill=C_DARK, width=4)
                draw.line([(cx - 18, ey_lo), (cx + 18, ey_lo)], fill=C_DARK, width=4)
                draw_centered_text(draw, (cx, ey_hi - 46), f"{val*100:.2f}%", fnt(18, bold=True))

    lx, ly = 1050, H - 70
    draw.rectangle([lx, ly - 16, lx + 44, ly + 16], fill=C_GREEN)
    draw.text((lx + 60, ly - 22), "Full 36-Dim PPO (with 30d & floor b_t)", font=fnt(20, sans=True), fill=C_DARK)
    draw.rectangle([lx + 960, ly - 16, lx + 1004, ly + 16], fill=C_ORANGE)
    draw.text((lx + 1020, ly - 22), "Ablated 23-Dim PPO (short-term p_med only)", font=fnt(20, sans=True), fill=C_DARK)

    save_canvas(img, FIG_DIR / "fig4_ablation3.png", 2000, 1050)


def make_fig5_budget_curve():
    """Fig 5: Matched Budget Curves (16, 24, 32, 40) Day-30 Pass Rate across 3 cohorts."""
    img, draw, W, H = new_canvas(2200, 1000)
    bc = T4["pre_registered"]["budget_curves"]
    budgets = [16.0, 24.0, 32.0, 40.0]
    cohorts = [("main_test", "Main Test"), ("power_law_test", "Power-Law Test"), ("long_gap_test", "Long-Gap Test")]
    policies = [
        ("delayed_retention_ppo", "Delayed PPO (Ours)", C_GREEN),
        ("threshold_heuristic", "Threshold Heuristic", C_BLUE),
        ("myopic_rl_bandit", "Myopic RL Bandit", C_ORANGE),
    ]

    draw_centered_text(
        draw, (W // 2, 66),
        "Day-30 Pass Rate vs. Study Budget (Pre-Registered, 10 Seeds, N=400 Learners)",
        fnt(32, bold=True),
    )

    y_top, y_bot = 230, 1620
    vmin, vmax = 0.42, 0.65
    panel_w = 1180
    gap_w = 180
    x_start = 320

    for c_idx, (ckey, clab) in enumerate(cohorts):
        x0 = x_start + c_idx * (panel_w + gap_w)
        x1 = x0 + panel_w
        draw_centered_text(draw, ((x0 + x1) // 2, 175), clab, fnt(24, bold=True))

        for t in range(6):
            val = vmin + (vmax - vmin) * t / 5
            gy = int(y_bot - (y_bot - y_top) * (val - vmin) / (vmax - vmin))
            draw.line([(x0, gy), (x1, gy)], fill=C_GRID, width=3)
            if c_idx == 0:
                draw_right_text(draw, (x0 - 20, gy), f"{val*100:.0f}%", fnt(19, sans=True), fill=C_GRAY)

        for b_idx, b in enumerate(budgets):
            bx = int(x0 + 100 + (x1 - x0 - 200) * b_idx / (len(budgets) - 1))
            draw.line([(bx, y_top), (bx, y_bot)], fill=C_GRID, width=3)
            draw_centered_text(draw, (bx, y_bot + 42), f"{int(b)}", fnt(20, sans=True), fill=C_DARK)

        draw.line([(x0, y_top), (x0, y_bot)], fill=C_DARK, width=4)
        draw.line([(x0, y_bot), (x1, y_bot)], fill=C_DARK, width=4)
        draw_centered_text(draw, ((x0 + x1) // 2, y_bot + 100), "Study Budget (B)", fnt(20, sans=True))

        for pkey, _, col in policies:
            pts = []
            for b_idx, b in enumerate(budgets):
                val = bc[f"budget_{b:g}"][ckey]["summary"][pkey]["day_30"]["mean"]
                bx = int(x0 + 100 + (x1 - x0 - 200) * b_idx / (len(budgets) - 1))
                by = int(y_bot - (y_bot - y_top) * (val - vmin) / (vmax - vmin))
                pts.append((bx, by, val))
            draw.line([(p[0], p[1]) for p in pts], fill=col, width=7)
            for bx, by, val in pts:
                draw.ellipse([bx - 13, by - 13, bx + 13, by + 13], fill=col, outline=C_DARK, width=3)

    lx, ly = 1100, H - 70
    for pkey, plab, col in policies:
        draw.line([(lx, ly), (lx + 55, ly)], fill=col, width=8)
        draw.ellipse([lx + 18, ly - 10, lx + 38, ly + 10], fill=col, outline=C_DARK, width=2)
        draw.text((lx + 72, ly - 22), plab, font=fnt(20, sans=True), fill=C_DARK)
        lx += 700

    save_canvas(img, FIG_DIR / "fig5_budget_curve.png", 2200, 1000)


if __name__ == "__main__":
    make_fig1_tier1_models()
    make_fig2_tier1_reliability()
    make_fig3_tier4_policies()
    make_fig4_ablation3()
    make_fig5_budget_curve()
