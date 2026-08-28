"""
Targeted Reliability Diagram Plotting Module (Niño 3.4 vs. High-Skill/Global)
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_reliability_diagram(
    rel_dict: dict,
    title: str = "",
    title_str: str = "",
    output_png: str = "figures/reliability_diagram.png",
):
    """
    Plots a 2-Column Reliability Diagram:
    - Panel (a): Niño 3.4 Region (5°S–5°N, 170°W–120°W)
    - Panel (b): High-Skill Domain (ACC >= 0.3) or Global Baseline
    """
    main_title = title if title else (title_str if title_str else "Upper-Tercile Event Reliability (>67th Percentile)")
    second_label = rel_dict.get("second_label", "Global Domain")

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.10, wspace=0.20, top=0.88, bottom=0.10, left=0.07, right=0.96)

    ax1_top = fig.add_subplot(gs[0, 0])
    ax1_bot = fig.add_subplot(gs[1, 0], sharex=ax1_top)

    ax2_top = fig.add_subplot(gs[0, 1], sharey=ax1_top)
    ax2_bot = fig.add_subplot(gs[1, 1], sharex=ax2_top, sharey=ax1_bot)

    panels = [
        (ax1_top, ax1_bot, rel_dict["nino34"], "(a) Niño 3.4 Region (5°S–5°N, 170°W–120°W)"),
        (ax2_top, ax2_bot, rel_dict["second_domain"], f"(b) {second_label}"),
    ]

    for ax_top, ax_bot, rdata, panel_title in panels:
        ax_top.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect Reliability (1:1)", zorder=2)
        ax_top.plot(rdata["prob_pred_raw"], rdata["prob_obs_raw"], "o--", color="#d95f02", linewidth=2, markersize=6, label="Raw Model", zorder=4)
        ax_top.plot(rdata["prob_pred_cal"], rdata["prob_obs_cal"], "s-", color="#1b9e77", linewidth=2.2, markersize=6, label="Calibrated Model", zorder=5)

        ax_top.set_xlim([0.0, 1.0])
        ax_top.set_ylim([0.0, 1.0])
        ax_top.grid(True, linestyle=":", alpha=0.5)
        ax_top.set_title(panel_title, fontsize=10.5, fontweight="bold", pad=8)
        ax_top.legend(loc="upper left", frameon=True, fontsize=8.5)

        width = 0.03
        centers = rdata["bin_centers"]
        t_raw, t_cal = np.sum(rdata["counts_raw"]), np.sum(rdata["counts_cal"])

        freq_raw = (rdata["counts_raw"] / t_raw) * 100.0 if t_raw > 0 else np.zeros_like(centers)
        freq_cal = (rdata["counts_cal"] / t_cal) * 100.0 if t_cal > 0 else np.zeros_like(centers)

        ax_bot.bar(centers - width / 2, freq_raw, width=width, color="#d95f02", alpha=0.7, label="Raw Freq (%)")
        ax_bot.bar(centers + width / 2, freq_cal, width=width, color="#1b9e77", alpha=0.7, label="Calibrated Freq (%)")

        ax_bot.set_xlabel("Forecasted Probability Bin", fontsize=9.5, fontweight="bold")
        ax_bot.grid(True, linestyle=":", alpha=0.5)
        ax_bot.set_yscale("log")
        ax_bot.set_ylim([0.1, 100.0])

    ax1_top.set_ylabel("Observed Relative Frequency", fontsize=9.5, fontweight="bold")
    ax1_bot.set_ylabel("Frequency (%)", fontsize=8.5, fontweight="bold")
    plt.setp(ax2_top.get_yticklabels(), visible=False)
    plt.setp(ax2_bot.get_yticklabels(), visible=False)

    fig.suptitle(main_title, fontsize=12, fontweight="bold", y=0.97)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Reliability diagram saved to '{output_png}'!")
