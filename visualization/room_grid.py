"""
room_grid.py — Matplotlib 4-panel room state visualization.

Panel 1: Room layout with zones, fans, lights, occupancy
Panel 2: Airflow heatmap from active fans
Panel 3: Lux heatmap from active lights
Panel 4: Optimizer comparison bar chart
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from core.room_config import RoomConfig
from core.coverage import CoverageResult

AIRFLOW_CMAP = LinearSegmentedColormap.from_list(
    "airflow", ["#1a1a2e", "#16213e", "#0f3460", "#00b4d8", "#90e0ef"]
)
LUX_CMAP = LinearSegmentedColormap.from_list(
    "lux", ["#1a1a2e", "#2d1b69", "#7b2cbf", "#e0aaff", "#ffffc2"]
)


def _draw_room_layout(ax, cfg, zone_occupancy):
    ax.set_title("Room Layout & Occupancy", color="#e2e8f0", fontsize=13, pad=10)
    ax.set_xlim(0, cfg.width); ax.set_ylim(0, cfg.depth); ax.set_aspect("equal")
    ax.set_xlabel("Width (m)", color="#94a3b8"); ax.set_ylabel("Depth (m)", color="#94a3b8")
    for zi, zone in enumerate(cfg.zones):
        occ = int(zone_occupancy[zi])
        alpha = min(0.15 + 0.12 * occ, 0.9)
        color = "#3b82f6" if occ > 0 else "#334155"
        rect = plt.Rectangle((zone.x_min, zone.y_min), zone.x_max - zone.x_min,
                              zone.y_max - zone.y_min, facecolor=color, alpha=alpha,
                              edgecolor="#475569", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(zone.cx, zone.cy, f"Z{zi}\n{occ}p", ha="center", va="center",
                fontsize=9, color="#e2e8f0", fontweight="bold")
    for fan in cfg.fans:
        ax.plot(fan.x, fan.y, "^", color="#22c55e", markersize=10,
                markeredgecolor="#166534", markeredgewidth=1.5, zorder=5)
    for light in cfg.lights:
        ax.plot(light.x, light.y, "o", color="#facc15", markersize=8,
                markeredgecolor="#a16207", markeredgewidth=1.5, zorder=5)
    ax.legend(handles=[
        mpatches.Patch(color="#22c55e", label=f"Fans ({cfg.n_fans})"),
        mpatches.Patch(color="#facc15", label=f"Lights ({cfg.n_lights})"),
        mpatches.Patch(color="#3b82f6", label="Occupied"),
    ], loc="upper right", fontsize=8, facecolor="#1e293b",
       edgecolor="#475569", labelcolor="#e2e8f0")


def _draw_heatmap(ax, cfg, matrix, selected_indices, cmap, title, is_fan):
    ax.set_title(title, color="#e2e8f0", fontsize=13, pad=10)
    ax.set_xlim(0, cfg.width); ax.set_ylim(0, cfg.depth); ax.set_aspect("equal")
    ax.set_xlabel("Width (m)", color="#94a3b8"); ax.set_ylabel("Depth (m)", color="#94a3b8")
    n_fans = cfg.n_fans
    if is_fan:
        relevant = [i for i in selected_indices if i < n_fans]
        mask = np.zeros(n_fans, dtype=bool)
        for i in relevant: mask[i] = True
        values = matrix[mask].sum(axis=0) if mask.any() else np.zeros(cfg.n_zones)
    else:
        relevant = [i - n_fans for i in selected_indices if i >= n_fans]
        mask = np.zeros(cfg.n_lights, dtype=bool)
        for i in relevant: mask[i] = True
        values = matrix[mask].sum(axis=0) if mask.any() else np.zeros(cfg.n_zones)
    vmax = values.max() if values.max() > 0 else 1.0
    for zi, zone in enumerate(cfg.zones):
        rect = plt.Rectangle((zone.x_min, zone.y_min), zone.x_max - zone.x_min,
                              zone.y_max - zone.y_min, facecolor=cmap(values[zi] / vmax),
                              edgecolor="#475569", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(zone.cx, zone.cy, f"{values[zi]:.1f}", ha="center", va="center",
                fontsize=8, color="#e2e8f0")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#94a3b8")


def _draw_optimizer_comparison(ax, cfg, greedy, ilp):
    ax.set_title("Optimizer Comparison", color="#e2e8f0", fontsize=13, pad=10)
    gs, ils = set(greedy["selected"]), set(ilp["selected"])
    all_ids = [a.id for a in cfg.all_appliances]
    x = np.arange(len(all_ids)); w = 0.35
    ax.bar(x - w/2, [1 if a in gs else 0 for a in all_ids], w,
           color="#22c55e", alpha=0.8, label=f"Greedy ({greedy['total_watts']:.0f}W)")
    ax.bar(x + w/2, [1 if a in ils else 0 for a in all_ids], w,
           color="#3b82f6", alpha=0.8, label=f"ILP ({ilp['total_watts']:.0f}W)")
    ax.set_xticks(x); ax.set_xticklabels(all_ids, rotation=45, ha="right", fontsize=7, color="#94a3b8")
    ax.set_ylabel("On / Off", color="#94a3b8"); ax.set_yticks([0, 1])
    ax.set_yticklabels(["Off", "On"], color="#94a3b8")
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")
    max_power = sum(a.power_watts for a in cfg.all_appliances)
    if greedy["total_watts"] > 0:
        ax.text(0.98, 0.95,
                f"Greedy saves {(1-greedy['total_watts']/max_power)*100:.0f}%\n"
                f"ILP saves {(1-ilp['total_watts']/max_power)*100:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#22c55e",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a",
                          edgecolor="#22c55e", alpha=0.8))


def render_room_state(cfg, coverage, zone_occupancy, greedy_result, ilp_result,
                      airflow_matrix, lux_matrix, save_path=None, show=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"SRACE v2 — {cfg.name}  |  {cfg.n_zones} zones  |  "
                 f"{int(zone_occupancy.sum())} people",
                 fontsize=16, fontweight="bold", color="#e2e8f0")
    fig.patch.set_facecolor("#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values(): spine.set_color("#334155")

    _draw_room_layout(axes[0, 0], cfg, zone_occupancy)
    _draw_heatmap(axes[0, 1], cfg, airflow_matrix, greedy_result["selected_indices"],
                  AIRFLOW_CMAP, "Airflow (m/s) — Active Fans", is_fan=True)
    _draw_heatmap(axes[1, 0], cfg, lux_matrix, greedy_result["selected_indices"],
                  LUX_CMAP, "Illuminance (lux) — Active Lights", is_fan=False)
    _draw_optimizer_comparison(axes[1, 1], cfg, greedy_result, ilp_result)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  📊 Saved visualization → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
