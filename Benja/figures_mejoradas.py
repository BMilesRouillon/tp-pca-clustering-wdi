"""
Genera las figuras mejoradas del TP de Análisis Multivariado (WDI · PCA + Clustering).

Uso:
    cd tp-pca-clustering-wdi
    python Benja/figures_mejoradas.py

Las figuras se guardan en  Benja/figures/.
Requiere que el pipeline principal ya haya corrido (data/processed/ debe existir).
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── rutas ────────────────────────────────────────────────────────────────── #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROC = ROOT / "data" / "processed"
OUT  = HERE / "figures"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))
import theme as T
T.apply_theme()

# ── etiquetas de variables ───────────────────────────────────────────────── #
COLS = [
    "NY.GDP.PCAP.KD", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS",
    "NE.GDI.TOTL.ZS", "NE.EXP.GNFS.ZS",    "NE.IMP.GNFS.ZS", "NV.AGR.TOTL.ZS",
    "NV.IND.TOTL.ZS", "NV.SRV.TOTL.ZS",    "SP.URB.TOTL.IN.ZS", "SP.DYN.LE00.IN",
    "IT.NET.USER.ZS", "EN.GHG.CO2.PC.CE.AR5",
]
SHORT = {
    "NY.GDP.PCAP.KD":       "PBI p/c",
    "NY.GDP.MKTP.KD.ZG":    "Crec. PBI",
    "FP.CPI.TOTL.ZG":       "Inflación",
    "SL.UEM.TOTL.ZS":       "Desempleo",
    "NE.GDI.TOTL.ZS":       "Cap. Fijo",
    "NE.EXP.GNFS.ZS":       "Export.",
    "NE.IMP.GNFS.ZS":       "Import.",
    "NV.AGR.TOTL.ZS":       "Agricultura",
    "NV.IND.TOTL.ZS":       "Industria",
    "NV.SRV.TOTL.ZS":       "Servicios",
    "SP.URB.TOTL.IN.ZS":    "Urbanización",
    "SP.DYN.LE00.IN":       "Esp. de vida",
    "IT.NET.USER.ZS":       "Internet",
    "EN.GHG.CO2.PC.CE.AR5": "CO₂ p/c",
}
LABELS = [SHORT[c] for c in COLS]

# ── helpers de color ─────────────────────────────────────────────────────── #
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def _lerp(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return (r1+(r2-r1)*t, g1+(g2-g1)*t, b1+(b2-b1)*t)

# ══════════════════════════════════════════════════════════════════════════════
# 1 · Matriz de correlaciones con distribuciones en la diagonal
# ══════════════════════════════════════════════════════════════════════════════
def fig_correlacion():
    df = pd.read_csv(PROC / "wide_2023.csv")
    df["NY.GDP.PCAP.KD"]       = np.log(df["NY.GDP.PCAP.KD"].clip(lower=1e-6))
    df["EN.GHG.CO2.PC.CE.AR5"] = np.log1p(df["EN.GHG.CO2.PC.CE.AR5"].clip(lower=0))
    data = df[COLS].dropna()
    n = len(COLS)
    corr = data.corr(method="spearman").values

    fig, axes = plt.subplots(n, n, figsize=(18, 16))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.06, wspace=0.06)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.25); sp.set_color("#dddddd")

            if i == j:
                vals = data.iloc[:, i].dropna().values
                ax.hist(vals, bins=18, color=T.AZUL_MEDIO, alpha=0.80, linewidth=0)
                ax.set_facecolor(T.GRIS_CLARO)
            elif i < j:
                r = corr[i, j]
                t = abs(r)
                bg = _lerp(T.AZUL_CLARO, T.AZUL_PRINCIPAL, t) if r >= 0 \
                     else _lerp(T.ROJO_CLARO, T.ROJO_PRINCIPAL, t)
                ax.set_facecolor(bg)
                tc = "white" if t > 0.45 else T.NEGRO
                ax.text(0.5, 0.5, f"{r:.2f}", ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold",
                        transform=ax.transAxes)
            else:
                ax.set_facecolor("white")
                for sp in ax.spines.values(): sp.set_visible(False)

            # etiquetas de filas (izquierda)
            if j == 0:
                ax.set_ylabel(LABELS[i], fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=42)
                ax.set_yticks([])
            # etiquetas de columnas (abajo)
            if i == n - 1:
                ax.set_xlabel(LABELS[j], fontsize=6.5, rotation=40,
                              ha="right", va="top", labelpad=2)

    from matplotlib.patches import Patch
    leyenda = [
        Patch(facecolor=T.AZUL_PRINCIPAL, label="Correl. positiva fuerte"),
        Patch(facecolor=T.AZUL_CLARO,     label="Correl. positiva débil"),
        Patch(facecolor=T.ROJO_PRINCIPAL,  label="Correl. negativa fuerte"),
        Patch(facecolor=T.ROJO_CLARO,     label="Correl. negativa débil"),
        Patch(facecolor=T.GRIS_CLARO,     label="Distribución marginal"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True,
               edgecolor=T.GRIS_MEDIO, fancybox=True)

    fig.suptitle(
        "Correlaciones de Spearman entre las 14 variables de desarrollo — 2023\n"
        "Triángulo superior: coeficiente ρ  ·  Diagonal: distribución marginal de cada variable",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.savefig(OUT / "fig_correlacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_correlacion.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Scree plot + análisis paralelo de Horn
# ══════════════════════════════════════════════════════════════════════════════
def fig_scree():
    df = pd.read_csv(PROC / "pca_autovalores_2023.csv")
    n_horn = 3

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    x       = np.arange(1, len(df) + 1)
    var_pct = df["var_explicada"].values * 100
    cum_pct = df["var_acumulada"].values * 100
    horn    = df["umbral_horn_p95"].values

    bar_colors = [T.AZUL_PRINCIPAL if i < n_horn else T.AZUL_CLARO
                  for i in range(len(df))]
    bars = ax1.bar(x, var_pct, color=bar_colors, edgecolor="white",
                   linewidth=0.5, zorder=3, width=0.7)

    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct, "o-", color=T.NARANJA, lw=2, ms=6, zorder=4,
             label="Var. acumulada (%)")
    ax2.axhline(80, ls="--", color=T.GRIS_MEDIO, lw=1, label="Referencia 80 %")
    ax2.set_ylabel("Varianza acumulada (%)", fontsize=11, color=T.NARANJA)
    ax2.tick_params(axis="y", colors=T.NARANJA)
    ax2.set_ylim(0, 108)

    ax1.plot(x, horn, "^--", color=T.ROJO_MEDIO, lw=1.5, ms=7, zorder=5,
             label="Umbral Horn (p95)")
    ax1.set_xlabel("Componente principal", fontsize=11)
    ax1.set_ylabel("Varianza explicada (%)", fontsize=11)
    ax1.set_title(
        "Scree plot y Análisis Paralelo de Horn — PCA sobre 14 variables (2023)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x], fontsize=9)
    ax1.set_ylim(0, var_pct.max() * 1.3)

    ax1.axvspan(0.5, n_horn + 0.5, alpha=0.06, color=T.AZUL_PRINCIPAL, zorder=1)
    ax1.text(n_horn / 2 + 0.5, var_pct.max() * 1.18,
             f"{n_horn} componentes retenidos\n(criterio de Horn)",
             ha="center", fontsize=9, color=T.AZUL_PRINCIPAL, fontweight="bold")

    for bar, val in zip(bars[:n_horn], var_pct[:n_horn]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                 color=T.AZUL_PRINCIPAL, fontweight="bold")

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_scree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_scree.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Cargas de PC1-PC3
# ══════════════════════════════════════════════════════════════════════════════
def fig_loadings():
    load = pd.read_csv(PROC / "pca_loadings_2023.csv", index_col=0)
    load.index = [SHORT.get(i, i) for i in load.index]
    data = load.iloc[:, :3].copy()
    # ordenar por |PC1|
    data = data.reindex(data["PC1"].abs().sort_values(ascending=True).index)

    pct = pd.read_csv(PROC / "pca_autovalores_2023.csv")["var_explicada"].values[:3] * 100
    titles = [
        f"PC1 ({pct[0]:.1f} %)\nGradiente de desarrollo",
        f"PC2 ({pct[1]:.1f} %)\nEstructura productiva",
        f"PC3 ({pct[2]:.1f} %)\nApertura comercial",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharey=True)
    plt.subplots_adjust(wspace=0.08)

    for ax, col, title in zip(axes, data.columns, titles):
        vals = data[col].values
        clrs = [T.AZUL_PRINCIPAL if v >= 0 else T.ROJO_MEDIO for v in vals]
        bars = ax.barh(range(len(vals)), vals, color=clrs,
                       edgecolor="white", lw=0.5, height=0.75)
        ax.axvline(0, color=T.NEGRO, lw=0.8)
        ax.axvline(-0.5, ls=":", color=T.GRIS_MEDIO, lw=0.7)
        ax.axvline(0.5,  ls=":", color=T.GRIS_MEDIO, lw=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=T.AZUL_PRINCIPAL, pad=8)
        ax.set_xlim(-1.08, 1.08)
        ax.set_xlabel("Carga (ρ variable–componente)", fontsize=9)

        for bar, v in zip(bars, vals):
            offset = 0.04 if v >= 0 else -0.04
            ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=7.5, color=T.NEGRO)

    axes[0].set_yticks(range(len(data)))
    axes[0].set_yticklabels(data.index, fontsize=9)

    fig.suptitle(
        "Cargas factoriales de los tres primeros componentes principales",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_loadings.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Círculo de correlaciones (variable factor map)
# ══════════════════════════════════════════════════════════════════════════════
def fig_circulo_cargas():
    load = pd.read_csv(PROC / "pca_loadings_2023.csv", index_col=0)
    pct  = pd.read_csv(PROC / "pca_autovalores_2023.csv")["var_explicada"].values * 100

    fig, ax = plt.subplots(figsize=(9, 9))

    # Círculo unitario
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color=T.GRIS_MEDIO, lw=1.0, ls="-", zorder=1)
    # Círculo de calidad (r = 0.6)
    ax.plot(0.6 * np.cos(theta), 0.6 * np.sin(theta),
            color=T.GRIS_CLARO, lw=0.8, ls="--", zorder=1)

    ax.axhline(0, color=T.GRIS_MEDIO, lw=0.7, ls="--", zorder=1)
    ax.axvline(0, color=T.GRIS_MEDIO, lw=0.7, ls="--", zorder=1)

    for var in COLS:
        lx = load.loc[var, "PC1"]
        ly = load.loc[var, "PC2"]
        mag = np.sqrt(lx**2 + ly**2)

        # color: azul si PC1 positivo, rojo si negativo; intensidad por magnitud
        color = _lerp(T.AZUL_CLARO, T.AZUL_PRINCIPAL, mag) if lx >= 0 \
                else _lerp(T.ROJO_CLARO, T.ROJO_PRINCIPAL, mag)

        ax.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=1.8, mutation_scale=13),
            zorder=3,
        )

        # posición de la etiqueta: ligeramente más allá de la punta
        pad = 0.07
        ha  = "left" if lx >= 0 else "right"
        va  = "bottom" if ly >= 0 else "top"
        ax.text(lx + pad * np.sign(lx), ly + pad * np.sign(ly),
                SHORT[var], fontsize=9, ha=ha, va=va,
                color=color, fontweight="bold", zorder=4)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 — Gradiente de desarrollo ({pct[0]:.1f} %)", fontsize=11)
    ax.set_ylabel(f"PC2 — Estructura productiva ({pct[1]:.1f} %)", fontsize=11)
    ax.set_title(
        "Círculo de correlaciones — variables sobre PC1 y PC2",
        fontsize=13, fontweight="bold", pad=12,
    )

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=T.AZUL_PRINCIPAL, label="Correlación positiva con PC1"),
        Patch(facecolor=T.ROJO_PRINCIPAL,  label="Correlación negativa con PC1"),
        plt.Line2D([0],[0], ls="--", color=T.GRIS_CLARO, lw=1,
                   label="Umbral calidad (r = 0,6)"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="lower left",
              frameon=True, edgecolor=T.GRIS_MEDIO)

    fig.tight_layout()
    fig.savefig(OUT / "fig_circulo_cargas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_circulo_cargas.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5 & 6 · Clusters k=2 y k=3 sobre proyección PCA
# ══════════════════════════════════════════════════════════════════════════════
def fig_clusters(k=2):
    clusters = pd.read_csv(PROC / "clusters_2023.csv")
    pct = pd.read_csv(PROC / "pca_autovalores_2023.csv")["var_explicada"].values * 100
    col = f"cluster_k{k}"

    if k == 2:
        colors = T.CLUSTER2_COLORS
        labels_map = T.CLUSTER2_LABELS
    else:
        colors = T.CLUSTER3_COLORS
        labels_map = T.CLUSTER3_LABELS

    fig, ax = plt.subplots(figsize=(11, 8))

    for c, clr in colors.items():
        mask = clusters[col] == c
        n_c  = mask.sum()
        ax.scatter(clusters.loc[mask, "PC1"], clusters.loc[mask, "PC2"],
                   s=55, alpha=0.85, color=clr,
                   edgecolors="white", linewidths=0.5,
                   label=f"{labels_map[c]}  (n = {n_c})", zorder=3)

    ax.axhline(0, color=T.GRIS_MEDIO, lw=0.8, ls="--", zorder=1)
    ax.axvline(0, color=T.GRIS_MEDIO, lw=0.8, ls="--", zorder=1)
    ax.set_xlabel(f"PC1 — Gradiente de desarrollo ({pct[0]:.1f} %)", fontsize=11)
    ax.set_ylabel(f"PC2 — Estructura productiva ({pct[1]:.1f} %)", fontsize=11)
    ax.set_title(
        f"Clustering k-means (k={k}) sobre 14 variables — proyección en PC1-PC2, 2023",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=10, loc="lower right", title="Grupos detectados",
              title_fontsize=9)

    notables = {"CHN": "China", "USA": "EE.UU.", "IND": "India",
                "NGA": "Nigeria", "LBN": "Líbano", "SGP": "Singapur",
                "VEN": "Venezuela", "ETH": "Etiopía"}
    for iso, name in notables.items():
        row = clusters[clusters["countryiso3code"] == iso]
        if not row.empty:
            ax.text(row["PC1"].values[0] + 0.12, row["PC2"].values[0] + 0.12,
                    name, fontsize=7.5, color=T.NEGRO, style="italic")

    fig.tight_layout()
    fig.savefig(OUT / f"fig_clusters_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ fig_clusters_k{k}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7 · Panel de métricas para selección de k
# ══════════════════════════════════════════════════════════════════════════════
def fig_metricas_k():
    results = json.loads(
        (PROC / "clustering_resultados_2023.json").read_text(encoding="utf-8")
    )
    mk = pd.DataFrame(results["metricas_k"])
    k_opt = 2

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    specs = [
        ("silhouette",      "Silhouette\n(↑ mejor)",         axes[0, 0]),
        ("calinski_harabasz","Calinski-Harabasz\n(↑ mejor)", axes[0, 1]),
        ("davies_bouldin",  "Davies-Bouldin\n(↓ mejor)",     axes[1, 0]),
        ("inercia",         "Codo de inercia",               axes[1, 1]),
    ]

    for metric, title, ax in specs:
        y = mk[metric].values
        x = mk["k"].values
        ax.plot(x, y, "o-", color=T.AZUL_MEDIO, lw=2, ms=7, zorder=3)
        opt_y = mk.loc[mk["k"] == k_opt, metric].values
        if len(opt_y):
            ax.scatter([k_opt], opt_y, s=130, color=T.ROJO_MEDIO, zorder=5,
                       edgecolors="white", lw=1.5, label=f"k={k_opt} (elegido)")
        ax.set_title(title, fontsize=11, fontweight="bold", color=T.AZUL_PRINCIPAL)
        ax.set_xlabel("Número de clusters (k)", fontsize=10)
        ax.set_xticks(x)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)

    fig.suptitle(
        "Criterios de selección del número de clusters k-means (2023)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(OUT / "fig_metricas_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_metricas_k.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8 · Silhouette k=2
# ══════════════════════════════════════════════════════════════════════════════
def fig_silhouette_k2():
    from sklearn.metrics import silhouette_samples, silhouette_score

    clusters  = pd.read_csv(PROC / "clusters_2023.csv")
    scores_df = pd.read_csv(PROC / "pca_scores_2023.csv")

    # distancias en espacio PCA completo ≡ distancias en espacio estandarizado
    pc_cols = [c for c in scores_df.columns if c.startswith("PC") and c[2:].isdigit()]
    X      = scores_df[pc_cols].values
    labels = clusters["cluster_k2"].values

    sv  = silhouette_samples(X, labels)
    avg = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 0
    for c in [0, 1]:
        vals    = np.sort(sv[labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         color=T.CLUSTER2_COLORS[c], alpha=0.85,
                         label=f"{T.CLUSTER2_LABELS[c]}  (n = {(labels==c).sum()})")
        ax.text(-0.07, y_lower + len(vals) / 2, str(c), va="center",
                fontsize=11, color=T.CLUSTER2_COLORS[c], fontweight="bold")
        y_lower = y_upper + 8

    ax.axvline(avg, color=T.NEGRO, ls="--", lw=1.5,
               label=f"Silhouette promedio = {avg:.3f}")
    ax.set_xlabel("Coeficiente de silhouette", fontsize=11)
    ax.set_ylabel("Países (ordenados dentro de cada cluster)", fontsize=10)
    ax.set_title("Diagrama de silhouette — Clustering k=2 (2023)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(-0.25, 1.0)

    fig.tight_layout()
    fig.savefig(OUT / "fig_silhouette_k2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_silhouette_k2.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9 · Trayectorias — top 5 mejores y top 5 peores
# ══════════════════════════════════════════════════════════════════════════════
def fig_trayectorias():
    import matplotlib.font_manager as fm

    traj = pd.read_csv(PROC / "trayectorias.csv")
    pct  = pd.read_csv(PROC / "pca_autovalores_2023.csv")["var_explicada"].values * 100

    top_mejor = traj.nlargest(5,  "dPC1")
    top_peor  = traj.nsmallest(5, "dPC1")

    # Detectar Nunito; si no está, usar Arial
    try:
        fp = fm.findfont(fm.FontProperties(family="Nunito"))
        _fam = "Nunito" if "nunito" in fp.lower() else "Arial"
    except Exception:
        _fam = "Arial"

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("white")

    def _draw_country(row, color):
        x0, y0 = row.PC1_e, row.PC2_e
        x1, y1 = row.PC1_m, row.PC2_m
        # punto gris en 2005
        ax.scatter(x0, y0, s=55, color=T.GRIS_MEDIO, zorder=3,
                   edgecolors="white", lw=0.8)
        # flecha
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=2.0, mutation_scale=14,
                            connectionstyle="arc3,rad=0.0"),
            zorder=4,
        )
        # punto coloreado en 2023
        ax.scatter(x1, y1, s=75, color=color, zorder=5,
                   edgecolors="white", lw=1.0)
        # etiqueta país
        offset_x = 0.15
        offset_y = 0.15
        ax.text(x1 + offset_x, y1 + offset_y, row.country,
                fontsize=9, color=color, fontweight="bold",
                fontfamily=_fam, zorder=6)

    for _, r in top_mejor.iterrows():
        _draw_country(r, T.VERDE_PRINCIPAL)
    for _, r in top_peor.iterrows():
        _draw_country(r, T.ROJO_PRINCIPAL)

    ax.axhline(0, color=T.GRIS_CLARO, lw=1.0, ls="--", zorder=1)
    ax.axvline(0, color=T.GRIS_CLARO, lw=1.0, ls="--", zorder=1)
    ax.set_xlabel(f"PC1 — Gradiente de desarrollo ({pct[0]:.1f} %)", fontsize=11)
    ax.set_ylabel(f"PC2 — Estructura productiva ({pct[1]:.1f} %)", fontsize=11)
    ax.set_title(
        "Mayores avances y retrocesos en el gradiente de desarrollo — 2005 a 2023",
        fontsize=13, fontweight="bold", pad=12,
    )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=T.GRIS_MEDIO,
               ms=8, label="Posición en 2005"),
        Line2D([0], [0], marker=">", color=T.VERDE_PRINCIPAL, lw=2,
               ms=8, label="Top 5 — mayor avance"),
        Line2D([0], [0], marker=">", color=T.ROJO_PRINCIPAL, lw=2,
               ms=8, label="Top 5 — mayor retroceso"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="lower right",
              frameon=True, edgecolor=T.GRIS_MEDIO)

    fig.tight_layout()
    fig.savefig(OUT / "fig_trayectorias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_trayectorias.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10A · Transición — diagrama de flujo tipo Sankey
# ══════════════════════════════════════════════════════════════════════════════
def fig_transicion_sankey():
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Rectangle

    # Datos fijos del análisis
    n_dev_dev  = 66   # en desarrollo → en desarrollo
    n_dev_des  = 40   # en desarrollo → desarrollado  (graduaron)
    n_des_des  = 77   # desarrollado  → desarrollado
    total      = n_dev_dev + n_dev_des + n_des_des  # 183

    scale = 10 / total  # 10 unidades de alto para el eje y

    # Posiciones en el eje y (apiladas sin solapamiento)
    # Barra izquierda (2005) — de abajo hacia arriba:
    #   [0]  desarrollado que se queda  (77)
    #   [1]  en-dev que gradúa          (40)
    #   [2]  en-dev que se queda        (66)
    L_des_lo  = 0
    L_des_hi  = n_des_des * scale
    L_grad_lo = L_des_hi
    L_grad_hi = L_grad_lo + n_dev_des * scale
    L_dev_lo  = L_grad_hi
    L_dev_hi  = 10.0

    # Barra derecha (2023) — misma distribución para flujos rectos
    R_des_lo  = 0
    R_des_hi  = n_des_des * scale
    R_grad_lo = R_des_hi
    R_grad_hi = R_grad_lo + n_dev_des * scale
    R_dev_lo  = R_grad_hi
    R_dev_hi  = 10.0

    X_L0, X_L1 = 0.0, 1.2   # barra izquierda
    X_R0, X_R1 = 4.8, 6.0   # barra derecha
    XM = (X_L1 + X_R0) / 2  # punto medio para curvas bezier

    def bezier_flow(ax, yl_lo, yl_hi, yr_lo, yr_hi, color, alpha=0.55):
        verts = [
            (X_L1, yl_hi),
            (XM,   yl_hi), (XM, yr_hi), (X_R0, yr_hi),
            (X_R0, yr_lo),
            (XM,   yr_lo), (XM, yl_lo), (X_L1, yl_lo),
            (X_L1, yl_hi),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(Path(verts, codes),
                               facecolor=color, alpha=alpha, edgecolor="none"))

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.2, 11.5)
    ax.axis("off")

    # Barras izquierda
    ax.add_patch(Rectangle((X_L0, L_des_lo),  X_L1-X_L0, L_des_hi-L_des_lo,
                            facecolor=T.AZUL_PRINCIPAL,  edgecolor="white", lw=1.5))
    ax.add_patch(Rectangle((X_L0, L_grad_lo), X_L1-X_L0, L_grad_hi-L_grad_lo,
                            facecolor=T.VERDE_MEDIO,     edgecolor="white", lw=1.5))
    ax.add_patch(Rectangle((X_L0, L_dev_lo),  X_L1-X_L0, L_dev_hi-L_dev_lo,
                            facecolor=T.ROJO_MEDIO,      edgecolor="white", lw=1.5))

    # Barras derecha
    ax.add_patch(Rectangle((X_R0, R_des_lo),  X_R1-X_R0, R_des_hi-R_des_lo,
                            facecolor=T.AZUL_PRINCIPAL,  edgecolor="white", lw=1.5))
    ax.add_patch(Rectangle((X_R0, R_grad_lo), X_R1-X_R0, R_grad_hi-R_grad_lo,
                            facecolor=T.VERDE_MEDIO,     edgecolor="white", lw=1.5))
    ax.add_patch(Rectangle((X_R0, R_dev_lo),  X_R1-X_R0, R_dev_hi-R_dev_lo,
                            facecolor=T.ROJO_MEDIO,      edgecolor="white", lw=1.5))

    # Flujos
    bezier_flow(ax, L_des_lo,  L_des_hi,  R_des_lo,  R_des_hi,  T.AZUL_PRINCIPAL)
    bezier_flow(ax, L_grad_lo, L_grad_hi, R_grad_lo, R_grad_hi, T.VERDE_MEDIO)
    bezier_flow(ax, L_dev_lo,  L_dev_hi,  R_dev_lo,  R_dev_hi,  T.ROJO_MEDIO)

    # Etiquetas en las barras
    kw = dict(ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.text((X_L0+X_L1)/2, (L_des_lo+L_des_hi)/2,  f"{n_des_des}", **kw)
    ax.text((X_L0+X_L1)/2, (L_grad_lo+L_grad_hi)/2, f"{n_dev_des}", **kw)
    ax.text((X_L0+X_L1)/2, (L_dev_lo+L_dev_hi)/2,  f"{n_dev_dev}", **kw)
    ax.text((X_R0+X_R1)/2, (R_des_lo+R_des_hi)/2,  f"{n_des_des}", **kw)
    ax.text((X_R0+X_R1)/2, (R_grad_lo+R_grad_hi)/2, f"{n_dev_des}", **kw)
    ax.text((X_R0+X_R1)/2, (R_dev_lo+R_dev_hi)/2,  f"{n_dev_dev}", **kw)

    # Títulos de columnas
    tx = dict(ha="center", fontsize=12, fontweight="bold", color=T.NEGRO)
    ax.text((X_L0+X_L1)/2, 10.6, "2005", **tx)
    ax.text((X_R0+X_R1)/2, 10.6, "2023", **tx)

    # Leyenda lateral
    ly = 9.5
    for color, label in [
        (T.AZUL_PRINCIPAL, f"Permaneció desarrollado  ({n_des_des} países)"),
        (T.VERDE_MEDIO,    f"Graduó al grupo desarrollado  ({n_dev_des} países)"),
        (T.ROJO_MEDIO,     f"Permaneció en desarrollo  ({n_dev_dev} países)"),
    ]:
        ax.add_patch(Rectangle((6.3, ly-0.2), 0.25, 0.4,
                               facecolor=color, edgecolor="none"))
        ax.text(6.65, ly, label, va="center", fontsize=9, color=T.NEGRO)
        ly -= 1.1

    ax.set_title("Flujo de países entre grupos — 2005 a 2023",
                 fontsize=14, fontweight="bold", pad=10, x=0.42)

    fig.tight_layout()
    fig.savefig(OUT / "fig_transicion_sankey.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_transicion_sankey.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10B · Transición — waffle chart comparativo 2005 vs 2023
# ══════════════════════════════════════════════════════════════════════════════
def fig_transicion_waffle():
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patches import Patch

    # Distribución por año
    years = {
        "2005": {"En desarrollo": 106, "Desarrollado": 77},
        "2023": {"En desarrollo": 66,  "Desarrollado": 117},
    }
    total  = 183
    COLS_W = 15   # columnas del waffle
    ROWS_W = 13   # filas   → 195 celdas; 12 vacías al final
    SIZE   = 0.82
    GAP    = 0.12
    STEP   = SIZE + GAP

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.patch.set_facecolor("white")

    for ax, (year, dist) in zip(axes, years.items()):
        ax.set_facecolor("white")
        ax.set_xlim(-0.3, COLS_W * STEP)
        ax.set_ylim(-0.3, ROWS_W * STEP + 0.8)
        ax.axis("off")

        n_dev = dist["En desarrollo"]
        n_des = dist["Desarrollado"]
        seq   = [T.ROJO_MEDIO] * n_dev + [T.AZUL_PRINCIPAL] * n_des + \
                [T.GRIS_CLARO] * (COLS_W * ROWS_W - total)

        for idx, color in enumerate(seq):
            col = idx % COLS_W
            row = idx // COLS_W
            x   = col * STEP
            y   = (ROWS_W - 1 - row) * STEP
            ec  = "white" if color != T.GRIS_CLARO else "#e0e0e0"
            ax.add_patch(FancyBboxPatch(
                (x, y), SIZE, SIZE,
                boxstyle="round,pad=0.06",
                facecolor=color, edgecolor=ec, lw=0.8,
            ))

        ax.set_title(year, fontsize=16, fontweight="bold",
                     color=T.AZUL_PRINCIPAL, pad=10)
        ax.text(COLS_W * STEP / 2, -0.1,
                f"En desarrollo: {n_dev}  |  Desarrollado: {n_des}",
                ha="center", va="top", fontsize=9, color=T.NEGRO)

    legend_handles = [
        Patch(facecolor=T.ROJO_MEDIO,      label="En desarrollo"),
        Patch(facecolor=T.AZUL_PRINCIPAL,  label="Desarrollado"),
        Patch(facecolor=T.GRIS_CLARO,      edgecolor="#e0e0e0", label="Sin datos / vacío"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor=T.GRIS_MEDIO)

    fig.suptitle(
        "Distribución de países por grupo — cada cuadro = 1 país",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_transicion_waffle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_transicion_waffle.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11 · Descomposición del avance en PC1
# ══════════════════════════════════════════════════════════════════════════════
def fig_decomposicion():
    df = pd.read_csv(PROC / "descomposicion_dPC1.csv")
    df = df.sort_values("pct_del_total", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    clrs = [T.VERDE_PRINCIPAL if v >= 0 else T.ROJO_MEDIO
            for v in df["pct_del_total"].values]
    bars = ax.barh(range(len(df)), df["pct_del_total"].values,
                   color=clrs, edgecolor="white", lw=0.5, height=0.68)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["variable"].values, fontsize=9)
    ax.axvline(0, color=T.NEGRO, lw=0.8)
    ax.set_xlabel("Contribución al avance promedio en PC1 (%)", fontsize=11)
    ax.set_title(
        "¿Qué explica el avance de los países entre 2005 y 2023?",
        fontsize=13, fontweight="bold", pad=10,
    )

    for bar, val in zip(bars, df["pct_del_total"].values):
        x = bar.get_width()
        ax.text(
            x + (0.5 if x >= 0 else -0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} %", va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8.5, fontweight="bold",
            color=T.VERDE_PRINCIPAL if x >= 0 else T.ROJO_MEDIO,
        )

    fig.tight_layout()
    fig.savefig(OUT / "fig_decomposicion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_decomposicion.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Generando figuras en: {OUT}\n")
    fig_correlacion()
    fig_scree()
    fig_loadings()
    fig_circulo_cargas()
    fig_clusters(2)
    fig_clusters(3)
    fig_metricas_k()
    fig_silhouette_k2()
    fig_trayectorias()
    fig_transicion_sankey()
    fig_transicion_waffle()
    fig_decomposicion()
    print(f"\nListo. {len(list(OUT.glob('*.png')))} figuras generadas en {OUT}")


if __name__ == "__main__":
    main()
