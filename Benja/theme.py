"""
Paleta institucional y tema matplotlib compartido para la presentación mejorada.
"""
from matplotlib import rcParams

# ── Paleta de colores ────────────────────────────────────────────────────── #
AZUL_PRINCIPAL  = "#174EA6"
AZUL_MEDIO      = "#4285F4"
AZUL_CLARO      = "#D2E3FC"
ROJO_PRINCIPAL  = "#A50E0E"
ROJO_MEDIO      = "#EA4335"
ROJO_CLARO      = "#FAD2CF"
NARANJA         = "#E37400"
AMARILLO        = "#FBBC04"
AMARILLO_CLARO  = "#FEEFC3"
VERDE_PRINCIPAL = "#0D652D"
VERDE_MEDIO     = "#34A853"
VERDE_CLARO     = "#CEEAD6"
GRIS_CLARO      = "#F1F3F4"
GRIS_MEDIO      = "#9AA0A6"
NEGRO           = "#202124"

# ── Colores por cluster ──────────────────────────────────────────────────── #
CLUSTER2_COLORS  = {0: ROJO_MEDIO,     1: AZUL_PRINCIPAL}
CLUSTER2_LABELS  = {0: "En desarrollo", 1: "Desarrollado"}
CLUSTER3_COLORS  = {0: ROJO_MEDIO, 1: NARANJA, 2: AZUL_PRINCIPAL}
CLUSTER3_LABELS  = {0: "Bajo desarrollo", 1: "Emergente", 2: "Desarrollado"}

# ── Colores por nivel de ingreso ─────────────────────────────────────────── #
INCOME_COLORS = {
    "Low income":          ROJO_PRINCIPAL,
    "Lower middle income": NARANJA,
    "Upper middle income": AMARILLO,
    "High income":         AZUL_PRINCIPAL,
}
INCOME_LABELS = {
    "Low income":          "Bajo",
    "Lower middle income": "Medio-bajo",
    "Upper middle income": "Medio-alto",
    "High income":         "Alto",
}


def apply_theme():
    """Aplica el tema global de matplotlib al proceso actual."""
    rcParams.update({
        "font.family":        "Arial",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "grid.color":         GRIS_CLARO,
        "grid.linewidth":     0.8,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    10,
        "legend.frameon":     False,
        "figure.dpi":         150,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })
