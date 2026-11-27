"""Example populations used in tests and demos (formerly LIBRARY). NPL thickness on z."""

from .population import Population

EXAMPLES: dict[str, Population] = {
    "cube_6": Population("isotropic", [6, 6, 6], [0.10] * 3, label="cube 6 uc"),
    "cube_10": Population("isotropic", [10, 10, 10], [0.10] * 3, label="cube 10 uc"),
    "cube_15": Population("isotropic", [15, 15, 15], [0.10] * 3, label="cube 15 uc"),
    "cube_25": Population("isotropic", [25, 25, 25], [0.10] * 3, label="cube 25 uc"),
    "npl_2ML_L30": Population("npl", [30, 30, 2], [0.10, 0.10, 0], label="2 ML, L 30 uc"),
    "npl_3ML_L20": Population("npl", [20, 20, 3], [0.10, 0.10, 0], label="3 ML, L 20 uc"),
    "npl_4ML_L40": Population("npl", [40, 40, 4], [0.10, 0.10, 0], label="4 ML, L 40 uc"),
    "npl_3ML_rect": Population("npl", [15, 30, 3], [0.10, 0.10, 0], label="3 ML, 15x30 uc"),
    "edge_narrow": Population("isotropic", [10, 10, 10], [0.05] * 3, label="cube 10 uc, 5 %"),
    "edge_broad": Population("isotropic", [10, 10, 10], [0.20] * 3, label="cube 10 uc, 20 %"),
    "edge_large": Population("isotropic", [45, 45, 45], [0.05] * 3, label="cube 45 uc (grid edge)"),
}
