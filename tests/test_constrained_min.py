import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from unittest import TestCase

import examples
import sys

sys.path.insert(1, "src")
from src.constrained_min import interior_pt


def make_objective_figure(df: pd.DataFrame, title: str):
    """Plot objective value versus outer-iteration number."""
    fig, ax = plt.subplots()
    ax.plot(df.index, df["f_x"], "-o")
    ax.set_xlabel("outer iteration k")
    ax.set_ylabel("f(xₖ)")
    ax.set_title(title)
    ax.grid(True, ls="--", lw=0.5)
    return fig


def draw_3d_path_figure(df: pd.DataFrame, title: str = "Central path"):
    """
    Feasible triangle (x+y+z=1, x,y,z≥0) + central-path points in 3-D.
    `df` must contain x1,x2,x3 columns.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # triangle
    verts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ax.add_collection3d(
        Poly3DCollection([verts], alpha=0.25, facecolor="tab:blue", edgecolor="k")
    )

    # central path
    ax.plot(df["x1"], df["x2"], df["f_x"], "-o", lw=1.5, ms=5, label="central path")
    ax.plot(
        df["x1"].iloc[-1],
        df["x2"].iloc[-1],
        df["f_x"].iloc[-1],
        "r*",
        ms=12,
        label="final sol.",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_feasible_region_and_path_2d(df, ineq_constraints, title="Central path"):

    grid_res = 400
    padding = 0.1
    x_lo, x_hi = df["x1"].min(), df["x1"].max()
    y_lo, y_hi = df["x2"].min(), df["x2"].max()
    x_pad = (x_hi - x_lo) * padding or 1.0
    y_pad = (y_hi - y_lo) * padding or 1.0
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad

    xx, yy = np.meshgrid(
        np.linspace(x_lo, x_hi, grid_res),
        np.linspace(y_lo, y_hi, grid_res),
    )
    grid_pts = np.array([xx, yy])
    feas = np.ones_like(xx, dtype=bool)
    for g in ineq_constraints:
        feas &= g(grid_pts)[0] <= 0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        feas.astype(float),
        extent=[x_lo, x_hi, y_lo, y_hi],
        origin="lower",
        cmap="Greys",
        alpha=0.3,
        aspect="auto",
    )
    ax.plot(df["x1"], df["x2"], "-o", ms=5, lw=1.5, label="central path")
    ax.plot(df["x1"].iloc[-1], df["x2"].iloc[-1], "r*", ms=12, label="final sol.")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls="--", lw=0.5)
    return fig


class TestInteriorPoint(TestCase):
    _figures = []

    def test_qp(self):
        f_obj, g_list, A, b, x0 = examples.get_quadratic_probel()
        sol, trace = interior_pt(f_obj, g_list, A, b, x0, verbose=False)

        df = pd.DataFrame(trace)

        self._figures.append(
            draw_3d_path_figure(df, title="QP - feasible triangle & central path")
        )

        self._figures.append(
            make_objective_figure(df, title="QP - objective vs outer iteration")
        )

        final_f = f_obj(sol)[0]
        final_g = np.array([g(sol)[0] for g in g_list])
        final_h = (A @ sol - b) if A.size else np.array([])
        print(
            f"\n[QP]  f(x*) = {final_f:.6f},  "
            f"max g_i(x*) = {final_g.max():.2e},  "
            f"|Ax-b| = {np.linalg.norm(final_h):.2e}"
        )

    def test_lp(self):
        f_obj, g_list, A, b, x0 = examples.get_linear_problem()
        sol, trace = interior_pt(f_obj, g_list, A, b, x0, verbose=False)

        df = pd.DataFrame(trace)
        self._figures.append(
            plot_feasible_region_and_path_2d(
                df, g_list, title="LP - feasible region & central path"
            )
        )

        self._figures.append(
            make_objective_figure(df, title="LP - objective vs outer iteration")
        )

        final_f = f_obj(sol)[0]
        final_g = np.array([g(sol)[0] for g in g_list])
        final_h = (A @ sol - b) if A.size else np.array([])
        print(
            f"\n[LP]  f(x*) = {final_f:.6f},  "
            f"max g_i(x*) = {final_g.max():.2e},  "
            f"|Ax-b| = {np.linalg.norm(final_h):.2e}"
        )

    @classmethod
    def tearDownClass(cls):
        for fig in cls._figures:
            fig.tight_layout()
        plt.show()
