import matplotlib as mpl
import numba
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class Paths:
    ASSEMBLY_ACVITITIES = DATA_DIR / "assembly_acvitities.txt"
    FMT_WEIGHTS_E_FN = "W_E_{sim_n}.txt"
    FMT_WEIGHTS_I_FN = "W_I_{sim_n}.txt"

    @classmethod
    def get_path_weights_e(cls, sim_n: int) -> Path:
        return DATA_DIR / cls.FMT_WEIGHTS_E_FN.format(sim_n=sim_n)

    @classmethod
    def get_path_weights_i(cls, sim_n: int) -> Path:
        return DATA_DIR / cls.FMT_WEIGHTS_I_FN.format(sim_n=sim_n)


def configure_mpl():
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["pdf.fonttype"] = 42
    # params = {
    #     "backend": "ps",
    #     "axes.labelsize": 11,
    #     "text.fontsize": 11,
    #     "legend.fontsize": 11,
    #     "xtick.labelsize": 11,
    #     "ytick.labelsize": 11,
    #     "text.usetex": False,
    #     "figure.figsize": [10 / 2.54, 6 / 2.54],
    # }


def get_activation_function():
    alpha = 1
    theta = 1
    beta = 5

    @numba.njit(fastmath=True, nogil=True)
    def g(x):
        ans = 1 / (1 + alpha * np.exp(beta * (-x + theta)))
        return ans

    return g
