import matplotlib as mpl


class Paths:
    ASSEMBLY_ACVITITIES = "assembly_acvitities.txt"
    INIT_WEIGHTS_E = "W_E_0.txt"
    INIT_WEIGHTS_I = "W_I_0.txt"


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
