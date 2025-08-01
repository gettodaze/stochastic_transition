"""
Copyright 2022 Toshitake Asabuki

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file recept in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# coding: UTF-8
from __future__ import division
import numpy as np
import pylab as pl
import numba
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import seaborn as sns
from matplotlib import gridspec
from warnings import simplefilter
import sklearn.decomposition

from util import Paths, get_activation_function
import util


def configure():
    sns.set_theme(
        context="paper", style="white", font_scale=1.5, rc={"lines.linewidth": 2.5}
    )
    sns.set_palette("muted")

    simplefilter(action="ignore", category=FutureWarning)
    util.configure_mpl()
    mpl.rcParams["axes.xmargin"] = 0
    mpl.rcParams["axes.ymargin"] = 0


def run_simulation():
    W_rec = np.loadtxt(Paths.INIT_WEIGHTS_E, delimiter=",")
    W_I = np.loadtxt(Paths.INIT_WEIGHTS_I, delimiter=",")

    pat_list = [0, 1, 2]

    # transition_prob = [[0,1/2,1/2,0,0],[1/3,0,1/3,1/3,0],[1/3,0,0,1/3,1/3],[0,1/2,1/2,0,0],[0,0,1,0,0]]
    # pat_list = [0, 1, 2, 3,4]
    N = len(W_rec[0, :])
    N_E = int(N * 1 / 2)

    n_pat = len(set(pat_list))

    np.random.seed()

    activation_f = get_activation_function()

    vfunc = np.vectorize(activation_f)
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    pl.plot(np.arange(-1, 2, 0.01), vfunc(np.arange(-1, 2, 0.01)))
    fig.subplots_adjust(bottom=0.25, left=0.15)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.spines["right"].set_color("none")
    ax1.spines["top"].set_color("none")
    plt.savefig("sigmoid.pdf", format="pdf", dpi=350)

    raeter_size = 1.1
    dt = 1
    test_len = int(100 * 1000 / dt)
    test_len2 = int(100 * 1000 / dt)

    gain = 1
    y = np.random.rand(N)

    f = activation_f(y)

    tau_m = 15
    tau_syn = 5

    100 * n_pat
    pat_color = sns.color_palette("Set2")
    max_rate = 0.05

    I_syn = 0 * np.random.rand(N) / tau_m / tau_syn
    PSP = np.zeros(N)

    tau_max = 10 * 1000
    max_trace = np.zeros(N) + 1

    id_rec = np.zeros(N, dtype=bool)

    """
    Spontaneous
    """

    dt = 1

    gain = 1

    y = np.random.rand(N)

    f = activation_f(y)

    tau_m = 15
    tau_syn = 5

    100 * n_pat

    I_syn = np.random.rand(N) / tau_m / tau_syn
    PSP = np.zeros(N)

    id_rec = np.zeros(N, dtype=bool)

    np.zeros(N)
    np.zeros(N) + 1

    pat_mat_supervise = np.zeros((N, n_pat))
    for i in range(n_pat):
        pat_mat_supervise[int(i * N_E / n_pat) : int((i + 1) * N_E / n_pat), i] = 0.5

    max_trace = np.zeros(N) + 1
    np.zeros(N) + 10 ** (-5)

    np.zeros(N) + 1

    tau_max = 10 * 1000

    const_noise = np.zeros(N)
    const_noise[N_E:] = 0
    const_noise[0:N_E] = 0.3

    """
    Spont1
    """
    pat_start_list_blank = []
    for i in range(n_pat):
        pat_start_list_blank.append([])

    for i in tqdm(range(test_len), desc="[spont1]"):
        I_syn = (1.0 - dt / tau_syn) * I_syn
        I_syn[id_rec] += 1 / tau_m / tau_syn
        PSP = (1.0 - dt / tau_m) * PSP + I_syn * 25

        rec_term = np.dot(W_rec, PSP)
        M_term = np.dot(W_I, PSP)
        x = rec_term - M_term + const_noise

        max_trace = (1.0 - dt / tau_max) * max_trace
        max_trace[max_trace < x * gain] = x[max_trace < x * gain]
        max_trace[N_E:] = 2

        y = 2 * x / max_trace

        f = activation_f(y)

        id_rec = np.random.rand(N) < f * dt * max_rate

    """
    Spont2
    """

    x_list = np.zeros((N, test_len2))
    id_list_E = np.zeros((test_len2, N_E), dtype=bool)
    f_list = np.zeros((N_E, test_len2))

    filtered_spikes = np.zeros(N_E)
    filtered_spikes_list = np.zeros((N_E, test_len2))
    tau_filter = 100

    EPSCs = np.zeros((n_pat, test_len2))
    IPSCs = np.zeros((n_pat, test_len2))

    for i in tqdm(range(test_len2), desc="[spont2]"):
        I_syn = (1.0 - dt / tau_syn) * I_syn
        I_syn[id_rec] += 1 / tau_m / tau_syn
        PSP = (1.0 - dt / tau_m) * PSP + I_syn * 25
        rec_term = np.dot(W_rec, PSP)
        M_term = np.dot(W_I, PSP)
        x = rec_term - M_term + const_noise

        max_trace = (1.0 - dt / tau_max) * max_trace
        max_trace[max_trace < x * gain] = x[max_trace < x * gain]
        max_trace[N_E:] = 2

        y = 2 * x / max_trace

        f = activation_f(y)

        for mm in range(n_pat):
            EPSCs[mm, i] = np.mean(
                np.dot(
                    W_rec[
                        int(mm) * int(N_E / n_pat) : int(mm + 1) * int(N_E / n_pat), :
                    ],
                    PSP,
                )
            )
        for mm in range(n_pat):
            IPSCs[mm, i] = np.mean(
                np.dot(
                    W_I[int(mm) * int(N_E / n_pat) : int(mm + 1) * int(N_E / n_pat), :],
                    PSP,
                )
            )

        x_list[:, i] = 2 * y + const_noise

        id_rec = np.random.rand(N) < f * dt * max_rate

        filtered_spikes = (1 - dt / tau_filter) * filtered_spikes + id_rec[0:N_E]

        f_list[:, i] = f[0:N_E]
        filtered_spikes_list[:, i] = filtered_spikes
        id_list_E[i, :] = id_rec[0:N_E]

    print(1)

    pl.figure()

    fig = plt.figure(figsize=(4, 4))

    mean_std_vol = np.zeros(4)
    mean_std_vol[0] = np.mean(np.mean(x_list[0:N_E, :], axis=1))
    mean_std_vol[1] = np.mean(np.std(x_list[0:N_E, :], axis=1))
    mean_std_vol[2] = np.mean(np.mean(x_list[N_E:, :], axis=1))
    mean_std_vol[3] = np.mean(np.std(x_list[N_E:, :], axis=1))

    np.savetxt("mean_std_vol.txt", mean_std_vol, delimiter=",")

    mean_firing_rate = np.zeros((n_pat, test_len2))
    for i in range(n_pat):
        mean_firing_rate[i, :] = np.mean(
            f_list[i * int(N_E / n_pat) : (i + 1) * int(N_E / n_pat), 0:test_len2],
            axis=0,
        )

    np.savetxt(Paths.ASSEMBLY_ACVITITIES, mean_firing_rate, delimiter=",")

    firing_mat = f_list[0:N_E, 0:2000].T

    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(firing_mat)

    # print(np.sum(pca.explained_variance_ratio_))
    np.savetxt("variance_explained.txt", pca.explained_variance_ratio_, delimiter=",")

    mean_firing_rate_filtered_spikes = np.zeros((n_pat, test_len2))
    for i in range(n_pat):
        mean_firing_rate_filtered_spikes[i, :] = np.mean(
            filtered_spikes_list[
                i * int(N_E / n_pat) : (i + 1) * int(N_E / n_pat), 0:test_len2
            ],
            axis=0,
        )

    np.savetxt(
        "assembly_acvitities_filtered_spikes.txt",
        mean_firing_rate_filtered_spikes,
        delimiter=",",
    )

    print("raster_start")
    pl.figure()
    plot_len2 = 2 * 1000
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.tick_params(direction="in", width=1, length=2)
    for nn in range(n_pat):
        tspk, nspk = pl.nonzero(
            id_list_E[
                test_len2 - plot_len2 :,
                nn * int(N_E / n_pat) : (nn + 1) * int(N_E / n_pat),
            ]
            == 1
        )
        nspk += nn * int(N_E / n_pat)
        plt.plot(
            tspk,
            nspk,
            c=pat_color[nn],
            marker="o",
            lw=0,
            markersize=0.8,
            fillstyle="full",
            markeredgewidth=0.0,
        )

    plt.ylabel("Neurons" + r"$\ \times \ 10^2$ ", fontsize=10)
    plt.xlabel("Time (s)", fontsize=10)
    pl.xlim([0, plot_len2])
    # plt.xticks([0, 2*1000, 4*1000, 6*1000, 8*1000, 9.9*1000],
    # ['0', '2', '4', '6', '8', '10'], fontsize=10)
    # plt.yticks([100, 200, 300, 400, 499],
    # ['1', '2', '3', '4', '5'], fontsize=10)
    # plt.xticks(time_axis,np.arange(second_spont+1),fontsize=11)

    fig.subplots_adjust(bottom=0.25, left=0.15)

    # ax1.xaxis.set_ticks_position('bottom')
    # ax1.yaxis.set_ticks_position('left')
    # ax1.spines['right'].set_color('none')#
    # ax1.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    plt.savefig("raster_spont.pdf", dpi=350)

    print("raster_finish")
    # pl.plot()

    np.savetxt("EPSCs.txt", EPSCs, delimiter=",")
    np.savetxt("IPSCs.txt", IPSCs, delimiter=",")

    pl.figure()
    plot_len2 = 2 * 1000
    fig = plt.figure(figsize=(4, 3))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    ax = plt.subplot(gs[1])  # fig.add_subplot(212)
    ax.tick_params(direction="in", width=1, length=2)
    for nn in range(n_pat):
        tspk, nspk = pl.nonzero(
            id_list_E[
                test_len2 - plot_len2 :,
                nn * int(N_E / n_pat) : (nn + 1) * int(N_E / n_pat),
            ]
            == 1
        )
        nspk += nn * int(N_E / n_pat)
        plt.plot(
            tspk,
            nspk,
            c=pat_color[nn],
            marker="o",
            lw=0,
            markersize=raeter_size,
            fillstyle="full",
            markeredgewidth=0.0,
        )

    plt.ylabel("Neurons" + r"$\ \times \ 10^2$ ", fontsize=10)
    plt.xlabel("Time (s)", fontsize=10)
    pl.xlim([0, plot_len2])
    # plt.xticks([0, 2*1000, 4*1000, 6*1000, 8*1000, 9.9*1000],
    # ['0', '2', '4', '6', '8', '10'], fontsize=10)
    # plt.yticks([100, 200, 300, 400, 499],
    # ['1', '2', '3', '4', '5'], fontsize=10)
    # plt.xticks(time_axis,np.arange(second_spont+1),fontsize=11)

    fig.subplots_adjust(bottom=0.25, left=0.15)

    # ax1.xaxis.set_ticks_position('bottom')
    # ax1.yaxis.set_ticks_position('left')
    # ax1.spines['right'].set_color('none')#
    # ax1.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax = plt.subplot(gs[0])  # fig.add_subplot(211)
    ax.tick_params(direction="in", width=0, length=0)
    for i in range(n_pat):
        pl.plot(mean_firing_rate[i, test_len2 - plot_len2 :], c=pat_color[i], lw=1.3)
        # plt.hlines(activation_threshold[i],0,50*10,color=pat_color[i])
        pl.ylim([0, 1])
    plt.xticks([])
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    plt.savefig("raster_spont.pdf", dpi=350)


if __name__ == "__main__":
    configure()
    run_simulation()
