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
import matplotlib.pyplot as plt
import matplotlib as mpl


from warnings import simplefilter
from sklearn.preprocessing import normalize
import seaborn as sns
from scipy.optimize import curve_fit

from util import Paths

simplefilter(action="ignore", category=FutureWarning)
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
params = {
    "backend": "ps",
    "axes.labelsize": 11,
    "text.fontsize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "text.usetex": False,
    "figure.figsize": [10 / 2.54, 6 / 2.54],
}


def exp_fit(x, a, b):
    y = a * np.exp(-b * x)
    return y


transition_prob = [[0, 0.1, 0.9], [0.1, 0, 0.9], [0.5, 0.5, 0]]
pat_list = [0, 1, 2]
# transition_prob = [[0,1/2,1/2,0,0],[1/3,0,1/3,1/3,0],[1/3,0,0,1/3,1/3],[0,1/2,1/2,0,0],[0,0,1,0,0]]
# pat_list = [0, 1, 2, 3,4]
# os.chdir('simulation_9')
"""
transition_prob = np.zeros((15, 15))
transition_prob[0, [x - 1 for x in [15, 2, 3, 4]]] = 1
transition_prob[1, [x - 1 for x in [1, 3, 4, 5]]] = 1
transition_prob[2, [x - 1 for x in [1, 2, 4, 5]]] = 1
transition_prob[3, [x - 1 for x in [1, 2, 3, 5]]] = 1
transition_prob[4, [x - 1 for x in [2, 3, 4, 6]]] = 1

transition_prob[5, [x - 1 for x in [5, 7, 8, 9]]] = 1
transition_prob[6, [x - 1 for x in [6, 8, 9, 10]]] = 1
transition_prob[7, [x - 1 for x in [6, 7, 9, 10]]] = 1
transition_prob[8, [x - 1 for x in [6, 7, 8, 10]]] = 1
transition_prob[9, [x - 1 for x in [7, 8, 9, 11]]] = 1

transition_prob[10, [x - 1 for x in [10, 12, 13, 14]]] = 1
transition_prob[11, [x - 1 for x in [11, 13, 14, 15]]] = 1
transition_prob[12, [x - 1 for x in [11, 12, 14, 15]]] = 1
transition_prob[13, [x - 1 for x in [11, 12, 13, 15]]] = 1
transition_prob[14, [x - 1 for x in [1, 12, 13, 14]]] = 1
transition_prob = normalize(transition_prob, axis=1, norm='l1')
pat_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
"""
mean_firing_rate = np.loadtxt(Paths.ASSEMBLY_ACVITITIES, delimiter=",")

activation_threshold = (
    np.max(mean_firing_rate, axis=1) * 0.5
)  # +np.std(mean_firing_rate, axis=1)*3#+0.5#+ 1 * np.std(mean_firing_rate, axis=1)
print(activation_threshold)
plot_len = len(mean_firing_rate[0, :])
# print(plot_len)
n_pat = len(mean_firing_rate[:, 0])
pat_color_ = sns.color_palette("Paired")
pat_color = []  # plt.cm.get_cmap('tab20').colors#sns.color_palette("Set2")
for n in range(15):
    if n == 0 or n == 4:
        pat_color.append(pat_color_[0])
    if n == 5 or n == 9:
        pat_color.append(pat_color_[2])
    if n == 10 or n == 14:
        pat_color.append(pat_color_[4])
    if 0 < n and n < 4:
        pat_color.append(pat_color_[1])
    if 5 < n and n < 9:
        pat_color.append(pat_color_[3])
    if 10 < n and n < 14:
        pat_color.append(pat_color_[5])

# print(np.shape(pat_color))
event_start = []

state_start_stop = []
start = 0
duration_list = []
for i in range(plot_len):
    max_id = np.argmax(mean_firing_rate[:, i])
    # print(max_id)
    if mean_firing_rate[max_id, i - 1] < activation_threshold[max_id]:
        if mean_firing_rate[max_id, i] >= activation_threshold[max_id]:
            for jj in range(plot_len):
                if i + jj >= plot_len:
                    break
                elif mean_firing_rate[max_id, i + jj] < activation_threshold[max_id]:
                    state_start_stop.append([max_id, i, i + jj])
                    duration_list.append(jj)
                    break

hist = np.histogram(duration_list, bins="auto")[0]
x = np.arange(0, len(hist), 1)
x_fit = x + x[1] / 2
fit_ = curve_fit(exp_fit, x_fit, hist)

fit_eq = fit_[0][0] * np.exp(-fit_[0][1] * x_fit)

fig, ax1 = plt.subplots(figsize=(3, 3))
plt.bar(x, hist, align="edge", width=0.98, alpha=0.4, color="k")
pl.plot(x_fit, fit_eq, c="k")
plt.xticks([0, x_fit[-1]], ["0", str(max(duration_list))], fontsize=10)

ax1.xaxis.set_ticks_position("bottom")
ax1.yaxis.set_ticks_position("left")
ax1.spines["right"].set_color("none")
ax1.spines["top"].set_color("none")

plt.savefig("duration_hist.pdf", format="pdf", dpi=350)
print(max(duration_list) / len(hist) / fit_[0][1])

state_pare_list = []
dur = 50
transition_time_list = []
transition_estimated = np.zeros((n_pat, n_pat))

for i in range(len(state_start_stop) - 1):
    j = i + 1
    if state_start_stop[i][1] < state_start_stop[j][1]:
        if abs(state_start_stop[i][2] - state_start_stop[j][1]) < dur:
            if int(state_start_stop[i][0]) != int(state_start_stop[j][0]):
                transition_time_list.append(state_start_stop[i][2])
                transition_estimated[
                    int(state_start_stop[i][0]), int(state_start_stop[j][0])
                ] += 1
                state_pare_list.append(
                    [int(state_start_stop[i][0]), int(state_start_stop[j][0])]
                )

print(state_pare_list)

print(transition_estimated)
normed_transition_estimated = normalize((transition_estimated), axis=1, norm="l1")
print(normed_transition_estimated)
performance = np.mean((normed_transition_estimated - (np.array(transition_prob))) ** 2)
print(np.log10(performance))

fig, ax = plt.subplots(figsize=(3, 3))

pl.plot(
    np.array(transition_prob).reshape(n_pat**2, 1),
    normed_transition_estimated.reshape(n_pat**2, 1),
    lw=0,
    marker="o",
    markersize=4,
    clip_on=False,
    c="k",
)

ax.tick_params(direction="in", width=1, length=2)

plt.xlabel("True transition", fontsize=10)
plt.ylabel("Model transition", fontsize=10)
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
pl.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), lw=1, c="grey")
plt.xticks(np.array([0, 0.5, 0.9999]), ["0", "0.5", "1"])
plt.yticks(np.array([0, 0.5, 0.9999]), ["0", "0.5", "1"])
pl.ylim([-0.05, 1])
pl.xlim([-0.05, 1])
fig.subplots_adjust(left=0.15, bottom=0.25, right=0.8)

plt.savefig("transition_true_vs_estimated.pdf", format="pdf", dpi=350)
"""
fig, ax = plt.subplots(figsize=(1.8, 3))
id_within = np.array(transition_prob).reshape(n_pat**2, 1)>0
id_between = np.array(transition_prob).reshape(n_pat**2, 1)==0
plt.bar([1,1.5], [np.mean(normed_transition_estimated.reshape(n_pat**2, 1)[id_within]), np.mean(normed_transition_estimated.reshape(n_pat**2, 1)[id_between])],width = 0.4)
plt.vlines(x=1, ymin=np.mean(normed_transition_estimated.reshape(n_pat**2, 1)[id_within])-np.std(normed_transition_estimated.reshape(n_pat**2, 1)[id_within])/np.sqrt(len(normed_transition_estimated.reshape(n_pat**2, 1)[id_within])), ymax=np.mean(normed_transition_estimated.reshape(n_pat**2, 1)[id_within])+np.std(normed_transition_estimated.reshape(n_pat**2, 1)[id_within])/np.sqrt(len(normed_transition_estimated.reshape(n_pat**2, 1)[id_within])), color='k',lw=1.5)
plt.vlines(x=1.5, ymin=np.mean(normed_transition_estimated.reshape(n_pat**2, 1)[id_between])-np.std(normed_transition_estimated.reshape(n_pat**2, 1)[id_between])/np.sqrt(len(normed_transition_estimated.reshape(n_pat**2, 1)[id_between])), ymax=np.mean(normed_transition_estimated.reshape(n_pat**2, 1)[id_between])+np.std(normed_transition_estimated.reshape(n_pat**2, 1)[id_between])/np.sqrt(len(normed_transition_estimated.reshape(n_pat**2, 1)[id_between])), color='k',lw=1.5)
#ax.violinplot([normed_transition_estimated.reshape(n_pat**2, 1)[id_within], normed_transition_estimated.reshape(n_pat**2, 1)[id_between]], positions=[1,1.7], vert=True, showmeans=True)
#pl.plot(np.array(transition_prob).reshape(n_pat**2, 1)[id_within], normed_transition_estimated.reshape(n_pat**2, 1)[id_within], lw=0, marker='o',markersize = 4,clip_on=False,c='k')
#pl.plot(np.array(transition_prob).reshape(n_pat**2, 1)[id_between], normed_transition_estimated.reshape(n_pat**2, 1)[id_between], lw=0, marker='o',markersize = 4,clip_on=False,c='k')
ax.tick_params(direction="out", width=1,length = 2)
plt.xticks([1,1.5],
            ['within', 'between'])
#plt.xlabel("True transition", fontsize=10)
plt.ylabel("Model transition", fontsize=10)
fig.subplots_adjust(bottom=0.1, left=0.35)
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# ax1.spines['right'].set_color('none')#
# ax1.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.spines["left"].set_linewidth(1)
ax.spines["bottom"].set_linewidth(1)



plt.savefig('transition_true_vs_estimated_bargraph.pdf', format='pdf', dpi=350)
"""
fig, ax = plt.subplots(figsize=(4, 3))

cax = plt.imshow((np.array(transition_prob).T), cmap="RdPu", origin="lower")

cbar = fig.colorbar(cax, orientation="vertical")

plt.xlabel("From", fontsize=10)
plt.ylabel("To", fontsize=10)
plt.xticks(np.arange(0, n_pat), np.arange(1, int(n_pat + 1), dtype=int))
plt.yticks(np.arange(0, n_pat), np.arange(1, int(n_pat + 1), dtype=int))
ax.tick_params(length=1.3, width=0.05, labelsize=11)
ax.xaxis.set_ticks_position("none")
ax.yaxis.set_ticks_position("none")

fig.subplots_adjust(left=0.15, bottom=0.25, right=0.8)

plt.savefig("true_transition.pdf", format="pdf", dpi=350)

fig, ax = plt.subplots(figsize=(4, 3))

cax = plt.imshow((normed_transition_estimated).T, cmap="RdPu", origin="lower")

cbar = fig.colorbar(cax, orientation="vertical")

plt.xlabel("From", fontsize=10)
plt.ylabel("To", fontsize=10)
plt.xticks(np.arange(0, n_pat), np.arange(1, int(n_pat + 1), dtype=int))
plt.yticks(np.arange(0, n_pat), np.arange(1, int(n_pat + 1), dtype=int))
ax.tick_params(length=1.3, width=0.05, labelsize=11)
ax.xaxis.set_ticks_position("none")
ax.yaxis.set_ticks_position("none")

fig.subplots_adjust(left=0.15, bottom=0.25, right=0.8)

plt.savefig("estimated_transition.pdf", format="pdf", dpi=350)
