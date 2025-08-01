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
from __future__ import division, annotations
import numpy as np
import numba
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import simplefilter
from numpy import linalg as LA
import typing as tp

import util


class SimOutput(tp.NamedTuple):
    nsim: int
    error_list: np.ndarray
    error_filtered_list_E_mean: np.ndarray
    error_filtered_list_E_std: np.ndarray
    error_filtered_list_I_mean: np.ndarray
    error_filtered_list_I_std: np.ndarray

    @classmethod
    def from_empty(cls, simtime_len: int, nsim: int) -> SimOutput:
        return cls(
            nsim=nsim,
            error_list=np.zeros(simtime_len),
            error_filtered_list_E_mean=np.zeros(simtime_len),
            error_filtered_list_E_std=np.zeros(simtime_len),
            error_filtered_list_I_mean=np.zeros(simtime_len),
            error_filtered_list_I_std=np.zeros(simtime_len),
        )

    def _from_concat(cls, outputs: tp.Sequence[SimOutput]) -> tp.Self:
        error_list = outputs[0].error_list  # note: we only keep the first one
        expected_size = error_list.shape[0]
        for o in outputs:
            for arr in o:
                if not isinstance(arr, np.ndarray):
                    continue
                if arr.ndim != 1:
                    raise ValueError("All arrays must be 1D.")
                if arr.shape[0] != expected_size:
                    raise ValueError(
                        f"All arrays must have the same length. Expected {expected_size}, got {arr.shape[0]}."
                    )

        E_means = [o.error_filtered_list_E_mean for o in outputs]
        E_stds = [o.error_filtered_list_E_std for o in outputs]
        I_means = [o.error_filtered_list_I_mean for o in outputs]
        I_stds = [o.error_filtered_list_I_std for o in outputs]
        return cls(
            nsim=len(outputs),
            error_list=error_list,
            error_filtered_list_E_mean=np.vstack(E_means),
            error_filtered_list_E_std=np.vstack(E_stds),
            error_filtered_list_I_mean=np.vstack(I_means),
            error_filtered_list_I_std=np.vstack(I_stds),
        )

    @classmethod
    def save_all(cls, outputs: tp.Sequence[SimOutput]) -> None:

        concatenated = cls._from_concat(outputs)

        np.savetxt("error_list.txt", concatenated.error_list, delimiter=",")

        np.savetxt(
            "error_filtered_list_E_mean.txt",
            concatenated.error_filtered_list_E_mean,
            delimiter=",",
        )
        np.savetxt(
            "error_filtered_list_E_std.txt",
            concatenated.error_filtered_list_E_std,
            delimiter=",",
        )
        np.savetxt(
            "error_filtered_list_I_mean.txt",
            concatenated.error_filtered_list_I_mean,
            delimiter=",",
        )
        np.savetxt(
            "error_filtered_list_I_std.txt",
            concatenated.error_filtered_list_I_std,
            delimiter=",",
        )


def run_simulations():

    # transition_prob = [[0,1/2,1/2,0,0],[1/3,0,1/3,1/3,0],[1/3,0,0,1/3,1/3],[0,1/2,1/2,0,0],[0,0,1,0,0]]
    # state_list = [0, 1, 2, 3,4]
    transition_prob = [[0, 0.1, 0.9], [0.1, 0, 0.9], [0.5, 0.5, 0]]
    num_states = len(transition_prob)  # num of states

    N = 1000  # network size
    N_E = int(N * 1 / 2)  # num of exc neurons
    N_I = N - N_E  # num of inh neurons

    activation_f = util.get_activation_function()

    ###

    ## learning rule
    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def learning(w, g_V_star, PSP_star, eps, g_V_som, mask):
        for i in numba.prange(len(w[:, 0])):
            for l in numba.prange(len(w[0, :])):  # noqa: E741
                delta = (-(g_V_star[i]) + g_V_som[i]) * PSP_star[l]
                w[i, l] += eps * delta
                w[i, l] *= mask[i, l]
                w[i, l] *= w[i, l] >= 0
        return w

    ##

    dt = 1  # integration time step

    dur_state = 200  # duration of states in ms

    eps = 10 ** (-5)  # learning rate
    # JNOTE: the methods list 10^-4

    msecs_learning = 200 * 1000  # ms for learning
    simtime_len = msecs_learning // dt

    # JNOTE: these are in ms
    tau_m = 15  # membrane time constant
    tau_syn = 5  # synaptic time constant

    # JNOTE: paper says 50 Hz. This is not passed into sigmoid as alpha, but rather multiplied after the fact
    max_rate = 0.05  # maxumus firing rate of sigmoid

    p_connect = 0.5  # connection probability

    n_sim = 1

    state3_start_time = []

    sim_outputs = []
    for sim in range(n_sim):
        sim_output = run_simulation(
            N_I=N_I,
            activation_f=activation_f,
            dur_state=dur_state,
            eps=eps,
            tau_m=tau_m,
            tau_syn=tau_syn,
            max_rate=max_rate,
            p_connect=p_connect,
            state3_start_time=state3_start_time,
            transition_prob=transition_prob,
            N_E=N_E,
            num_states=num_states,
            simtime_len=simtime_len,
            dt=dt,
            learning=learning,
            sim=sim,
            N=N,
        )
        sim_outputs.append(sim_output)

    SimOutput.save_all(sim_outputs)


def run_simulation(
    N_I: tp.Any,
    activation_f: tp.Any,
    dur_state: tp.Any,
    eps: tp.Any,
    tau_m: tp.Any,
    tau_syn: tp.Any,
    max_rate: tp.Any,
    p_connect: tp.Any,
    state3_start_time: tp.Any,
    transition_prob: tp.Any,
    N_E: tp.Any,
    num_states: tp.Any,
    simtime_len: tp.Any,
    dt: tp.Any,
    learning: tp.Any,
    sim: tp.Any,
    N: tp.Any,
) -> SimOutput:
    W_E = np.ones((N, N)) / np.sqrt(p_connect * N_E) * 1  # E connections
    W_E[0:N_E, 0:N_E] *= 0.1
    W_E_mask = np.zeros((N, N))  # E connectivity
    W_E_mask[:, 0:N_E] = np.random.rand(N, N_E) < p_connect
    W_E_mask[np.eye(N, dtype=bool)] = 0
    W_E *= W_E_mask

    W_I = np.ones((N, N)) / np.sqrt(p_connect * N_I) * 1  # I connections
    W_I[0:N_E, N_E:] *= 0.1
    W_I_mask = np.zeros((N, N))  # I connectivity
    W_I_mask[:, N_E:] = np.random.rand(N, N_I) < p_connect
    W_I_mask[np.eye(N, dtype=bool)] = 0
    W_I *= W_I_mask

    I_syn = np.zeros(N)  # synaptic current
    PSP = np.zeros(N)  # post synaptic potential

    id_rec = np.zeros(N, dtype=bool)  # neurons that spike (at each time)

    strength_input = 2  # strength of external input
    input_mat = np.zeros((N, num_states))  # set of external input
    for i in range(num_states):
        input_mat[int(i * N_E / num_states) : int((i + 1) * N_E / num_states), i] = (
            strength_input
        )

    """
    Training
    """
    PSP_amp = (tau_m - tau_syn) / (
        (tau_m / tau_syn) ** (-(tau_syn) / (tau_m - tau_syn))
        - (tau_m / tau_syn) ** (-(tau_m) / (tau_m - tau_syn))
    )

    pat_start = 0  # time of next stimulus presentation
    pattern_id = 0  # initial state
    max_trace = (
        np.zeros(N) + 1
    )  # exponential trace for tracking maximum value of membrane potential
    tau_max = 10 * 1000  # time scale of max_trace

    delta_weight_save = 2000  # synaptic weights are saved every 2 sec.
    W_E_list = np.zeros(
        (N_E, N_E, int(simtime_len / delta_weight_save))
    )  # array of saved W_E
    W_I_list = np.zeros(
        (N_E, N_I, int(simtime_len / delta_weight_save))
    )  # array of saved W_I

    W_E_between = np.ones((N, N))
    W_E_between[:, N_E:] *= 0
    for i in range(num_states):
        W_E_between[
            int(i * N_E / num_states) : int((i + 1) * N_E / num_states),
            int(i * N_E / num_states) : int((i + 1) * N_E / num_states),
        ] *= 0
    W_E_within = np.zeros((N, N))
    W_E_within[:, N_E:] *= 0
    for i in range(num_states):
        W_E_within[
            int(i * N_E / num_states) : int((i + 1) * N_E / num_states),
            int(i * N_E / num_states) : int((i + 1) * N_E / num_states),
        ] = 1

    error_filtered_E = np.zeros(int(N_E / 1))
    error_filtered_I = np.zeros(int(N_E / 1))

    sim_output = SimOutput.from_empty(simtime_len, n_sim=sim)
    # training
    for i in tqdm(range(simtime_len), desc="[training]"):
        if i == pat_start:
            if pattern_id == 2:
                state3_start_time.append(i)

            I_ext = input_mat[:, pattern_id]  # exteternal corrent
            pat_start += dur_state  # time of next stimulus presentation
            ### updating state
            next_prob = transition_prob[pattern_id]
            dice = np.random.rand()
            for xx in range(num_states):
                if dice > np.sum(next_prob[0:xx]):
                    if dice <= np.sum(next_prob[0 : xx + 1]):
                        pattern_id = xx
                        break

        # updating recurrent synaptic current and PSP
        I_syn = (1.0 - dt / tau_syn) * I_syn
        I_syn[id_rec] += 1 / tau_m / tau_syn
        PSP = (1.0 - dt / tau_m) * PSP + I_syn * PSP_amp

        I_term = np.dot(W_I, PSP)  # inhibitory effect on membrane potential
        input_term = I_ext  # external input effect on membrane potential
        E_term = np.dot(W_E, PSP)

        x = input_term + E_term - I_term  # total membrane potential

        # updating max_trace
        max_trace = (1.0 - dt / tau_max) * max_trace
        max_trace[max_trace < x] = x[max_trace < x]
        max_trace[N_E:] = 2

        y = (
            x / max_trace
        )  # normalized membrane potential with max_trace. This will be replaced with homeostatis of weights?

        f = activation_f(
            2 * y
        )  # firing rate of neurons. y is scaled with a constant value of 2.

        # JNOTE: is 30000 the 30s time constant for low pass error filter?
        if i > 0:
            error_filtered_E = (1.0 - 1 / 30000) * error_filtered_E + (
                f[int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)]
                - activation_f(E_term)[
                    int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)
                ]
            ) / 30000
            error_filtered_I = (1.0 - 1 / 30000) * error_filtered_I + (
                activation_f(E_term)[
                    int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)
                ]
                - activation_f(I_term)[
                    int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)
                ]
            ) / 30000
        else:
            error_filtered_E = (
                f[int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)]
                - activation_f(E_term)[
                    int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)
                ]
            )
            error_filtered_I = (
                activation_f(E_term)[
                    int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)
                ]
                - activation_f(I_term)[
                    int(0 * N_E / num_states) : int((2 + 1) * N_E / num_states)
                ]
            )
        sim_output.error_filtered_list_E_mean[i] = np.mean(np.abs(error_filtered_E))
        sim_output.error_filtered_list_E_std[i] = np.std(np.abs(error_filtered_E))

        sim_output.error_filtered_list_I_mean[i] = np.mean(np.abs(error_filtered_I))
        sim_output.error_filtered_list_I_std[i] = np.std(np.abs(error_filtered_I))

        id_rec = np.random.rand(N) < f * dt * max_rate  # neuron id that emit spikes

        # training. only synapses onto E-neurons are plastic.
        W_E[0:N_E, :] = learning(
            W_E[0:N_E, :],
            activation_f(E_term)[0:N_E],
            PSP,
            eps,
            f[0:N_E],
            W_E_mask[0:N_E, :],
        )
        W_I[0:N_E, :] = learning(
            W_I[0:N_E, :],
            activation_f(I_term)[0:N_E],
            PSP,
            eps,
            activation_f(E_term)[0:N_E],
            W_I_mask[0:N_E, :],
        )

        # saving weights
        if i % delta_weight_save == 0:
            W_E_list[:, :, int(i / delta_weight_save)] = W_E[0:N_E, 0:N_E]
            W_I_list[:, :, int(i / delta_weight_save)] = W_I[0:N_E, N_E:]

        # saving errors
        sim_output.error_list[i] = np.mean(f - activation_f(E_term))

    # learning curve
    LC_E = []
    LC_I = []
    for kk in range(int(simtime_len / delta_weight_save)):
        if kk > 0:
            LC_E.append(LA.norm(W_E_list[:, :, kk] - W_E_list[:, :, kk - 1]) / (N_E**2))
            LC_I.append(
                LA.norm(W_I_list[:, :, kk] - W_I_list[:, :, kk - 1]) / (N_E * N_I)
            )

    np.savetxt("learning_curve_E.txt", LC_E, delimiter=",")
    np.savetxt("learning_curve_I.txt", LC_I, delimiter=",")

    arr_reshaped = W_E_list.reshape(W_E_list.shape[0], -1)
    # saving reshaped array to file.
    np.savetxt("W_E_list.txt", arr_reshaped)

    np.savetxt("W_E_%s.txt" % (sim), W_E, delimiter=",")
    np.savetxt("W_I_%s.txt" % (sim), W_I, delimiter=",")

    plt.close("all")

    return error_list


if __name__ == "__main__":
    simplefilter(action="ignore", category=FutureWarning)
    util.configure_mpl()
    run_simulations()
