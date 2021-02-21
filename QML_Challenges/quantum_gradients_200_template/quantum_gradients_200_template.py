#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    # QHACK #
    s_param_g = 6

    def calc_gradient(weights):
        unit_v = np.zeros_like(weights)
        gradient_ = np.zeros_like(weights)
        add_calc = np.zeros_like(weights)
        substr_cal = np.zeros_like(weights)
        for ii in np.ndenumerate(unit_v):
            ii = ii[0]
            unit_v[ii] = 1
            add_calc[ii] = circuit(weights + s_param_g * unit_v)
            substr_cal[ii] = circuit(weights - s_param_g * unit_v)
            gradient_[ii] = (add_calc[ii] - substr_cal[ii]) / (2 * np.sin(s_param_g))
            unit_v[ii] = 0
        return gradient_, add_calc, substr_cal

    s_param_h = 3

    def calc_hessian(weights, add_calc, substr_cal):
        hessian = np.zeros([5, 5], dtype=np.float64)

        base_result = circuit(weights)
        unit_1 = np.zeros_like(weights)
        unit_2 = np.zeros_like(weights)

        for ii in range(unit_1.shape[0]):
            for jj in range(ii + 1):
                if ii == jj:
                    hessian[ii, jj] = (((add_calc[ii] + substr_cal[ii] - 2 * base_result) / (2 * np.sin(s_param_h)))) / (2 * np.sin(s_param_h))
                else:
                    unit_1[ii] = 1
                    unit_2[jj] = 1
                    param_shift_00 = circuit(weights + s_param_h * unit_1 + s_param_h * unit_2)
                    param_shift_01 = circuit(weights + s_param_h * unit_1 - s_param_h * unit_2)
                    param_shift_10 = circuit(weights - s_param_h * unit_1 + s_param_h * unit_2)
                    param_shift_11 = circuit(weights - s_param_h * unit_1 - s_param_h * unit_2)
                    param_shift_1 = (param_shift_00 - param_shift_01) / (2 * np.sin(s_param_h))
                    param_shift_2 = (param_shift_10 - param_shift_11) / (2 * np.sin(s_param_h))
                    hessian[ii, jj] = (param_shift_1 - param_shift_2) / (2 * np.sin(s_param_h))
                    unit_1[ii] = 0
                    unit_2[jj] = 0
        hessian = hessian + hessian.T - np.diag(np.diag(hessian))
        return hessian

    gradient, add_calc, substr_cal = calc_gradient(weights)
    hessian = calc_hessian(weights, add_calc, substr_cal)
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
