.. role:: html(raw)
   :format: html

.. _intro_ref_meas:

Measurements
============

.. currentmodule:: pennylane.measure

PennyLane can extract different types of measurement results from quantum
devices: the expectation of an observable, its variance,
samples of a single measurement, or computational basis state probabilities.

For example, the following circuit returns the expectation value of the
:class:`~pennylane.PauliZ` observable on wire 1:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))

The available measurement functions are

:html:`<div class="summary-table">`

.. autosummary::

    ~pennylane.expval
    ~pennylane.sample
    ~pennylane.var
    ~pennylane.probs
    ~pennylane.state
    ~pennylane.density_matrix

:html:`</div>`

.. note::

    All measurement functions support analytic differentiation, with the
    exception of :func:`~.pennylane.sample`, as it returns *stochastic*
    results.

Combined measurements
---------------------

Quantum functions can also return combined measurements of multiple observables, as long as each wire
is not measured more than once:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1)), qml.var(qml.PauliX(0))

You can also use list comprehensions, and other common Python patterns:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

As a full example of combined measurements, let us look at
a Bell state :math:`(|00\rangle + |11\rangle)/\sqrt{2}`, prepared
by a ``Hadamard`` and ``CNOT`` gate.

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

The combined PauliZ-measurement of the
first and second qubit returns a list of two lists, each containing
the measurement results of the respective qubit. As a default, :func:`~.pennylane.sample`
returns 1000 samples per observable.

>>> result = circuit()
>>> result.shape
(2, 1000)

Since the two qubits are maximally entangled,
the measurement results always coincide, and the lists are therefore equal:

>>> np.all(result[0] == result[1])
True


Tensor observables
------------------

PennyLane supports measuring the tensor product of observables, by using
the ``@`` notation. For example, to measure the expectation value of
:math:`Z\otimes I \otimes X`:

.. code-block:: python3

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(2))

Note that we don't need to declare the identity observable on wire 1; this is
implicitly assumed.

The tensor observable notation can be used inside all measurement functions that
accept observables as arguments,
including :func:`~.pennylane.expval`, :func:`~.pennylane.var`,
and :func:`~.pennylane.sample`.

Probability
-----------

You can also train QNodes on computational basis probabilities, by using
the :func:`~.pennylane.probs` measurement function. The function can
accept either specified ``wires`` or an observable that rotates the 
computational basis.

.. code-block:: python3

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        return qml.probs(wires=[0, 1])

For example:

>>> dev = qml.device("default.qubit", wires=3)
>>> qnode = qml.QNode(my_quantum_function, dev)
>>> qnode(0.56, 0.1)
array([0.99750208, 0.00249792, 0.        , 0.        ])

The returned probability array uses lexicographical ordering,
so corresponds to a :math:`99.75\%` probability of measuring
state :math:`|00\rangle`, and a :math:`0.25\%` probability of
measuring state :math:`|01\rangle`.

Changing the number of shots
----------------------------

For hardware devices where the number of shots determines the accuracy
of the expectation value and variance, as well as the number of samples returned,
it can sometimes be convenient to execute the same QNode with differing
number of shots.

For simulators like ``default.qubit``, finite shots will be simulated if
we set ``shots`` to a positive integer.

The shot number can be changed on the device itself, or temporarily altered
by the ``shots`` keyword argument when executing the QNode:


.. code-block:: python

    dev = qml.device("default.qubit", wires=1, shots=10)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y, wires=0)
        return qml.expval(qml.PauliZ(0))

    # execute the QNode using 10 shots
    result = circuit(0.54, 0.1)

    # execute the QNode again, now using 1 shot
    result = circuit(0.54, 0.1, shots=1)


With an increasing number of shots, the average over
measurement samples converges to the exact expectation of an observable. Consider the following
circuit:

.. code-block:: python

    # fix seed to make results reproducable
    np.random.seed(1)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

Running the simulator with ``shots=None`` returns the exact expectation.

>>> circuit(shots=None)
0.0

Now we set the device to return stochastic results, and increase the number of shots starting from ``10``.

>>> circuit(shots=10)
0.2

>>> circuit(shots=1000)
-0.062

>>> circuit(shots=100000)
0.00056

The result converges to the exact expectation.
