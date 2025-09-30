.. math::
    k = \frac{\mu}{\|\mathbf{x}_k - \mathbf{x}^\star\|^q}

where k is the iteration,
:math:`\mu` is the iterative convergence rate,
:math:`\mathbf{x}_k` is the agent's local x at iteration k,
:math:`\mathbf{x}^\star` is the optimal x defined in the *problem*,
and :math:`q` is the iterative convergence order.

As per the definition, iterative convergence is a measure of how many iterations are needed to reach a certain error.
This makes iterative convergence order and rate suitable metrics for sublinear algorithms; a sublinear algorithm with
iterative convergence order 0.5 generally converges significantly faster than one with order 1.
