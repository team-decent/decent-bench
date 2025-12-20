Computational cost is calculated as:

.. math::
    \text{Total Cost} = c_f N_f + c_g N_g + c_h N_h + c_p N_p + c_c N_c

where :math:`c_f, c_g, c_h, c_p, c_c` are the costs per function, gradient, Hessian, proximal, and communication
call respectively, and :math:`N_f, N_g, N_h, N_p, N_c` are the mean number of function, gradient, Hessian,
proximal, and communication calls across all agents and trials.