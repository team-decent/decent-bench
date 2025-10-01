Background
==========

Decentralized optimization is a paradigm in the field of mathematical optimization. It addresses several of the
challenges posed by traditional, centralized optimization. In centralized optimization, all training data is transferred
to a central server that employs an optimization algorithm. In addition to increased network and power consumption,
transferring data may raise privacy concerns, especially in the context of sensitive information such as medical data.
There are also regulatory restrictions such as GDPR and the EU AI Act that may impact the feasibility of centralized
optimization.

The decentralized paradigm addresses these issues. A network of agents participate in the optimization
process by sending local variable updates to their neighbors, no training data is transmitted. There are two main
approaches in decentralized optimization, federated and distributed. In the federated approach, agents only communicate
with a coordinator. In each iteration, the coordinator retrieves local variable updates from the agents, updates the
global model, and then distributes it back to the agents for the next iteration. In contrast, distributed optimization
does not use a coordinator, agents communicate directly with their neighbors instead. Both approaches have their pros
and cons, with federated being faster and distributed more robust. Despite their differences, both approaches take
advantage of the Internet of Things and address the privacy concerns detailed earlier.

However, as decentralized optimization relies on network communication, factors such as noise, packet loss, compression,
network sparsity, and agent heterogeneity may all impact the optimization process. Therefore, these constraints must be
considered when evaluating an algorithm's performance. This is where decent-bench comes in. By
benchmarking algorithms in different settings with different communication constraints, decent-bench provides you with
realistic algorithm comparisons in a user-friendly and highly configurable manner.
