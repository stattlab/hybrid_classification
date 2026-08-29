# hybrid_classification
Script to automatically classify hybrid membranes


Requires:
- Assumes trajectories are in GSDs
- Scipy/sklearn/DBSCAN/numpy

Note that the exact thresholds were fit to match hand-labeled data for the DPPC-1,4 PBD-b-PEO system. This classification method should work, generally, for different systems, but the exact thresholds will have to change. It is best to visually inspect the different morphologies and assess their nature.

 The thresholds are fit such that 74/79 morphologies are classified the same as the hand-labeled set. The morphologies which are not classified as the same are on the boundary between two different morphologies, and thus not easily classified either by eye or numerically.