# Use geometry raycast observations for MID-360-like 3D perception

MID-360-like 3D perception will treat obstacle geometry as the authority and generate per-step observed point clouds by raycasting the sensor model against that geometry. The controller will consume a robot-local, point-budgeted subset of those observed points, and the old global sampled obstacle points will be removed from the new observation and replay schema rather than remaining as a parallel truth source.
