# Use box-union YAML for the first 3D volume config

The first yaw-only 3D Core Python path will expose robot volume configuration as a `box_union` helper instead of requiring users to hand-author triangle faces and halfspaces. Each body-frame box will be converted internally into closed triangle faces for exact point-to-surface distance and halfspaces for inside/outside classification, which is enough to compose non-convex L/T-style robot volumes while keeping example YAML readable.
