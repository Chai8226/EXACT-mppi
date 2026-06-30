# Use closed convex polyhedron unions for 3D robot volumes

The first yaw-only 3D MPPI version represents arbitrary 3D robot bodies as unions of closed convex polyhedra. Each convex part stores triangle faces for exact point-to-surface distance and halfspaces for inside/outside classification; this keeps the representation close to the existing 2D rectangle/polygon union model, supports non-convex bodies through composition, and avoids treating arbitrary triangle meshes as a first-version public shape format.
