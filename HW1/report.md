# HW1 Report: Kinematics and Dynamics of Rotation

Author: Anas Khmais Hamrouni



## Task 1: Lie Integration of Quaternion Kinematics

### Problem statement
A rotating rigid body is parameterized by quaternion $q = [q_0, q_1, q_2, q_3]^T$ with kinematics

$$
\dot{q} = \frac{1}{2} q \otimes \hat{\omega}, \quad \hat{\omega} = [0, \omega_x, \omega_y, \omega_z]^T.
$$

The exact discrete update used in the simulation is

$$
q_{k+1} = q_k \otimes \exp\!\left(\frac{1}{2}\hat{\omega}_k\,dt\right).
$$

For comparison, Forward Euler is

$$
q_{k+1}^{\text{Euler}} = q_k + dt\,\dot{q}_k.
$$

### Numerical setup
- Final time: $T = 12$ s
- Time step: $dt = 0.02$ s
- Angular velocity: $\omega = [0.7,\,1.1,\,0.5]^T$ rad/s
- Initial quaternion: $q_0 = [1,0,0,0]^T$

### Results
The Lie integrator preserves the unit norm much better than Forward Euler.

![Task 1 norm comparison](outputs/task1_norm_comparison.png)

Quaternion component trajectories for Lie vs Euler are shown below.

![Task 1 quaternion components](outputs/task1_quaternion_components.png)

Animation files (after running the notebook):
- `outputs/rotation_exact.mp4`
- `outputs/rotation_euler.mp4`

### Task 1 conclusion
- Exact Lie integration keeps $\|q\|$ close to 1.
- Forward Euler introduces norm drift and orientation error over time.


## Task 2: Newton-Euler Dynamics Integration

### Problem statement
Simulate rigid-body translational and rotational dynamics using Lie integration for quaternion kinematics, and provide plots of:
- position $(x,y,z)$ and linear velocities,
- quaternion components and angular velocities.

### Dynamic model
State vector:

$$
x = [q,\,\omega,\,p,\,v]^T
$$

with equations:

$$
\dot{p} = v,
$$

$$
\dot{v} = g + \frac{1}{m} R(q) F_{\text{body}},
$$

$$
\dot{\omega} = J^{-1}\!\left(\tau - \omega \times (J\omega)\right),
$$

and quaternion update on the Lie group:

$$
q_{k+1} = q_k \otimes \exp\!\left(\frac{1}{2}\hat{\omega}_{\text{mid}}\,dt\right),
\quad \omega_{\text{mid}} = \frac{\omega_k + \omega_{k+1}}{2}.
$$

### Numerical setup
- Final time: $T = 12$ s
- Time step: $dt = 0.01$ s
- Mass: $m = 1.8$ kg
- Inertia: $J = \mathrm{diag}(2.0, 1.3, 0.8)$
- Gravity: $g = [0,0,-9.81]^T$ m/s$^2$
- Body force: $F_{\text{body}} = [4,0,18]^T$
- Body torque: $\tau = [0.15,-0.05,0.10]^T$
- Initial conditions:
  - $q_0 = [1,0,0,0]^T$
  - $\omega_0 = [0.6,1.0,0.3]^T$
  - $p_0 = [0,0,0]^T$
  - $v_0 = [1.0,0.4,0.2]^T$

### Results
Position components:

![Task 2 position](outputs/task2_position_components.png)

Velocity components:

![Task 2 velocity](outputs/task2_velocity_components.png)

Quaternion components:

![Task 2 quaternion](outputs/task2_quaternion_components.png)

Angular velocity components:

![Task 2 angular velocity](outputs/task2_angular_velocity_components.png)

Animation file (after running the notebook):
- `outputs/rigid_body_motion_problem2.mp4`

### Task 2 conclusion
- The simulation shows coupled translational and rotational motion under constant body-frame force and torque.
- Quaternion norm remains near 1 due to Lie-group update.