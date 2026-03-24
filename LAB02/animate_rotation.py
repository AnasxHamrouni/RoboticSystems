import numpy as np
from matplotlib import pyplot as plt, animation


# simple local visualizer (so the script runs standalone)
def visualize_rotation(rotation_matrices, stats=None, title="Rotation animation", show=True):
    """Animate body-frame axes represented by a sequence of rotation matrices."""
    axes_t = np.array([
        rotation_matrices[:, :, 0],
        rotation_matrices[:, :, 1],
        rotation_matrices[:, :, 2],
    ])
    n_frames = len(axes_t[0])
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_zlim((-1.1, 1.1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    colors = ["r", "g", "b"]
    axis_lines = [ax.plot([], [], [], "-", c=c, lw=3)[0] for c in colors]
    trails = [ax.plot([], [], [], "--", c=c, alpha=0.3)[0] for c in colors]
    points = [ax.plot([], [], [], "o", c=c)[0] for c in colors]
    info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    lag = 30

    def init():
        for l, tr, p in zip(axis_lines, trails, points):
            l.set_data([], [])
            l.set_3d_properties([])
            tr.set_data([], [])
            tr.set_3d_properties([])
            p.set_data([], [])
            p.set_3d_properties([])
        info_text.set_text("")
        return axis_lines + trails + points + [info_text]

    def animate(i):
        stat_text = ""
        for l, tr, p, xi in zip(axis_lines, trails, points, axes_t):
            x, y, z = xi[: i + 1].T
            tr.set_data(x[-lag:], y[-lag:])
            tr.set_3d_properties(z[-lag:])
            l.set_data([0, x[-1]], [0, y[-1]])
            l.set_3d_properties([0, z[-1]])
            p.set_data([x[-1]], [y[-1]])
            p.set_3d_properties([z[-1]])

        if stats is not None:
            for name, values in stats:
                stat_text += f"{name}: {values[i]:.4f}\n"
            info_text.set_text(stat_text)

        return axis_lines + trails + points + [info_text]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_frames,
        interval=1000 / 60,
        blit=True,
    )

    if show:
        plt.show()

    return anim


# small math helpers
def skew(omega):
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return q.copy()
    return q / n


def axis_angle_to_quat(axis, theta):
    n = np.linalg.norm(axis)
    if n < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / n
    return np.hstack(([np.cos(theta / 2.0)], axis * np.sin(theta / 2.0)))


def quat_to_rot(q, normalize_input=True):
    if normalize_input:
        q = quat_normalize(q)
    q0, q1, q2, q3 = q
    return np.array(
        [
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
        ]
    )


def q_dot_from_omega(q, omega):
    # quaternion kinematics: qdot = 0.5 * R(q) * omega
    q0, q1, q2, q3 = q
    rq = np.array(
        [
            [-q1, -q2, -q3],
            [q0, q3, -q2],
            [-q3, q0, q1],
            [q2, -q1, q0],
        ]
    )
    return 0.5 * (rq @ omega)


def reorthonormalize(R):
    """Projection onto SO(3): closest proper rotation matrix via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


# Task 2.1
def task_21_rotation_matrices(n=240):
    theta = np.linspace(0.0, np.pi / 2.0, n)
    rm_x = np.array([rot_x(th) for th in theta])
    rm_y = np.array([rot_y(th) for th in theta])
    rm_z = np.array([rot_z(th) for th in theta])

    visualize_rotation(rm_x, stats=[["theta", theta]], title="2.1 Rotation matrix around x")
    visualize_rotation(rm_y, stats=[["theta", theta]], title="2.1 Rotation matrix around y")
    visualize_rotation(rm_z, stats=[["theta", theta]], title="2.1 Rotation matrix around z")


# Task 2.2
def task_22_quaternion_rotation(n=240):
    theta = np.linspace(0.0, np.pi / 2.0, n)
    rm_x = np.array([quat_to_rot(axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), th)) for th in theta])
    rm_y = np.array([quat_to_rot(axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), th)) for th in theta])
    rm_z = np.array([quat_to_rot(axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), th)) for th in theta])

    visualize_rotation(rm_x, stats=[["theta", theta]], title="2.2 Quaternion rotation around x")
    visualize_rotation(rm_y, stats=[["theta", theta]], title="2.2 Quaternion rotation around y")
    visualize_rotation(rm_z, stats=[["theta", theta]], title="2.2 Quaternion rotation around z")


# Task 2.3
def task_23_quaternion_kinematics(omega, tf=5.0, dt=1 / 120, normalize=True):
    t = np.arange(0.0, tf, dt)
    n = len(t)

    q = np.array([1.0, 0.0, 0.0, 0.0])
    rm = np.zeros((n, 3, 3))
    q_norm = np.zeros(n)

    for k in range(n):
        # store first, then do Euler step
        q_norm[k] = np.linalg.norm(q)
        # for the non-normalized demo, also build R from non-unit q to expose drift
        rm[k] = quat_to_rot(q, normalize_input=normalize)
        q = q + dt * q_dot_from_omega(q, omega)
        if normalize:
            # this keeps q on the unit sphere
            q = quat_normalize(q)

    visualize_rotation(
        rm,
        stats=[["t", t], ["||q||", q_norm]],
        title=f"2.3 Quaternion Euler, omega={omega}, normalize={normalize}",
    )


# Task 2.4
def task_24_rotation_matrix_kinematics(omega, tf=5.0, dt=1 / 120, orthonormalize=False):
    t = np.arange(0.0, tf, dt)
    n = len(t)

    R = np.eye(3)
    rm = np.zeros((n, 3, 3))
    ortho_error = np.zeros(n)
    W = skew(omega)

    for k in range(n):
        # same pattern: store then integrate
        rm[k] = R
        ortho_error[k] = np.linalg.norm(R.T @ R - np.eye(3))
        R = R + dt * (W @ R)
        if orthonormalize:
            # projection back to a valid rotation matrix
            R = reorthonormalize(R)

    visualize_rotation(
        rm,
        stats=[["t", t], ["||R^TR-I||", ortho_error]],
        title=f"2.4 Matrix Euler, omega={omega}, orthonormalize={orthonormalize}",
    )


# Tasks 2.5 and 2.6 
# 2.5 (increase angular velocity in quaternion integration):
# If omega is large, Forward Euler causes stronger numerical drift.
# Without normalization, ||q|| deviates from 1, and orientation becomes physically inconsistent.
# Fix: normalize q at every step and/or use smaller dt and better integrator (RK4, exponential map).
#
# 2.6 (increase angular velocity in matrix integration):
# Forward Euler breaks orthogonality of R, so R^T R != I and det(R) moves away from +1.
# This gets worse for larger omega or dt.
# Fix: project R back to SO(3) (SVD re-orthonormalization), reduce dt,
# or use Lie-group/exponential integration for rotations.


if __name__ == "__main__":
    task_21_rotation_matrices()
    task_22_quaternion_rotation()
    task_23_quaternion_kinematics(np.array([1.0, 0.0, 0.0]), normalize=True)
    task_23_quaternion_kinematics(np.array([0.0, 1.0, 0.0]), normalize=True)
    task_23_quaternion_kinematics(np.array([0.0, 0.0, 1.0]), normalize=True)
    task_24_rotation_matrix_kinematics(np.array([1.0, 0.0, 0.0]), orthonormalize=True)
    task_24_rotation_matrix_kinematics(np.array([0.0, 1.0, 0.0]), orthonormalize=True)
    task_24_rotation_matrix_kinematics(np.array([0.0, 0.0, 1.0]), orthonormalize=True)

    # extra runs for 2.5/2.6 observations with larger angular speed
    high_omega = np.array([6.0, 0.0, 0.0])
    task_23_quaternion_kinematics(high_omega, normalize=False)
    task_23_quaternion_kinematics(high_omega, normalize=True)
    task_24_rotation_matrix_kinematics(high_omega, orthonormalize=False)
    task_24_rotation_matrix_kinematics(high_omega, orthonormalize=True)
