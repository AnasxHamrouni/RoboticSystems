import numpy as np
import matplotlib.pyplot as plt


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return q.copy()
    return q / n


def quat_kinematics_matrix(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.array(
        [
            [-q1, -q2, -q3],
            [q0, q3, -q2],
            [-q3, q0, q1],
            [q2, -q1, q0],
        ],
        dtype=float,
    )


def rigid_body_dynamics(x: np.ndarray, inertia: np.ndarray, torque: np.ndarray) -> np.ndarray:
    """
    State-space model for a single rigid body attitude dynamics.

    x = [q0, q1, q2, q3, wx, wy, wz]^T
    qdot = 0.5 * R(q) * omega
    wdot = J^{-1} * (tau - omega x (J * omega))
    """
    q = x[:4]
    w = x[4:]

    qdot = 0.5 * (quat_kinematics_matrix(q) @ w)
    wdot = np.linalg.solve(inertia, torque - np.cross(w, inertia @ w))

    return np.hstack((qdot, wdot))


def rk4_step(x: np.ndarray, dt: float, inertia: np.ndarray, torque: np.ndarray) -> np.ndarray:
    k1 = rigid_body_dynamics(x, inertia, torque)
    k2 = rigid_body_dynamics(x + 0.5 * dt * k1, inertia, torque)
    k3 = rigid_body_dynamics(x + 0.5 * dt * k2, inertia, torque)
    k4 = rigid_body_dynamics(x + dt * k3, inertia, torque)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[:4] = quat_normalize(x_next[:4])
    return x_next


def quat_to_axis_angle_vector(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion to axis-angle vector r = theta * axis (R^3 representation).
    """
    q = quat_normalize(q)
    q0 = np.clip(q[0], -1.0, 1.0)
    qv = q[1:]
    s = np.linalg.norm(qv)

    if s < 1e-12:
        return np.zeros(3)

    theta = 2.0 * np.arctan2(s, q0)
    axis = qv / s
    return axis * theta


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    q0, q1, q2, q3 = q
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=float,
    )


def simulate_attitude_rk4(
    inertia: np.ndarray,
    w0: np.ndarray,
    q0: np.ndarray | None = None,
    torque: np.ndarray | None = None,
    tf: float = 12.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if q0 is None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if torque is None:
        torque = np.zeros(3, dtype=float)

    t = np.arange(0.0, tf + dt, dt)
    x = np.zeros((len(t), 7), dtype=float)
    x[0, :4] = quat_normalize(q0)
    x[0, 4:] = w0

    for k in range(len(t) - 1):
        x[k + 1] = rk4_step(x[k], dt, inertia, torque)

    r = np.array([quat_to_axis_angle_vector(xk[:4]) for xk in x])
    return t, x, r


def plot_task_22(
    t: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
    out_prefix: str,
    title_prefix: str = "Task 2.2",
) -> None:
    q = x[:, :4]

    plt.figure(figsize=(9, 5))
    plt.plot(t, q[:, 0], label="q0")
    plt.plot(t, q[:, 1], label="q1")
    plt.plot(t, q[:, 2], label="q2")
    plt.plot(t, q[:, 3], label="q3")
    plt.xlabel("Time [s]")
    plt.ylabel("Quaternion components")
    plt.title(f"{title_prefix}: Quaternion Components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_quaternion.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(t, r[:, 0], label="r_x = theta * a_x")
    plt.plot(t, r[:, 1], label="r_y = theta * a_y")
    plt.plot(t, r[:, 2], label="r_z = theta * a_z")
    plt.xlabel("Time [s]")
    plt.ylabel("Axis-angle vector components [rad]")
    plt.title(f"{title_prefix}: Axis-Angle Components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_axis_angle.png", dpi=160)
    plt.close()


def attitude_position_dynamics(
    x: np.ndarray,
    inertia: np.ndarray,
    mass: float,
    torque: np.ndarray,
    force_body: np.ndarray,
) -> np.ndarray:
    """
    Extended state: x = [q(4), w(3), p(3), v(3)]
    qdot = 0.5*R(q)*w
    wdot = J^{-1}(tau - w x Jw)
    pdot = v
    vdot = (1/m) * C(q) * F_body
    """
    q = x[:4]
    w = x[4:7]
    v = x[10:13]

    qdot = 0.5 * (quat_kinematics_matrix(q) @ w)
    wdot = np.linalg.solve(inertia, torque - np.cross(w, inertia @ w))
    pdot = v
    vdot = (quat_to_rot_matrix(q) @ force_body) / mass

    return np.hstack((qdot, wdot, pdot, vdot))


def rk4_step_attitude_position(
    x: np.ndarray,
    dt: float,
    inertia: np.ndarray,
    mass: float,
    torque: np.ndarray,
    force_body: np.ndarray,
) -> np.ndarray:
    k1 = attitude_position_dynamics(x, inertia, mass, torque, force_body)
    k2 = attitude_position_dynamics(x + 0.5 * dt * k1, inertia, mass, torque, force_body)
    k3 = attitude_position_dynamics(x + 0.5 * dt * k2, inertia, mass, torque, force_body)
    k4 = attitude_position_dynamics(x + dt * k3, inertia, mass, torque, force_body)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[:4] = quat_normalize(x_next[:4])
    return x_next


def simulate_attitude_position_rk4(
    inertia: np.ndarray,
    mass: float,
    w0: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    q0: np.ndarray | None = None,
    torque: np.ndarray | None = None,
    force_body: np.ndarray | None = None,
    tf: float = 12.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if q0 is None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if torque is None:
        torque = np.zeros(3, dtype=float)
    if force_body is None:
        force_body = np.zeros(3, dtype=float)

    t = np.arange(0.0, tf + dt, dt)
    x = np.zeros((len(t), 13), dtype=float)
    x[0, :4] = quat_normalize(q0)
    x[0, 4:7] = w0
    x[0, 7:10] = p0
    x[0, 10:13] = v0

    for k in range(len(t) - 1):
        x[k + 1] = rk4_step_attitude_position(x[k], dt, inertia, mass, torque, force_body)

    r = np.array([quat_to_axis_angle_vector(xk[:4]) for xk in x])
    return t, x, r


def plot_position_components(t: np.ndarray, p: np.ndarray, out_prefix: str, title_prefix: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(t, p[:, 0], label="p_x")
    plt.plot(t, p[:, 1], label="p_y")
    plt.plot(t, p[:, 2], label="p_z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position components")
    plt.title(f"{title_prefix}: Position Components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_position.png", dpi=160)
    plt.close()


def main() -> None:
    # Non-symmetric inertia makes Euler dynamics visible for general initial conditions.
    inertia = np.diag([2.0, 1.0, 0.5])
    torque = np.zeros(3)

    initial_omegas = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    print("Task 2.2 simulations (tau = 0):")
    for w0 in initial_omegas:
        t, x, _ = simulate_attitude_rk4(inertia=inertia, w0=w0, torque=torque)
        w_final = x[-1, 4:]
        print(f"w0={w0.tolist()} -> w(T)={np.round(w_final, 6).tolist()}")

    # Save plots for one chosen initial angular velocity.
    chosen_w0 = np.array([1.0, 0.0, 0.0])
    t, x, r = simulate_attitude_rk4(inertia=inertia, w0=chosen_w0, torque=torque)
    plot_task_22(t, x, r, out_prefix="LAB03/task22_w100", title_prefix="Task 2.2")

    print("Saved:")
    print("- LAB03/task22_w100_quaternion.png")
    print("- LAB03/task22_w100_axis_angle.png")

    # Task 2.3: small deviation in another component.
    w0_task23 = np.array([0.1, 0.0, 1.0])
    t23, x23, r23 = simulate_attitude_rk4(inertia=inertia, w0=w0_task23, torque=torque)
    plot_task_22(t23, x23, r23, out_prefix="LAB03/task23_w0101", title_prefix="Task 2.3")

    print("Saved:")
    print("- LAB03/task23_w0101_quaternion.png")
    print("- LAB03/task23_w0101_axis_angle.png")
    print(f"Task 2.3 final omega: {np.round(x23[-1, 4:], 6).tolist()}")

    # Task 2.4: middle-axis theorem check (dominant y with small perturbations in x,z).
    w0_task24 = np.array([0.08, 1.0, 0.06])
    t24, x24, r24 = simulate_attitude_rk4(inertia=inertia, w0=w0_task24, torque=np.zeros(3), tf=18.0)
    plot_task_22(t24, x24, r24, out_prefix="LAB03/task24_w081006", title_prefix="Task 2.4")
    print("Saved:")
    print("- LAB03/task24_w081006_quaternion.png")
    print("- LAB03/task24_w081006_axis_angle.png")
    print(f"Task 2.4 final omega: {np.round(x24[-1, 4:], 6).tolist()}")

    # Task 2.5: non-zero external torque.
    tau_task25 = np.array([0.03, -0.025, 0.02])
    w0_task25 = np.array([0.1, 0.2, 0.6])
    t25, x25, r25 = simulate_attitude_rk4(inertia=inertia, w0=w0_task25, torque=tau_task25, tf=12.0)
    plot_task_22(t25, x25, r25, out_prefix="LAB03/task25_tau", title_prefix="Task 2.5")
    print("Saved:")
    print("- LAB03/task25_tau_quaternion.png")
    print("- LAB03/task25_tau_axis_angle.png")
    print(f"Task 2.5 final omega: {np.round(x25[-1, 4:], 6).tolist()}")

    # Task 2.6 bonus: extend state with position and velocity.
    # x = [q(4), w(3), p(3), v(3)]
    mass = 2.0
    w0_task26 = np.array([0.05, 0.4, 0.1])
    p0_task26 = np.array([0.0, 0.0, 0.0])
    v0_task26 = np.array([0.0, 0.0, 0.0])
    tau_task26 = np.array([0.0, 0.015, 0.0])
    force_body_task26 = np.array([0.4, 0.0, 0.0])

    t26, x26, r26 = simulate_attitude_position_rk4(
        inertia=inertia,
        mass=mass,
        w0=w0_task26,
        p0=p0_task26,
        v0=v0_task26,
        torque=tau_task26,
        force_body=force_body_task26,
        tf=14.0,
    )
    plot_task_22(t26, x26[:, :7], r26, out_prefix="LAB03/task26_bonus", title_prefix="Task 2.6")
    plot_position_components(t26, x26[:, 7:10], out_prefix="LAB03/task26_bonus", title_prefix="Task 2.6")
    print("Saved:")
    print("- LAB03/task26_bonus_quaternion.png")
    print("- LAB03/task26_bonus_axis_angle.png")
    print("- LAB03/task26_bonus_position.png")
    print(f"Task 2.6 final omega: {np.round(x26[-1, 4:7], 6).tolist()}")
    print(f"Task 2.6 final position: {np.round(x26[-1, 7:10], 6).tolist()}")


if __name__ == "__main__":
    main()
