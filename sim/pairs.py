"""Plan both members of a controlled start comparison before either rollout."""
import numpy as np

from .robot import FR3
from .sim import prepare_case
from .starts import NO_IK, StartProblem, StartUnavailable


def pairing_settings(pairing):
    fields = {"posture": {"min_sigma_gap"}, "lock": {"fixed_joints_rad"}, "robot": set()}
    mode = pairing.get("mode") if isinstance(pairing, dict) else None
    if mode not in tuple(fields):
        raise ValueError("pairing.mode must be posture, lock or robot")
    unknown = pairing.keys() - fields[mode] - {"mode"}
    if unknown:
        raise ValueError(f"Unknown pairing fields for {mode}: {sorted(unknown)}")
    if mode == "posture":
        gap = pairing.get("min_sigma_gap", 0.001)
        if type(gap) not in (int, float) or not (np.isfinite(gap) and gap > 0):
            raise ValueError("pairing.min_sigma_gap must be finite and positive")
        return mode, None, gap
    locks = pairing.get("fixed_joints_rad")
    if mode == "lock":
        if not isinstance(locks, dict) or len(locks) != 1:
            raise ValueError("pairing.fixed_joints_rad must specify one FR3 joint and angle")
        return mode, next(iter(locks.items())), None
    return mode, None, None


def comparison_configs(cfg, comparison, lock=None):
    if cfg["sim"].get("start_criterion") != "closest_to_reference":
        raise ValueError("Paired starts require sim.start_criterion=closest_to_reference")
    if comparison == "posture":
        return {"nominal": cfg, "lower_sigma": cfg}
    if cfg["robot"] != "FR3" or cfg["task"].get("fixed_joints_rad"):
        raise ValueError("Robot and lock comparisons start from a full FR3 config")
    if comparison == "robot":
        return {"FR3": cfg, "UR5e": {**cfg, "robot": "UR5e"}}
    if comparison != "lock" or lock is None or lock[0] not in FR3.joints:
        raise ValueError("pairing.fixed_joints_rad must name an FR3 joint")
    joint, angle = lock
    index = FR3.joints.index(joint)
    if type(angle) not in (int, float) or not FR3.q_min_doc[index] <= angle <= FR3.q_max_doc[index]:
        raise ValueError(f"Invalid prescribed lock angle {joint}={angle}")
    locked = {
        **cfg,
        "robot": f"FR3_fixed_J{index + 1}",
        "task": {**cfg["task"], "fixed_joints_rad": {joint: float(angle)}},
    }
    return {"free": cfg, "locked": locked}


def plan_pair(configs, start_id, comparison, sigma_gap):
    problems = {
        member: StartProblem(prepare_case(cfg), start_id)
        for member, cfg in configs.items() if member != "lower_sigma"
    }
    if comparison == "robot":
        rows = {}
        for member, problem in problems.items():
            try:
                rows[member] = problem.record(problem.solve().q)
            except StartUnavailable as exc:
                rows[member] = exc.row
    elif comparison == "posture":
        nominal = problems["nominal"]
        try:
            result = nominal.solve()
            rows = {
                "nominal": nominal.record(result.q),
                "lower_sigma": nominal.lower_sigma(result, sigma_gap),
            }
        except StartUnavailable as exc:
            rows = dict.fromkeys(configs, exc.row)
        nominal_sigma = rows["nominal"].get("sigma_min_start")
        lower_sigma = rows["lower_sigma"].get("sigma_min_start")
        for row in rows.values():
            row.update(min_sigma_gap=sigma_gap, nominal_sigma_min=nominal_sigma, lower_sigma_min=lower_sigma)
            if nominal_sigma is not None and nominal.orientation in ("damped", "hold"):
                row["orientation_gain_q_rad"] = rows["nominal"]["q_start_rad"]
    else:
        locked, free = problems["locked"], problems["free"]

        def free_angles(q):
            return np.asarray(locked.robot.full_angles(q))

        try:
            result = locked.solve(lambda q: free.valid_candidate(free_angles(q)))
            q_free = free.from_angles(free_angles(result.q)).q
            rows = {"free": free.record(q_free), "locked": locked.record(result.q)}
        except StartUnavailable:
            rows = {member: problem.record(status=NO_IK) for member, problem in problems.items()}
        for row in rows.values():
            row["prescribed_joints_rad"] = dict(locked.robot.spec.fixed_joints)
    pair_id = f"{comparison}_{start_id}"
    return {member: {**row, "member": member, "pair_id": pair_id} for member, row in rows.items()}
