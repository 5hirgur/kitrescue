"""Settled, absolute NexArm waypoints; vendor SDK unchanged."""
import json
import math
import time

FIELDS = ("x", "y", "z", "pitch", "roll", "claw")



FEEDBACK_TOLERANCES = {
    "travel_mm": 12,
    "contact_xy_mm": 5,
    "contact_z_mm": 4,
    "pitch_deg": 6,
    "roll_deg": 5,
}

class Motion:
    def __init__(self, robot):
        self.r = robot
        self.arm = robot.arm
        self.clearance = max(84, int(robot.cal.get("travel_z", 84)))

    def state_pose(self, state):
        return {k: state[k] for k in FIELDS}

    def park_pose(self):
        p = self.r.park
        return {
            k: (
                p[k] if k in ("x", "y", "z")
                else self.r.base[k] if k in ("pitch", "roll")
                else -80
            )
            for k in FIELDS
        }

    def is_parked(self):
        state, target = self.arm.state(), self.park_pose()
        tolerances = {
            "x": FEEDBACK_TOLERANCES["travel_mm"],
            "y": FEEDBACK_TOLERANCES["travel_mm"],
            "z": FEEDBACK_TOLERANCES["travel_mm"],
            "pitch": FEEDBACK_TOLERANCES["pitch_deg"],
            "roll": FEEDBACK_TOLERANCES["roll_deg"],
            "claw": 5,
        }
        return all(
            abs(state[k] - target[k]) <= tolerances[k]
            for k in FIELDS
        )

    def require_park(self):
        if not self.is_parked():
            raise RuntimeError(
                "Not at camera_clear. Run park, or use "
                "--execute for automatic parking."
            )

    def segment(
        self, target, label,
        maintenance=False, minimum=None, precise=False
    ):
        before = self.arm.state()
        target = {
            k: round(float(target[k]), 1)
            if k == "pitch" else int(round(target[k]))
            for k in FIELDS
        }

        if not all(math.isfinite(v) for v in target.values()):
            raise RuntimeError("Non-finite motion target")
        if target["z"] < 5 or not -90 <= target["roll"] <= 90:
            raise RuntimeError("Target outside Z/wrist bounds")
        if (
            not maintenance
            and abs(target["pitch"] - self.r.base["pitch"]) > .1
        ):
            raise RuntimeError(
                "Task pitch must remain the taught top-down angle"
            )

        distance = math.dist(
            [before[k] for k in ("x", "y", "z")],
            [target[k] for k in ("x", "y", "z")],
        )
        near = (
            min(before["z"], target["z"])
            < self.r.base["z"] + 30
        )
        plan = self.arm.plan(target, duration_ms=5000)
        wanted = plan["solution"].get("servos", [])
        measured = before.get("servos", [])

        if len(wanted) != 6 or len(measured) != 6:
            raise RuntimeError(
                "Need six IK and measured servo positions"
            )

        joint_delta = max(
            abs(a - b)
            for a, b in zip(wanted[:5], measured[:5])
        )
        if joint_delta > 128:
            raise RuntimeError(
                f"IK discontinuity ({joint_delta} ticks); "
                "no command sent"
            )

        # Slow near contact and limit commanded average joint speed.
        duration = int(max(
            self.r.duration,
            2000,
            distance / (5 if near else 20) * 1000,
            joint_delta / 100 * 1000,
        ))
        if duration > 10000:
            raise RuntimeError(
                "Waypoint requires subdivision; no command sent"
            )
        plan["target"]["time_ms"] = duration

        if getattr(self.r, "fast_motion", False):
            slow_z = getattr(self, "_slow_zone", None)
            contact = slow_z is not None and min(before["z"], target["z"]) <= slow_z
            plan["fast_profile"] = {
                "contact": contact,
                "minimum": minimum,
                "tolerance": {
                    "x": FEEDBACK_TOLERANCES["contact_xy_mm"] if precise else FEEDBACK_TOLERANCES["travel_mm"],
                    "y": FEEDBACK_TOLERANCES["contact_xy_mm"] if precise else FEEDBACK_TOLERANCES["travel_mm"],
                    "z": FEEDBACK_TOLERANCES["contact_z_mm"] if precise else self.empty_retract_z_tolerance(label, before, target),
                    "pitch": FEEDBACK_TOLERANCES["pitch_deg"],
                    "roll": FEEDBACK_TOLERANCES["roll_deg"],
                },
            }
            plan["target"]["time_ms"] = max(5000, self.r.duration)

        readings = []
        lateral = math.dist(
            [before["x"], before["y"]],
            [target["x"], target["y"]],
        ) > 3

        floor = max(
            5, min(before["z"], target["z"]) - 10
        )
        if (
            minimum is not None
            and min(before["z"], target["z"]) >= minimum
        ):
            floor = max(floor, minimum)

        monitor = self.make_monitor(before, target, label, maintenance, minimum, readings)

        try:
            # One command, then feedback. No automatic motion retry.
            previous = getattr(self.arm, "last_target", None)
            same_command = previous is not None and all(
                abs(previous[k] - target[k]) < .05
                for k in FIELDS
            )
            if same_command:
                after = before
                monitor(after)
            else:
                self.r.log_dir.mkdir(parents=True, exist_ok=True)
                (
                    self.r.log_dir / "active_command.json"
                ).write_text(json.dumps({
                    "action": label,
                    "target": plan["target"],
                    "before": before,
                    "ik_solution": plan["solution"],
                }, indent=2))
                after = self.arm.execute_joints(plan, monitor=monitor)
            recent = []
            deadline = time.monotonic() + 3

            while time.monotonic() < deadline:
                after = self.arm.state()
                monitor(after)
                recent.append(after)
                recent = recent[-3:]

                quiet = len(recent) == 3 and all(
                    max(s[k] for s in recent)
                    - min(s[k] for s in recent)
                    <= (
                        1 if k in ("x", "y", "z") else .5
                    )
                    for k in ("x", "y", "z", "pitch", "roll")
                )
                if quiet:
                    break
                time.sleep(.08 if getattr(self.r, "fast_motion", False) and not precise else .25)
            else:
                raise RuntimeError("Feedback did not settle")

            if maintenance and label == "park_pitch":
                if not (
                    self.pitch_limits[0]
                    <= after["pitch"]
                    <= self.pitch_limits[1]
                ):
                    raise RuntimeError(
                        "Pitch did not return inside original "
                        "limits; no next waypoint sent"
                    )
            if maintenance and label == "park_pitch":
                delta = target["pitch"] - before["pitch"]
                progress = (
                    (after["pitch"] - before["pitch"])
                    * (1 if delta > 0 else -1)
                )
                if (
                    abs(delta) >= 2
                    and progress < max(1, abs(delta) / 4)
                ):
                    raise RuntimeError(
                        "Insufficient pitch alignment progress; "
                        "no next waypoint sent"
                    )
            errors = {
                k: round(after[k] - target[k], 2)
                for k in ("x", "y", "z", "pitch", "roll")
            }
            allowed = {
                "x": FEEDBACK_TOLERANCES["contact_xy_mm"] if precise else FEEDBACK_TOLERANCES["travel_mm"],
                "y": FEEDBACK_TOLERANCES["contact_xy_mm"] if precise else FEEDBACK_TOLERANCES["travel_mm"],
                "z": FEEDBACK_TOLERANCES["contact_z_mm"] if precise else self.empty_retract_z_tolerance(label, before, target),
                "pitch": FEEDBACK_TOLERANCES["pitch_deg"],
                "roll": FEEDBACK_TOLERANCES["roll_deg"],
            }
            if any(
                abs(errors[k]) > allowed[k]
                for k in errors
            ):
                raise RuntimeError(
                    f"Settled tracking error: {errors}; "
                    f"allowed={allowed}"
                )

            dz = target["z"] - before["z"]
            progress = (
                (after["z"] - before["z"])
                * (1 if dz > 0 else -1)
            )
            required_progress = abs(dz) / 2
            if getattr(self.r, "fast_motion", False) and allowed["z"] == 16:
                required_progress = max(5, abs(dz) - allowed["z"])
            empty_retract_progress = required_progress
            if (
                getattr(self.r, "fast_motion", False)
                and label.startswith("retract:")
                and dz > 0
                and target["claw"] == -80
                and abs(after["claw"] + 80) <= 5
            ):
                empty_retract_progress = max(5, abs(dz) - allowed["z"])

            if (abs(dz) >= 15 and progress < empty_retract_progress) and not (
    label.startswith("retract:")
    and label.endswith(":settle")
    and target["claw"] == -80
    and abs(after["claw"] + 80) <= 5
    and all(abs(after[k] - target[k]) <= allowed[k]
            for k in ("x", "y", "z", "pitch", "roll"))
    and (minimum is None or after["z"] >= minimum)
):
                raise RuntimeError("Insufficient measured vertical progress")

            if minimum is not None and after["z"] < minimum:
                raise RuntimeError(
                    "Required measured clearance not reached"
                )

        except BaseException as exc:
            self.r.log_dir.mkdir(
                parents=True, exist_ok=True
            )
            failure = {
                "action": label,
                "requested": target,
                "before": before,
                "ik_solution": plan["solution"],
                "feedback": readings,
                "error": str(exc),
                "active_segment_may_still_complete": True,
            }
            (
                self.r.log_dir / "last_motion_failure.json"
            ).write_text(json.dumps(failure, indent=2))
            raise

        self.r.log_dir.mkdir(parents=True, exist_ok=True)
        with (
            self.r.log_dir / "moves.jsonl"
        ).open("a") as f:
            f.write(json.dumps({
                "action": label,
                "requested": target,
                "after": after,
            }) + "\n")

        print(
            label, "XYZ=",
            [after[k] for k in ("x", "y", "z")],
            flush=True,
        )
        return after

    def interpolate(
        self, current, values, label, maximum,
        maintenance=False, minimum=None
    ):
        if getattr(self.r, "fast_motion", False):
            slow_z = getattr(self, "_slow_zone", None)
            raised = slow_z is None or min(current["z"], values.get("z", current["z"])) > slow_z
            if raised and any(k in values for k in ("x", "y", "z")):
                maximum = max(maximum, 40)
        origin = dict(current)
        command_origin = dict(current)
        count = max(1, math.ceil(
            max(
                abs(values[k] - origin[k])
                for k in values
            ) / maximum
        ))
        candidates = []

        for i in range(1, count + 1):
            target = dict(current)
            for k, v in values.items():
                value = (
                    origin[k]
                    + (v - origin[k]) * i / count
                )
                target[k] = (
                    round(value, 1)
                    if k == "pitch"
                    else int(round(value))
                )
            candidates.append(target)
            current = target

        # Preflight this leg; subdivide only when IK joint
        # changes require it.
        previous = self.arm.state()["servos"]
        origin = dict(command_origin)
        route = []

        def expand(a, b, pulses, depth=0):
            solution = self.arm.plan(
                b, duration_ms=5000
            )["solution"]["servos"]

            change = max(
                abs(x - y)
                for x, y in zip(
                    solution[:5], pulses[:5]
                )
            )
            if change <= 96:
                route.append(b)
                return solution

            if depth >= 8:
                raise RuntimeError(
                    "IK discontinuity during preflight; "
                    "no waypoint sent"
                )

            midpoint = {
                k: round((a[k] + b[k]) / 2, 1)
                if k == "pitch"
                else int(round((a[k] + b[k]) / 2))
                for k in FIELDS
            }
            midpoint["claw"] = b["claw"]
            if not maintenance:
                midpoint["pitch"] = b["pitch"]

            if midpoint == a or midpoint == b:
                raise RuntimeError(
                    "IK discontinuity cannot be subdivided; "
                    "no waypoint sent"
                )

            middle_pulses = expand(
                a, midpoint, pulses, depth + 1
            )
            return expand(
                midpoint, b, middle_pulses, depth + 1
            )

        for target in candidates:
            previous = expand(
                origin, target, previous
            )
            origin = target

        for target in route:
            self.segment(
                target, label, maintenance, minimum
            )

        # Advance the commanded path, not measured pose error.
        return current

    def park(self):
        self._slow_zone = None
        if self.is_parked():
            print("Already at camera_clear.")
            return

        s = self.arm.state()
        if s["z"] < self.clearance:
            raise RuntimeError(
                "Automatic park requires an empty gripper "
                "already raised above travel Z"
            )

        raw = self.arm.board.get_coord_limits()
        # Bounded raised pitch re-entry; targets remain in range.
        if not raw:
            raise RuntimeError("Cannot read controller limits")
        low, high = raw["pmin"] / 10, raw["pmax"] / 10
        # Bounded raised pitch alignment.
        if abs(s["claw"] + 80) > 5:
            raise RuntimeError(
                "Automatic parking requires an empty, open gripper"
            )
        if not low <= s["pitch"] <= high:
            if abs(s["pitch"] - self.r.base["pitch"]) > 15:
                raise RuntimeError(
                    "Pitch too far from top-down for automatic alignment"
                )

        self.pitch_limits = (
            raw["pmin"] / 10,
            raw["pmax"] / 10,
        )
        p = self.park_pose()
        current = self.state_pose(s)
        high = max(
            current["z"], p["z"], self.clearance
        )

        phases = (
            ({"pitch": p["pitch"]}, 10, "park_pitch"),
            ({"z": high}, 20, "park_raise"),
            ({"roll": p["roll"]}, 10, "park_roll"),
            (
                {"x": p["x"], "y": p["y"]},
                30, "park_xy",
            ),
            ({"z": p["z"]}, 20, "park_height"),
        )
        for values, maximum, label in phases:
            if (label == "park_pitch"
                    and self.pitch_limits[0] <= s["pitch"] <= self.pitch_limits[1]
                    and abs(s["pitch"] - p["pitch"]) <= FEEDBACK_TOLERANCES["pitch_deg"]):
                current["pitch"] = p["pitch"]
                print("Park: pitch already within tracking tolerance; continuing.", flush=True)
                continue
            if any(
                abs(current[k] - v) > .1
                for k, v in values.items()
            ):
                current = self.interpolate(
                    current, values, label, maximum,
                    True, self.clearance,
                )

        if current["claw"] != -80:
            current["claw"] = -80
            self.segment(
                current, "park_open_empty_gripper",
                True, self.clearance,
            )

        self.require_park()

    def move(self, step):
        target = self.state_pose(step["pose"])
        measured = self.arm.state()
        action = step["action"].split(":")[0]
        self._slow_zone = (
            min(measured["z"], target["z"]) + 20
            if action in ("grasp_height", "place_height", "lift", "retract")
            else None
        )
        clear = max(
            self.r.base["z"] + 50,
            step.get("minimum_z") or 5,
        )
        pitch_allowance = 10 if measured["z"] >= clear else 6
        if abs(
            measured["pitch"] - self.r.base["pitch"]
        ) > pitch_allowance:
            raise RuntimeError(
                "Starting top-down orientation outside tracking allowance"
            )
        if measured["z"] < 5 or not -90 <= measured["roll"] <= 90:
            raise RuntimeError("Starting Z/wrist outside bounds")

        previous = getattr(self.arm, "last_target", None)
        current = self.state_pose(
            previous if previous is not None else measured
        )
        current["pitch"] = self.r.base["pitch"]
        latched = getattr(self.arm, "claw_target", None)
        if latched is not None:
            current["claw"] = latched

        if previous is not None and any(
            abs(measured[k] - current[k])
            > self.travel_tolerance(k, measured, current)
            for k in ("x", "y", "z")
        ):
            raise RuntimeError(
                "Measured pose diverged from last command; "
                "no new movement sent"
            )

        lateral = math.dist(
            [current["x"], current["y"]],
            [target["x"], target["y"]],
        ) > .1
        rotate = abs(current["roll"] - target["roll"]) > .1

        if lateral or rotate:
            if min(measured["z"], current["z"]) < clear:
                raise RuntimeError(
                    "Insufficient measured clearance for XY/wrist movement"
                )

        if rotate:
            current = self.interpolate(
                current, {"roll": target["roll"]},
                step["action"] + ":wrist",
                10, minimum=clear,
            )
        if lateral:
            current = self.interpolate(
                current,
                {"x": target["x"], "y": target["y"]},
                step["action"] + ":xy",
                30, minimum=clear,
            )
        if current["z"] != target["z"]:
            travel_minimum = step.get("minimum_z")
            if min(
                measured["z"], target["z"]
            ) < (travel_minimum or 5):
                travel_minimum = None
            current = self.interpolate(
                current, {"z": target["z"]},
                step["action"] + ":z",
                20, minimum=travel_minimum,
            )

        return self.segment(
            target, step["action"] + ":settle",
            minimum=step.get("minimum_z"),
            precise=step["action"] in (
                "grasp_height", "close",
                "place_height", "release",
            ),
        )

    def test(self, axis):
        self._slow_zone = None
        self.require_park()
        before = self.arm.state()
        origin = self.park_pose()

        if axis == "z":
            target = dict(
                origin, z=int(before["z"]) + 20
            )
        else:
            target = dict(
                origin, roll=origin["roll"] + 10
            )

        self.interpolate(
            origin, {axis: target[axis]},
            "test_" + axis + "_out",
            20 if axis == "z" else 10,
            minimum=self.clearance,
        )
        after = self.arm.state()

        if abs(
            after[axis] - before[axis]
        ) < (10 if axis == "z" else 5):
            raise RuntimeError(
                "Test did not demonstrate sufficient actual "
                "movement; no return command sent"
            )

        current = dict(
            origin, **{axis: target[axis]}
        )
        self.interpolate(
            current, {axis: origin[axis]},
            "test_" + axis + "_back",
            20 if axis == "z" else 10,
            minimum=self.clearance,
        )

    def make_monitor(
        self, before, target, label,
        maintenance, minimum, readings,
    ):
        floor = max(5, min(before["z"], target["z"]) - 10)
        if (
            minimum is not None
            and min(before["z"], target["z"]) >= minimum
        ):
            floor = max(floor, minimum)
        def monitor(s):
            readings.append(dict(s))
            del readings[:-80]

            if not all(
                math.isfinite(float(s[k])) for k in FIELDS
            ):
                raise RuntimeError("Invalid feedback")

            if s["z"] < floor or not -90 <= s["roll"] <= 90:
                raise RuntimeError(
                    f"Clearance/wrist deviation: "
                    f"Z={s['z']}, floor={floor}"
                )

            # Command-anchored transfers: bounded raised transients.
            transient = (
                10 if not maintenance and min(
                    before["z"], target["z"], s["z"]
                ) >= self.r.base["z"] + 50 else 6
            )
            pitch_lo = min(
                before["pitch"], target["pitch"]
            ) - transient
            pitch_hi = max(
                before["pitch"], target["pitch"]
            ) + transient
            if not pitch_lo <= s["pitch"] <= pitch_hi:
                raise RuntimeError(
                    "Pitch left the commanded transition"
                )

            if (
                not maintenance
                and abs(s["pitch"] - self.r.base["pitch"]) > transient
            ):
                raise RuntimeError("Top-down pitch deviation")

            if maintenance and hasattr(self, "pitch_limits"):
                low, high = self.pitch_limits
                if label == "park_pitch":
                    if min(
                        before["z"], target["z"], s["z"]
                    ) < self.clearance:
                        raise RuntimeError(
                            "Pitch alignment requires raised clearance"
                        )
                    if not low <= target["pitch"] <= high:
                        raise RuntimeError(
                            "Requested pitch outside controller limits"
                        )
                    if not low <= before["pitch"] <= high:
                        if (
                            abs(
                                before["pitch"] - self.r.base["pitch"]
                            ) > 15
                            or abs(before["claw"] + 80) > 5
                        ):
                            raise RuntimeError(
                                "Invalid raised alignment start"
                            )
                        low = min(low, before["pitch"] - 6)
                        high = max(high, before["pitch"] + 6)
                if not low <= s["pitch"] <= high:
                    raise RuntimeError(
                        "Pitch left the allowed alignment envelope"
                    )

            roll_lo = min(
                before["roll"], target["roll"]
            ) - 5
            roll_hi = max(
                before["roll"], target["roll"]
            ) + 5
            if not roll_lo <= s["roll"] <= roll_hi:
                raise RuntimeError(
                    "Wrist left the commanded transition"
                )

        return monitor


    def travel_tolerance(self, key, before, target):
        if (key == "z" and getattr(self.r, "fast_motion", False)
                and min(before["z"], target["z"]) >= self.r.base["z"] + 50):
            return 16
        return FEEDBACK_TOLERANCES["travel_mm"]


    def empty_retract_z_tolerance(self, label, before, target):
        if (
            getattr(self.r, "fast_motion", False)
            and label.startswith("retract:")
            and target["claw"] == -80
            and abs(before["claw"] + 80) <= 5
        ):
            return 16
        return self.travel_tolerance("z", before, target)



