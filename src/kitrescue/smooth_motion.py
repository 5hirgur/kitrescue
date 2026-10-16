"""Use the established joint-waypoint controller for task execution."""

try:
    from .motion import Motion as WaypointMotion
except ImportError:
    from motion import Motion as WaypointMotion


class Motion(WaypointMotion):
    def follow_path(self, steps):
        steps = list(steps)

        if any(
            step["action"] in ("close", "release")
            for step in steps
        ):
            raise ValueError("Gripper actions must remain separate.")

        result = None
        for step in steps:
            result = self.move(step)

        return result
