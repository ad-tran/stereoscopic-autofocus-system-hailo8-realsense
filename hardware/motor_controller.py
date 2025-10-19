import bisect
import time
import multiprocessing as mp

from utils.config import MOTOR_LUT


def _distance_to_steps(distance_m: float) -> int:
    steps = [x[0] for x in MOTOR_LUT]
    distances = [x[1] for x in MOTOR_LUT]
    pos = bisect.bisect_left(distances, distance_m)
    if pos == 0:
        return steps[0]
    elif pos == len(distances):
        return steps[-1]
    else:
        prev_step, prev_dist = MOTOR_LUT[pos - 1]
        next_step, next_dist = MOTOR_LUT[pos]
        alpha = (distance_m - prev_dist) / (next_dist - prev_dist)
        return int(prev_step + alpha * (next_step - prev_step))


def _focus_plane_pos(curr_step: int) -> float:
    steps = [x[0] for x in MOTOR_LUT]
    distances = [x[1] for x in MOTOR_LUT]
    pos = bisect.bisect_left(steps, curr_step)
    if pos == 0:
        return distances[0]
    elif pos == len(steps):
        return distances[-1]
    else:
        prev_step, prev_dist = MOTOR_LUT[pos - 1]
        next_step, next_dist = MOTOR_LUT[pos]
        alpha = (curr_step - prev_step) / (next_step - prev_step)
        return prev_dist + alpha * (next_dist - prev_dist)


def _motor_worker(queue, stop_event, current_motor_steps, default_focus_time):
    #  immer echte Hardware verwenden, kein Fallback
    from adafruit_motorkit import MotorKit
    from adafruit_motor import stepper
    import time

    kit = MotorKit()
    max_speed_delay = 0.001
    homing_speed_delay = 0.01

    try:
        while not stop_event.is_set():
            target_steps = None
            focustime = default_focus_time
            while not queue.empty():
                item = queue.get_nowait()
                if isinstance(item, tuple):
                    target_steps, focustime = item
                else:
                    target_steps = item
                    focustime = default_focus_time

            if target_steps is None:
                time.sleep(0.01)
                continue

            with current_motor_steps.get_lock():
                steps_diff = target_steps - current_motor_steps.value
            if steps_diff == 0:
                continue

            direction = stepper.FORWARD if steps_diff > 0 else stepper.BACKWARD
            steps_remaining = abs(steps_diff)
            if focustime and steps_remaining > 0:
                delay = max(focustime / steps_remaining, max_speed_delay)
            else:
                delay = max_speed_delay

            for _ in range(steps_remaining):
                if stop_event.is_set() or not queue.empty():
                    break
                kit.stepper1.onestep(direction=direction, style=stepper.INTERLEAVE)
                with current_motor_steps.get_lock():
                    if direction == stepper.FORWARD:
                        current_motor_steps.value += 1
                    else:
                        current_motor_steps.value -= 1
                time.sleep(delay)

    finally:
        #  immer Homing und Release
        with current_motor_steps.get_lock():
            home_steps = -current_motor_steps.value
        if home_steps != 0:
            home_dir = stepper.FORWARD if home_steps > 0 else stepper.BACKWARD
            for _ in range(abs(home_steps)):
                kit.stepper1.onestep(direction=home_dir, style=stepper.INTERLEAVE)
                with current_motor_steps.get_lock():
                    if home_dir == stepper.FORWARD:
                        current_motor_steps.value += 1
                    else:
                        current_motor_steps.value -= 1
                time.sleep(homing_speed_delay)
        try:
            kit.stepper1.release()
        except Exception:
            pass


class MotorController:
    def __init__(self, initial_focus_time=0.0):
        #  kein get_context/daemon
        self.queue = mp.Queue()
        self.stop_event = mp.Event()
        self.current_motor_steps = mp.Value('i', 0)
        self.process = mp.Process(
            target=_motor_worker,
            args=(self.queue, self.stop_event, self.current_motor_steps, initial_focus_time)
        )
        self.process.start()

    def move_to(self, steps: int, focus_time: float = 0.001):
        try:
            self.queue.put((int(steps), float(focus_time)))
        except Exception as e:
            print(f"Failed to enqueue motor move: {e}")

    @property
    def current_steps(self) -> int:
        try:
            with self.current_motor_steps.get_lock():
                return int(self.current_motor_steps.value)
        except Exception:
            return 0

    @staticmethod
    def distance_to_steps(distance_m: float) -> int:
        return _distance_to_steps(distance_m)

    @staticmethod
    def focus_plane_pos(curr_step: int) -> float:
        return _focus_plane_pos(curr_step)

    def stop(self):
        try:
            self.stop_event.set()
            if self.process.is_alive():
                self.process.join(timeout=2)
        except Exception:
            pass