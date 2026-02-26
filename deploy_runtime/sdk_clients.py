import numpy as np
import importlib


class FairinoSDKClient:
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self.robot = None
        self._servo_started = False

    def _import_robot(self):
        try:
            module = importlib.import_module('fairino')
            if hasattr(module, 'Robot'):
                return module.Robot
        except Exception:
            pass

        try:
            module = importlib.import_module('fairino.Robot')
            if hasattr(module, 'Robot'):
                return module.Robot
        except Exception:
            pass

        raise ImportError('Cannot import Fairino SDK Robot. Please install fairino-python-sdk.')

    def connect(self):
        Robot = self._import_robot()
        self.robot = Robot.RPC(self.robot_ip)

    def _safe_call(self, fn_name, *args, **kwargs):
        if self.robot is None:
            raise RuntimeError('Fairino robot is not connected')
        fn = getattr(self.robot, fn_name, None)
        if fn is None:
            return None
        return fn(*args, **kwargs)

    def enable_robot(self):
        ret = self._safe_call('RobotEnable', 1)
        if ret is None:
            ret = self._safe_call('RobotEnable')
        return ret

    def reset_to_home(self):
        q_home = [0, -30, 90, 0, 60, 0]
        ret = self._safe_call('MoveJ', q_home, 0, 0, 30.0, 30.0)
        if ret is None:
            ret = self._safe_call('MoveJ', q_home)
        return ret

    @staticmethod
    def _parse_joint_return(ret, dof=6):
        if ret is None:
            return np.zeros(dof, dtype=np.float32)
        if isinstance(ret, (list, tuple)):
            if len(ret) == dof:
                return np.asarray(ret, dtype=np.float32)
            if len(ret) == dof + 1 and isinstance(ret[0], (int, float)):
                return np.asarray(ret[1:], dtype=np.float32)
            if len(ret) >= 2 and isinstance(ret[1], (list, tuple)):
                vec = list(ret[1])[:dof]
                if len(vec) < dof:
                    vec = vec + [0.0] * (dof - len(vec))
                return np.asarray(vec, dtype=np.float32)
        return np.zeros(dof, dtype=np.float32)

    def get_joint_state(self):
        q = self._safe_call('GetActualJointPosDegree')
        dq = self._safe_call('GetActualJointSpeedsDegree')
        q6 = self._parse_joint_return(q, dof=6)
        dq6 = self._parse_joint_return(dq, dof=6)

        q7 = np.concatenate([q6, np.array([0.0], dtype=np.float32)], axis=0)
        dq7 = np.concatenate([dq6, np.array([0.0], dtype=np.float32)], axis=0)
        return q7, dq7

    def servo_j(self, target_qpos):
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        target_q6 = target_qpos[:6].tolist()
        if not self._servo_started:
            self._safe_call('ServoMoveStart')
            self._servo_started = True
        ret = self._safe_call('ServoJ', target_q6)
        if ret is None:
            self._safe_call('MoveJ', target_q6)

    def stop(self):
        self._safe_call('StopMotion')
        self._safe_call('StopMove')
        if self._servo_started:
            self._safe_call('ServoMoveEnd')
            self._servo_started = False

    def disconnect(self):
        self.stop()


class ExcavatorSDKClient:
    def __init__(self, backend=None):
        self.backend = backend

    def connect(self):
        if self.backend is None:
            raise NotImplementedError('Excavator SDK backend is not configured')
        self.backend.connect()

    def enable_system(self):
        self.backend.enable_system()

    def reset_to_start_pose(self):
        self.backend.reset_to_start_pose()

    def get_joint_state(self):
        return self.backend.get_joint_state()

    def send_joint_target(self, target_qpos):
        self.backend.send_joint_target(target_qpos)

    def stop_motion(self):
        self.backend.stop_motion()

    def disconnect(self):
        self.backend.disconnect()
