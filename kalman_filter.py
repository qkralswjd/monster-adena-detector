"""
kalman_filter.py
----------------
칼만 필터 기반 위치/속도 예측 모듈.

상태 벡터: [cx, cy, w, h, vx, vy, vw, vh]
- cx, cy: 중심 좌표
- w, h  : 박스 크기
- vx, vy: 이동 속도
- vw, vh: 크기 변화 속도

몬스터가 잠깐 사라져도 마지막 속도/방향으로
예상 위치를 계산할 수 있다.
"""

import numpy as np


class KalmanFilter:
    """
    8차원 칼만 필터.
    상태: [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self):
        ndim = 4  # 관측 차원 (cx, cy, w, h)
        dt = 1.0  # 시간 스텝

        # 상태 전이 행렬 F (8x8)
        self.F = np.eye(2*ndim, 2*ndim)
        for i in range(ndim):
            self.F[i, ndim+i] = dt

        # 관측 행렬 H (4x8) - cx,cy,w,h만 관측
        self.H = np.eye(ndim, 2*ndim)

        # 프로세스 노이즈 Q
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # 관측 노이즈 R
        self._std_weight_meas = 1.0 / 20

    def initiate(self, measurement: np.ndarray):
        """
        초기 상태 생성.
        measurement: [cx, cy, w, h]
        Returns: (mean, covariance)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        다음 프레임 위치 예측.
        Returns: (predicted_mean, predicted_covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        Q = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray):
        """
        실제 관측값으로 상태 보정.
        measurement: [cx, cy, w, h]
        Returns: (updated_mean, updated_covariance)
        """
        std = [
            self._std_weight_meas * mean[2],
            self._std_weight_meas * mean[3],
            self._std_weight_meas * mean[2],
            self._std_weight_meas * mean[3],
        ]
        R = np.diag(np.square(std))

        S = self.H @ covariance @ self.H.T + R
        K = covariance @ self.H.T @ np.linalg.inv(S)

        innovation = measurement - self.H @ mean
        mean = mean + K @ innovation
        covariance = (np.eye(len(mean)) - K @ self.H) @ covariance
        return mean, covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray) -> np.ndarray:
        """
        예측 위치와 관측값들 사이의 마할라노비스 거리.
        ID 매칭 시 사용.
        """
        std = [
            self._std_weight_meas * mean[2],
            self._std_weight_meas * mean[3],
            self._std_weight_meas * mean[2],
            self._std_weight_meas * mean[3],
        ]
        R = np.diag(np.square(std))
        S = self.H @ covariance @ self.H.T + R

        d = measurements - self.H @ mean
        SI = np.linalg.inv(S)
        distances = np.sqrt(np.sum(d @ SI * d, axis=1))
        return distances
