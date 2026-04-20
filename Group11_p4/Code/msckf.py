import numpy as np
from scipy.stats import chi2

from utils import *
from feature import Feature

import time
from collections import namedtuple


class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0.0, 0.0, -9.81])

    # Transformation offset from the IMU frame to the body frame.
    # The transformation takes a vector from the IMU frame to the
    # body frame. The z axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    T_imu_wrt_body = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the IMU (body) frame to the world frame.
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Position of the IMU (body) frame in the world frame.
        self.position = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `position`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0.0, 0.0, 0.0, 1.0])
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_wrt_cam0 = np.identity(3)
        self.t_cam0_wrt_imu = np.zeros(3)


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_wrt_cam1 = None
    t_cam0_wrt_cam1 = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the camera frame to the world frame.
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Position of the camera frame in the world frame.
        self.position = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0.0, 0.0, 0.0, 1.0])
        self.position_null = np.zeros(3)


class StateServer(object):
    """
    Store one IMU states and several camera states for constructing
    measurement model.
    """

    def __init__(self):
        self.imu_state = IMUState()
        self.cam_states = dict()  # <CAMStateID, CAMState>, ordered dict

        # State covariance matrix
        self.state_cov = np.zeros((21, 21))
        self.continuous_noise_cov = np.zeros((12, 12))


class MSCKF(object):
    def __init__(self, config):
        self.config = config
        self.optimization_config = config.optimization_config

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()  # <FeatureID, Feature>

        # Chi squared test table.
        # Initialize the chi squared test table with confidence level 0.95.
        self.chi_squared_test_table = dict()
        for i in range(1, 100):
            self.chi_squared_test_table[i] = chi2.ppf(0.05, i)

        # Set the initial IMU state.
        # The intial orientation and position will be set to the origin implicitly.
        # But the initial velocity and bias can be set by parameters.
        # TODO: is it reasonable to set the initial bias to 0?
        self.state_server.imu_state.velocity = config.velocity
        self.reset_state_cov()

        continuous_noise_cov = np.identity(12)
        continuous_noise_cov[:3, :3] *= self.config.gyro_noise
        continuous_noise_cov[3:6, 3:6] *= self.config.gyro_bias_noise
        continuous_noise_cov[6:9, 6:9] *= self.config.acc_noise
        continuous_noise_cov[9:, 9:] *= self.config.acc_bias_noise
        self.state_server.continuous_noise_cov = continuous_noise_cov

        # Gravity vector in the world frame
        IMUState.gravity = config.gravity

        # Transformation between the IMU and the left camera (cam0)
        T_cam0_wrt_imu = np.linalg.inv(config.T_imu_cam0)
        self.state_server.imu_state.R_imu_wrt_cam0 = T_cam0_wrt_imu[:3, :3].T
        self.state_server.imu_state.t_cam0_wrt_imu = T_cam0_wrt_imu[:3, 3]

        # Extrinsic parameters of camera and IMU.
        T_cam0_wrt_cam1 = config.T_cn_cnm1
        CAMState.R_cam0_wrt_cam1 = T_cam0_wrt_cam1[:3, :3]
        CAMState.t_cam0_wrt_cam1 = T_cam0_wrt_cam1[:3, 3]
        Feature.R_cam0_cam1 = CAMState.R_cam0_wrt_cam1
        Feature.t_cam0_cam1 = CAMState.t_cam0_wrt_cam1
        IMUState.T_imu_wrt_body = Isometry3d(
            config.T_imu_body[:3, :3], config.T_imu_body[:3, 3]
        )

        # Tracking rate.
        self.tracking_rate = None

        # Indicate if the gravity vector is set.
        self.is_gravity_set = False
        # Indicate if the received image is the first one. The system will
        # start after receiving the first image.
        self.is_first_img = True

    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        # IMU msgs are pushed backed into a buffer instead of being processed
        # immediately. The IMU msgs are processed when the next image is
        # available, in which way, we can easily handle the transfer delay.
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg):
        """
        Callback function for feature measurements.
        """
        if not self.is_gravity_set:
            return
        start = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        t = time.time()

        # Propogate the IMU state.
        # that are received before the image msg.
        self.batch_imu_processing(feature_msg.timestamp)

        print("---batch_imu_processing    ", time.time() - t)
        t = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        print("---state_augmentation      ", time.time() - t)
        t = time.time()

        # Add new observations for existing features or new features
        # in the map server.
        self.add_feature_observations(feature_msg)

        print("---add_feature_observations", time.time() - t)
        t = time.time()

        # Perform measurement update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        print("---remove_lost_features    ", time.time() - t)
        t = time.time()

        self.prune_cam_state_buffer()

        print("---prune_cam_state_buffer  ", time.time() - t)
        print(
            "---msckf elapsed:          ",
            time.time() - start,
            f"({feature_msg.timestamp})",
        )

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()

    def initialize_gravity_and_bias(self):
        """
        Initialize the IMU bias and initial orientation based on the
        first few IMU readings.
        """
        # initialize gyro bias with mean of first 200 readings.
        self.state_server.imu_state.gyro_bias = np.mean(
            [imu_msg.angular_velocity for imu_msg in self.imu_msg_buffer]
        )

        # initialize gravity vector as normalized mean of first 200 accelerometer readings
        mean_acc = np.mean(
            [imu_msg.linear_acceleration for imu_msg in self.imu_msg_buffer]
        )
        self.state_server.imu_state.gravity = mean_acc / np.linalg.norm(mean_acc)

        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.
        # z should be aligned with gravity vector
        z_axis = self.state_server.imu_state.gravity
        x_axis = np.array([1.0, 0.0, 0.0])
        if abs(z_axis @ x_axis) > 0.9:
            x_axis = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        R_imu_wrt_world = np.column_stack((x_axis, y_axis, z_axis))
        self.state_server.imu_state.orientation = to_quaternion(R_imu_wrt_world)

        # rotate world gravity vector to IMU frame
        gravity_world = np.array([0.0, 0.0, -9.81])
        gravity_imu_expected = R_imu_wrt_world.T @ gravity_world

        # initialize accelerometer bias
        self.state_server.imu_state.acc_bias = mean_acc - gravity_imu_expected

    # Filter related functions
    # (batch_imu_processing, process_model, predict_new_state)
    def batch_imu_processing(self, time_bound):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Process the imu message given the time bound
        """
        # Process the imu messages in the imu_msg_buffer
        # Execute process model.
        # Update the state info
        # Repeat until the time_bound is reached
        i_message_used = 0

        for i, imu_msg in enumerate(self.imu_msg_buffer):
            # get current imu readings
            m_gyro = imu_msg.angular_velocity
            m_acc = imu_msg.linear_acceleration

            # check that message in question has timestamp of before the bound
            if imu_msg.timestamp < time_bound:
                # if it's the last message in the buffer
                if i == len(self.imu_msg_buffer) - 1:
                    dt = time_bound - imu_msg.timestamp

                else:
                    # look at the following message to compare the timestamps
                    next_msg = self.imu_msg_buffer[i + 1]
                    # if the next message will go over the time_bound, make the dt reflect that
                    if next_msg.timestamp > time_bound:
                        dt = time_bound - imu_msg.timestamp
                    else:
                        dt = next_msg.timestamp - imu_msg.timestamp

                # only process when valid dts
                if dt > 0:
                    self.process_model(dt, m_gyro, m_acc)
                    i_message_used += 1

                # time bound has been reached so stop processing so it doesn't go to the next one
                if imu_msg.timestamp + dt >= time_bound:
                    break

            # otherwise the message is not in bounds
            else:
                break

        # update the timestamp of the state
        self.state_server.imu_state.timestamp = time_bound

        # Set the current imu id to be the IMUState.next_id
        # IMUState.next_id increments
        self.state_server.imu_state.id = IMUState.next_id
        IMUState.next_id += 1

        # Remove all used IMU msgs.
        # remove messages from 0 to the lat one used
        self.imu_msg_buffer = self.imu_msg_buffer[max(0, i_message_used - 1) :]

    def process_model(self, dt, m_gyro, m_acc):
        """
        Section III.A: The dynamics of the error IMU state following equation (2) in the "MSCKF" paper.
        """
        # Get the error IMU state
        gyro = m_gyro - self.state_server.imu_state.gyro_bias
        acc = m_acc - self.state_server.imu_state.acc_bias

        # Compute discrete transition F, G matrices in Appendix A in "MSCKF" paper
        C = to_rotation(
            self.state_server.imu_state.orientation
        ).T  # Rotation from IMU to World

        F = np.zeros((15, 15))
        F[0:3, 0:3] = -skew(gyro)
        F[0:3, 3:6] = -np.eye(3)
        F[6:9, 0:3] = -C @ skew(acc)
        F[6:9, 9:12] = -C
        F[12:15, 6:9] = np.eye(3)

        G = np.zeros((15, 12))
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = -np.eye(3)
        G[6:9, 6:9] = -C
        G[9:12, 9:12] = -np.eye(3)

        # Approximate matrix exponential to the 3rd order, which can be
        # considered to be accurate enough assuming dt is within 0.01s.
        # continuous into discrete
        F_dt = F * dt
        F_dt2 = F_dt @ F_dt
        F_dt3 = F_dt2 @ F_dt

        # need phi instead of F because you need to account for how the rate of change changes as the state progresses
        Phi = np.eye(15) + F_dt + F_dt2 / 2 + F_dt3 / 6

        # Propogate the state using 4th order Runge-Kutta
        self.predict_new_state(dt, gyro, acc)

        # Modify the transition matrix
        # the imu changes, but not the transformation between the camera and the IMU. Still estimate extrinsics even if they don't change
        Phi_full = np.eye(21)
        Phi_full[0:15, 0:15] = Phi

        Q_c = self.state_server.continuous_noise_cov
        Q_d = Phi @ G @ Q_c @ G.T @ Phi.T * dt

        Q_d_full = np.zeros((21, 21))
        Q_d_full[0:15, 0:15] = Q_d

        # Propogate the state covariance matrix.
        state_cov = self.state_server.state_cov
        self.state_server.state_cov = Phi_full @ state_cov @ Phi_full.T + Q_d_full

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (
            self.state_server.state_cov + self.state_server.state_cov.T
        ) / 2.0

        # Update the state correspondes to null space.
        self.state_server.imu_state.orientation_null = (
            self.state_server.imu_state.orientation
        )
        self.state_server.imu_state.position_null = self.state_server.imu_state.position
        self.state_server.imu_state.velocity_null = self.state_server.imu_state.velocity

    def predict_new_state(self, dt, gyro, acc):
        """Propogate the state using 4th order Runge-Kutta for equation (1) in "MSCKF" paper"""

        # Get the Omega matrix, the equation above equation (2) in "MSCKF" paper
        omega = np.zeros((4, 4))
        omega[:3, :3] = -skew(gyro)
        omega[:3, 3] = gyro
        omega[3, :3] = -gyro

        def f(y):
            """derivative of the state vector"""
            q, v = y[0:4], y[4:7]

            # dq/dt
            q_dot = 0.5 * omega @ q

            # dv/dt
            R_imu_wrt_world = to_rotation(q)
            v_dot = R_imu_wrt_world @ acc + self.state_server.imu_state.gravity

            return np.concatenate((q_dot, v_dot, v))

        # Get the orientation, velocity, position
        y0 = np.concatenate(
            (
                self.state_server.imu_state.orientation,
                self.state_server.imu_state.velocity,
                self.state_server.imu_state.position,
            )
        )

        # Apply 4th order Runge-Kutta
        # k1 = f(tn, yn)
        k1 = f(y0)

        # k2 = f(tn+dt/2, yn+k1*dt/2)
        k2 = f(y0 + k1 * dt / 2)

        # k3 = f(tn+dt/2, yn+k2*dt/2)
        k3 = f(y0 + k2 * dt / 2)

        # k4 = f(tn+dt, yn+k3*dt)
        k4 = f(y0 + k3 * dt)

        # yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        y_new = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # update the imu state
        q_new = y_new[0:4]
        self.state_server.imu_state.orientation = q_new / np.linalg.norm(q_new)
        self.state_server.imu_state.velocity = y_new[4:7]
        self.state_server.imu_state.position = y_new[7:10]

    def state_augmentation(self, time):
        """
        When a new image is received, we need to augment the state vector by adding a new camera state.
        """
        # get the current imu_state
        q_imu = self.state_server.imu_state.orientation
        p_imu = self.state_server.imu_state.position

        # get pose of the new camera state using IMU->camera tranformation
        R_imu_wrt_world = to_rotation(q_imu)
        R_cam0_wrt_imu = self.state_server.imu_state.R_imu_wrt_cam0.T
        R_cam_wrt_world = R_imu_wrt_world @ R_cam0_wrt_imu
        q_cam = to_quaternion(R_cam_wrt_world)  # orientation of the new camera state

        t_cam0_wrt_imu = self.state_server.imu_state.t_cam0_wrt_imu
        p_cam = (
            p_imu + R_imu_wrt_world @ t_cam0_wrt_imu
        )  # position of the new camera state

        # Add a new camera state to the state server.
        cam_state = CAMState(self.state_server.imu_state.id)
        cam_state.timestamp = time
        cam_state.orientation = q_cam
        cam_state.position = p_cam

        old_cam_len = len(self.state_server.cam_states)
        self.state_server.cam_states[cam_state.id] = cam_state

        # Update the covariance matrix of the state, which contains the
        #   partial derivatives of the camera pose equations w.r.t. the IMU error state vector
        # To simplify computation, the matrix J below is the nontrivial block
        # Appendix B of "MSCKF" paper.
        J = np.zeros((6, 21 + 6 * old_cam_len))
        J[:3, :3] = R_cam0_wrt_imu
        J[3:, :3] = -R_cam0_wrt_imu @ skew(t_cam0_wrt_imu)
        J[3:, 12:15] = np.identity(3)
        J[:3, 15:18] = np.identity(3)
        J[3:, 18:21] = np.identity(3)

        # Resize the state covariance matrix to accommodate the new camera state.
        P = self.state_server.state_cov
        P_aug = np.zeros((P.shape[0] + 6, P.shape[1] + 6))

        # Fill in the augmented state covariance.
        P_aug[: P.shape[0], : P.shape[1]] = P
        P_aug[P.shape[0] :, : P.shape[1]] = J @ P
        P_aug[: P.shape[0], P.shape[1] :] = P @ J.T
        P_aug[P.shape[0] :, P.shape[1] :] = J @ P @ J.T

        # Fix the covariance to be symmetric
        P_aug = (P_aug + P_aug.T) / 2

        self.state_server.state_cov = P_aug

    def add_feature_observations(self, feature_msg):
        # get the current imu state id and number of current features
        imu_state_id = self.state_server.imu_state.id
        num_features = len(self.map_server)

        # keep track of known features getting tracked in the current frame
        num_tracked = 0

        # add all features in the feature_msg to self.map_server
        for feature in feature_msg.features:
            if feature.id not in self.map_server:
                # addding a new feature to map server begins tracking it
                self.map_server[feature.id] = Feature(feature.id)
            else:
                num_tracked += 1
            # add the observation to the feature in the map server
            self.map_server[feature.id].observations[imu_state_id] = feature.observation

        # update the tracking rate
        if num_features == 0:
            self.tracking_rate = 1.0
        else:
            self.tracking_rate = num_tracked / num_features

    def measurement_jacobian(self, cam_state_id, feature_id):
        """
        This function is used to compute the measurement Jacobian
        for a single feature observed at a single camera frame.
        """
        # Prepare all the required data.
        cam_state = self.state_server.cam_states[cam_state_id]
        feature = self.map_server[feature_id]

        # Cam0 pose.
        R_world_wrt_cam0 = to_rotation(cam_state.orientation)
        t_cam0_wrt_world = cam_state.position

        # Cam1 pose.
        R_world_wrt_cam1 = CAMState.R_cam0_wrt_cam1 @ R_world_wrt_cam0
        t_cam1_wrt_world = (
            t_cam0_wrt_world - R_world_wrt_cam1.T @ CAMState.t_cam0_wrt_cam1
        )

        # 3d feature position in the world frame.
        # And its observation with the stereo cameras.
        p_w = feature.position
        z = feature.observations[cam_state_id]

        # Convert the feature position from the world frame to
        # the cam0 and cam1 frame.
        p_c0 = R_world_wrt_cam0 @ (p_w - t_cam0_wrt_world)
        p_c1 = R_world_wrt_cam1 @ (p_w - t_cam1_wrt_world)

        # Compute the Jacobians.
        dz_dpc0 = np.zeros((4, 3))
        dz_dpc0[0, 0] = 1 / p_c0[2]
        dz_dpc0[1, 1] = 1 / p_c0[2]
        dz_dpc0[0, 2] = -p_c0[0] / (p_c0[2] * p_c0[2])
        dz_dpc0[1, 2] = -p_c0[1] / (p_c0[2] * p_c0[2])

        dz_dpc1 = np.zeros((4, 3))
        dz_dpc1[2, 0] = 1 / p_c1[2]
        dz_dpc1[3, 1] = 1 / p_c1[2]
        dz_dpc1[2, 2] = -p_c1[0] / (p_c1[2] * p_c1[2])
        dz_dpc1[3, 2] = -p_c1[1] / (p_c1[2] * p_c1[2])

        dpc0_dxc = np.zeros((3, 6))
        dpc0_dxc[:, :3] = skew(p_c0)
        dpc0_dxc[:, 3:] = -R_world_wrt_cam0

        dpc1_dxc = np.zeros((3, 6))
        dpc1_dxc[:, :3] = CAMState.R_cam0_wrt_cam1 @ skew(p_c0)
        dpc1_dxc[:, 3:] = -R_world_wrt_cam1

        dpc0_dpg = R_world_wrt_cam0
        dpc1_dpg = R_world_wrt_cam1

        H_x = dz_dpc0 @ dpc0_dxc + dz_dpc1 @ dpc1_dxc  # shape: (4, 6)
        H_f = dz_dpc0 @ dpc0_dpg + dz_dpc1 @ dpc1_dpg  # shape: (4, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x  # shape: (4, 6)
        u = np.zeros(6)
        u[:3] = to_rotation(cam_state.orientation_null) @ IMUState.gravity
        u[3:] = skew(p_w - cam_state.position_null) @ IMUState.gravity

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:4, 3:6]

        # Compute the residual.
        r = z - np.array([*p_c0[:2] / p_c0[2], *p_c1[:2] / p_c1[2]])

        # H_x: shape (4, 6)
        # H_f: shape (4, 3)
        # r  : shape (4,)
        return H_x, H_f, r

    def feature_jacobian(self, feature_id, cam_state_ids):
        """
        This function computes the Jacobian of all measurements viewed
        in the given camera states of this feature.
        """
        feature = self.map_server[feature_id]

        # Check how many camera states in the provided camera id
        # camera has actually seen this feature.
        valid_cam_state_ids = []
        for cam_id in cam_state_ids:
            if cam_id in feature.observations:
                valid_cam_state_ids.append(cam_id)

        jacobian_row_size = 4 * len(valid_cam_state_ids)

        cam_states = self.state_server.cam_states
        H_xj = np.zeros((jacobian_row_size, 21 + len(self.state_server.cam_states) * 6))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count : stack_count + 4, 21 + 6 * idx : 21 + 6 * (idx + 1)] = (
                H_xi
            )
            H_fj[stack_count : stack_count + 4, :3] = H_fi
            r_j[stack_count : stack_count + 4] = r_i
            stack_count += 4

        # Project the residual and Jacobians onto the nullspace of H_fj.
        # svd of H_fj
        U, _, _ = np.linalg.svd(H_fj)
        A = U[:, 3:]

        H_x = A.T @ H_xj
        r = A.T @ r_j

        return H_x, r

    def measurement_update(self, H, r):
        """
        Section III.B: by stacking multiple observations, we can compute the residuals in equation (6) in "MSCKF" paper
        """
        # Check if H and r are empty
        if H.shape[0] == 0 or r.shape[0] == 0:
            return

        # Decompose the final Jacobian matrix to reduce computational
        # complexity.
        Q, R = np.linalg.qr(H)
        r_thin = Q.T @ r  # residual in the reduced space
        H_thin = R  # measurement matrix in the reduced space
        R_noise = self.config.observation_noise * np.identity(
            H_thin.shape[0]
        )  # observation noise in the reduced space

        # Compute the Kalman gain, which determines how much we should trust the measurement vs. the current state estimate.
        # matrix K maps measurement errors in pixel space to errors in the state space
        P = self.state_server.state_cov
        K = P @ H_thin.T @ np.linalg.inv(H_thin @ P @ H_thin.T + R_noise)

        # Compute the error of the state.
        state_err = K @ r_thin

        # Update the IMU state.
        # update orientation
        rotation_err = state_err[0:3]
        q_err = to_quaternion(to_rotation(rotation_err))
        q_new = quaternion_multiplication(
            q_err, self.state_server.imu_state.orientation
        )
        self.state_server.imu_state.orientation = q_new / np.linalg.norm(q_new)

        # update biases, velocity, and position
        self.state_server.imu_state.gyro_bias += state_err[3:6]
        self.state_server.imu_state.velocity += state_err[6:9]
        self.state_server.imu_state.acc_bias += state_err[9:12]
        self.state_server.imu_state.position += state_err[12:15]

        # update extrinsics
        ext_rot_err = state_err[15:18]
        q_ext_err = to_quaternion(to_rotation(ext_rot_err))
        self.state_server.imu_state.R_imu_wrt_cam0 = (
            to_rotation(q_ext_err) @ self.state_server.imu_state.R_imu_wrt_cam0
        )
        self.state_server.imu_state.t_cam0_wrt_imu += state_err[18:21]

        # Update the camera states.
        cam_state_ids = list(self.state_server.cam_states.keys())
        for i, cam_state_id in enumerate(cam_state_ids):
            # get the error of the camera state
            cam_state_err = state_err[21 + 6 * i : 21 + 6 * (i + 1)]
            cam_rot_err = cam_state_err[0:3]
            cam_pos_err = cam_state_err[3:6]
            # update position
            self.state_server.cam_states[cam_state_id].position += cam_pos_err
            # update rotation
            q_err = to_quaternion(to_rotation(cam_rot_err))
            q_new = quaternion_multiplication(
                q_err, self.state_server.cam_states[cam_state_id].orientation
            )
            self.state_server.cam_states[cam_state_id].orientation = (
                q_new / np.linalg.norm(q_new)
            )

        # Update state covariance.
        I = np.eye(P.shape[0])
        self.state_server.state_cov = (I - K @ H_thin) @ P

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (
            self.state_server.state_cov + self.state_server.state_cov.T
        ) / 2

    def gating_test(self, H, r, dof):
        P1 = H @ self.state_server.state_cov @ H.T
        P2 = self.config.observation_noise * np.identity(len(H))
        gamma = r @ np.linalg.solve(P1 + P2, r)

        if gamma < self.chi_squared_test_table[dof]:
            return True
        else:
            return False

    def remove_lost_features(self):
        # Remove the features that lost track.
        # BTW, find the size the final Jacobian matrix and residual vector.
        jacobian_row_size = 0
        invalid_feature_ids = []
        processed_feature_ids = []

        for feature in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server.imu_state.id in feature.observations:
                continue
            if len(feature.observations) < 3:
                invalid_feature_ids.append(feature.id)
                continue

            # Check if the feature can be initialized if it has not been.
            if not feature.is_initialized:
                # Ensure there is enough translation to triangulate the feature
                if not feature.check_motion(self.state_server.cam_states):
                    invalid_feature_ids.append(feature.id)
                    continue

                # Intialize the feature position based on all current available
                # measurements.
                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    invalid_feature_ids.append(feature.id)
                    continue

            jacobian_row_size += 4 * len(feature.observations) - 3
            processed_feature_ids.append(feature.id)

        # Remove the features that do not have enough measurements.
        for feature_id in invalid_feature_ids:
            del self.map_server[feature_id]

        # Return if there is no lost feature to be processed.
        if len(processed_feature_ids) == 0:
            return

        H_x = np.zeros((jacobian_row_size, 21 + 6 * len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids) - 1):
                H_x[stack_count : stack_count + H_xj.shape[0], : H_xj.shape[1]] = H_xj
                r[stack_count : stack_count + len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            # Put an upper bound on the row size of measurement Jacobian,
            # which helps guarantee the executation time.
            if stack_count > 1500:
                break

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform the measurement update step.
        self.measurement_update(H_x, r)

        # Remove all processed features from the map.
        for feature_id in processed_feature_ids:
            del self.map_server[feature_id]

    def find_redundant_cam_states(self):
        # Move the iterator to the key position.
        cam_state_pairs = list(self.state_server.cam_states.items())

        key_cam_state_idx = len(cam_state_pairs) - 4
        cam_state_idx = key_cam_state_idx + 1
        first_cam_state_idx = 0

        # Pose of the key camera state.
        key_position = cam_state_pairs[key_cam_state_idx][1].position
        key_rotation = to_rotation(cam_state_pairs[key_cam_state_idx][1].orientation)

        rm_cam_state_ids = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for i in range(2):
            position = cam_state_pairs[cam_state_idx][1].position
            rotation = to_rotation(cam_state_pairs[cam_state_idx][1].orientation)

            distance = np.linalg.norm(position - key_position)
            angle = 2 * np.arccos(to_quaternion(rotation @ key_rotation.T)[-1])

            if angle < 0.2618 and distance < 0.4 and self.tracking_rate > 0.5:
                rm_cam_state_ids.append(cam_state_pairs[cam_state_idx][0])
                cam_state_idx += 1
            else:
                rm_cam_state_ids.append(cam_state_pairs[first_cam_state_idx][0])
                first_cam_state_idx += 1
                cam_state_idx += 1

        # Sort the elements in the output list.
        rm_cam_state_ids = sorted(rm_cam_state_ids)
        return rm_cam_state_ids

    def prune_cam_state_buffer(self):
        if len(self.state_server.cam_states) < self.config.max_cam_state_size:
            return

        # Find two camera states to be removed.
        rm_cam_state_ids = self.find_redundant_cam_states()

        # Find the size of the Jacobian matrix.
        jacobian_row_size = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue
            if len(involved_cam_state_ids) == 1:
                del feature.observations[involved_cam_state_ids[0]]
                continue

            if not feature.is_initialized:
                # Check if the feature can be initialize.
                if not feature.check_motion(self.state_server.cam_states):
                    # If the feature cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

            jacobian_row_size += 4 * len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        H_x = np.zeros((jacobian_row_size, 21 + 6 * len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)

        stack_count = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue

            H_xj, r_j = self.feature_jacobian(feature.id, involved_cam_state_ids)

            if self.gating_test(H_xj, r_j, len(involved_cam_state_ids)):
                H_x[stack_count : stack_count + H_xj.shape[0], : H_xj.shape[1]] = H_xj
                r[stack_count : stack_count + len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r)

        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6 * idx
            cam_state_end = cam_state_start + 6

            # Remove the corresponding rows and columns in the state
            # covariance matrix.
            state_cov = self.state_server.state_cov.copy()
            if cam_state_end < state_cov.shape[0]:
                size = state_cov.shape[0]
                state_cov[cam_state_start:-6, :] = state_cov[cam_state_end:, :]
                state_cov[:, cam_state_start:-6] = state_cov[:, cam_state_end:]
            self.state_server.state_cov = state_cov[:-6, :-6]

            # Remove this camera state in the state vector.
            del self.state_server.cam_states[cam_id]

    def reset_state_cov(self):
        """
        Reset the state covariance.
        """
        state_cov = np.zeros((21, 21))
        state_cov[3:6, 3:6] = self.config.gyro_bias_cov * np.identity(3)
        state_cov[6:9, 6:9] = self.config.velocity_cov * np.identity(3)
        state_cov[9:12, 9:12] = self.config.acc_bias_cov * np.identity(3)
        state_cov[15:18, 15:18] = self.config.extrinsic_rotation_cov * np.identity(3)
        state_cov[18:21, 18:21] = self.config.extrinsic_translation_cov * np.identity(3)
        self.state_server.state_cov = state_cov

    def reset(self):
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        imu_state = IMUState()
        imu_state.id = self.state_server.imu_state.id
        imu_state.R_imu_wrt_cam0 = self.state_server.imu_state.R_imu_wrt_cam0
        imu_state.t_cam0_wrt_imu = self.state_server.imu_state.t_cam0_wrt_imu
        self.state_server.imu_state = imu_state

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Reset the state covariance.
        self.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU msg buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self):
        """
        Reset the system online if the uncertainty is too large.
        """
        # Never perform online reset if position std threshold is non-positive.
        if self.config.position_std_threshold <= 0:
            return

        # Check the uncertainty of positions to determine if
        # the system can be reset.
        position_x_std = np.sqrt(self.state_server.state_cov[12, 12])
        position_y_std = np.sqrt(self.state_server.state_cov[13, 13])
        position_z_std = np.sqrt(self.state_server.state_cov[14, 14])

        if (
            max(position_x_std, position_y_std, position_z_std)
            < self.config.position_std_threshold
        ):
            return

        print("Start online reset...")

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.reset_state_cov()

    def publish(self, time):
        imu_state = self.state_server.imu_state
        print("+++publish:")
        print("   timestamp:", imu_state.timestamp)
        print("   orientation:", imu_state.orientation)
        print("   position:", imu_state.position)
        print("   velocity:", imu_state.velocity)
        print()

        T_imu_wrt_world = Isometry3d(
            to_rotation(imu_state.orientation).T, imu_state.position
        )
        T_body_wrt_world = (
            IMUState.T_imu_wrt_body
            * T_imu_wrt_world
            * IMUState.T_imu_wrt_body.inverse()
        )
        body_velocity = IMUState.T_imu_wrt_body.R @ imu_state.velocity

        R_world_wrt_cam = imu_state.R_imu_wrt_cam0 @ T_imu_wrt_world.R.T
        t_cam_wrt_world = (
            imu_state.position + T_imu_wrt_world.R @ imu_state.t_cam0_wrt_imu
        )
        T_cam_wrt_world = Isometry3d(R_world_wrt_cam.T, t_cam_wrt_world)

        return namedtuple("vio_result", ["timestamp", "pose", "velocity", "cam0_pose"])(
            time, T_body_wrt_world, body_velocity, T_cam_wrt_world
        )
