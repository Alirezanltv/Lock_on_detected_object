import cv2
import numpy as np
from ultralytics import YOLO
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class BYTETracker:
    track_thresh: float = 0.1
    track_buffer: int = 30
    match_thresh: float = 0.01
    min_box_area: float = 10
    mot20: bool = False
    max_age: int = 20
    min_hits: int = 1
    motion_history_size: int = 15
    turn_threshold: float = 0.05
    maneuver_compensation: float = 6.0
    max_angular_velocity: float = 1.5
    prediction_frames: int = 6
    turn_prediction_weight: float = 0.9
    velocity_prediction_weight: float = 0.1
    recovery_threshold: float = 0.1
    max_recovery_frames: int = 15
    turn_pattern_weight: float = 0.7
    acceleration_weight: float = 0.8
    velocity_smoothing: float = 0.2
    # New parameters for multi-bird handling
    bird_separation_threshold: float = 50.0
    flocking_weight: float = 0.3
    interaction_radius: float = 100.0
    confidence_decay: float = 0.85
    min_confidence: float = 0.4

    def __post_init__(self):
        self.tracks = []
        self.track_id = 0
        self.frame_id = 0
        self.kalman_filters = {}
        self.motion_history = {}
        self.turn_history = {}
        self.velocity_history = {}
        self.turn_patterns = {}
        self.recovery_attempts = {}
        self.acceleration_history = {}
        self.turn_sequence = {}
        # New attributes for multi-bird handling
        self.flock_centers = {}
        self.bird_interactions = {}
        self.track_confidences = {}
        self.flight_patterns = {}

    def init_kalman_filter(self, track_id):
        kf = cv2.KalmanFilter(8, 4)  # 8 states (x, y, w, h, vx, vy, ax, ay), 4 measurements
        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0.5, 0],
                                      [0, 1, 0, 0, 0, 1, 0, 0.5],
                                      [0, 0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1, 0, 1],
                                      [0, 0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        # Set process noise covariance
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        # Set measurement noise covariance
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        return kf

    def detect_turn(self, track_id):
        """Enhanced turn detection with pattern recognition"""
        if track_id not in self.motion_history or len(self.motion_history[track_id]) < 3:
            return False, 0, 0
        
        history = self.motion_history[track_id]
        if len(history) < 3:
            return False, 0, 0
        
        # Calculate direction changes with velocity consideration
        directions = []
        velocities = []
        accelerations = []
        angular_velocities = []
        
        for i in range(1, len(history)):
            dx = history[i][0] - history[i-1][0]
            dy = history[i][1] - history[i-1][1]
            angle = np.arctan2(dy, dx)
            velocity = np.sqrt(dx**2 + dy**2)
            directions.append(angle)
            velocities.append(velocity)
            
            if i > 1:
                acceleration = velocity - velocities[-2]
                accelerations.append(acceleration)
                
                # Calculate angular velocity
                angle_diff = angle - directions[-2]
                if angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                elif angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                angular_velocities.append(angle_diff)
        
        # Calculate changes and patterns
        angle_changes = []
        velocity_changes = []
        acceleration_changes = []
        
        for i in range(1, len(directions)):
            # Calculate angle change
            change = abs(directions[i] - directions[i-1])
            if change > np.pi:
                change = 2 * np.pi - change
            angle_changes.append(change)
            
            # Calculate velocity change
            vel_change = abs(velocities[i] - velocities[i-1])
            velocity_changes.append(vel_change)
            
            # Calculate acceleration change if available
            if i < len(accelerations):
                acc_change = abs(accelerations[i] - accelerations[i-1])
                acceleration_changes.append(acc_change)
        
        # Enhanced turn detection using multiple factors
        if angle_changes and velocity_changes:
            max_angle_change = max(angle_changes)
            max_vel_change = max(velocity_changes)
            mean_velocity = np.mean(velocities)
            
            # More aggressive turn detection with multiple factors
            is_turning = (max_angle_change > self.turn_threshold or 
                         max_vel_change > mean_velocity * 0.15 or
                         (acceleration_changes and max(acceleration_changes) > mean_velocity * 0.2) or
                         (angular_velocities and abs(angular_velocities[-1]) > self.max_angular_velocity * 0.8))
            
            # Calculate turn magnitude considering all factors
            turn_magnitude = max_angle_change * (1 + max_vel_change / mean_velocity)
            if acceleration_changes:
                turn_magnitude *= (1 + max(acceleration_changes) / mean_velocity)
            if angular_velocities:
                turn_magnitude *= (1 + abs(angular_velocities[-1]) / self.max_angular_velocity)
            
            # Predict turn direction with enhanced accuracy
            turn_direction = 0
            if len(angle_changes) > 0:
                last_angle_change = angle_changes[-1]
                if last_angle_change > 0:
                    # Consider both current and previous directions
                    current_direction = directions[-1]
                    prev_direction = directions[-2]
                    turn_direction = 1 if current_direction > prev_direction else -1
                    
                    # Adjust direction based on multiple factors
                    if acceleration_changes and acceleration_changes[-1] > 0:
                        turn_direction *= 1.3  # Increased amplification during acceleration
                    if angular_velocities:
                        turn_direction *= (1 + abs(angular_velocities[-1]) / self.max_angular_velocity)
            
            # Update turn sequence for pattern recognition
            if track_id not in self.turn_sequence:
                self.turn_sequence[track_id] = []
            self.turn_sequence[track_id].append({
                'direction': turn_direction,
                'magnitude': turn_magnitude,
                'velocity': mean_velocity
            })
            if len(self.turn_sequence[track_id]) > 5:
                self.turn_sequence[track_id].pop(0)
            
            return is_turning, turn_magnitude, turn_direction
        
        return False, 0, 0

    def calculate_flock_center(self, tracks):
        """Calculate the center of the flock"""
        if not tracks:
            return None
        
        centers = []
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])
        
        return np.mean(centers, axis=0)

    def update_bird_interactions(self, track_id, tracks):
        """Update bird interaction data"""
        if track_id not in self.bird_interactions:
            self.bird_interactions[track_id] = []
        
        x1, y1, x2, y2 = tracks[track_id]['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        interactions = []
        for other_id, other_track in enumerate(tracks):
            if other_id != track_id:
                ox1, oy1, ox2, oy2 = other_track['bbox']
                other_center_x = (ox1 + ox2) / 2
                other_center_y = (oy1 + oy2) / 2
                
                distance = np.sqrt((center_x - other_center_x)**2 + 
                                 (center_y - other_center_y)**2)
                
                if distance < self.interaction_radius:
                    interactions.append({
                        'track_id': other_id,
                        'distance': distance,
                        'relative_velocity': self.calculate_relative_velocity(
                            tracks[track_id], other_track)
                    })
        
        self.bird_interactions[track_id] = interactions

    def calculate_relative_velocity(self, track1, track2):
        """Calculate relative velocity between two tracks"""
        if 'velocity' not in track1 or 'velocity' not in track2:
            return [0, 0]
        
        v1 = track1['velocity']
        v2 = track2['velocity']
        return [v1[0] - v2[0], v1[1] - v2[1]]

    def update_flight_pattern(self, track_id, track):
        """Update flight pattern for a track"""
        if track_id not in self.flight_patterns:
            self.flight_patterns[track_id] = {
                'straight_flight': 0,
                'turns': 0,
                'maneuvers': 0,
                'last_pattern': 'straight_flight'
            }
        
        is_turning, turn_magnitude, _ = self.detect_turn(track_id)
        
        if is_turning:
            if turn_magnitude > self.turn_threshold * 2:
                self.flight_patterns[track_id]['maneuvers'] += 1
                self.flight_patterns[track_id]['last_pattern'] = 'maneuver'
            else:
                self.flight_patterns[track_id]['turns'] += 1
                self.flight_patterns[track_id]['last_pattern'] = 'turn'
        else:
            self.flight_patterns[track_id]['straight_flight'] += 1
            self.flight_patterns[track_id]['last_pattern'] = 'straight_flight'

    def update_track_confidence(self, track_id, detection_confidence):
        """Update track confidence score"""
        if track_id not in self.track_confidences:
            self.track_confidences[track_id] = detection_confidence
        else:
            # Update confidence with decay
            self.track_confidences[track_id] = (
                self.confidence_decay * self.track_confidences[track_id] +
                (1 - self.confidence_decay) * detection_confidence
            )

    def predict_next_position(self, track):
        track_id = track['id']
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self.init_kalman_filter(track_id)
            self.motion_history[track_id] = []
            self.turn_history[track_id] = []
            self.velocity_history[track_id] = []
            self.turn_patterns[track_id] = []
            self.recovery_attempts[track_id] = 0
            self.acceleration_history[track_id] = []
            self.turn_sequence[track_id] = []
        
        kf = self.kalman_filters[track_id]
        
        # Get current bbox
        x1, y1, x2, y2 = track['bbox']
        w = x2 - x1
        h = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Prepare measurement
        measurement = np.array([[center_x], [center_y], [w], [h]], np.float32)
        
        # Update Kalman filter
        kf.correct(measurement)
        prediction = kf.predict()
        
        # Extract predicted position
        pred_x = prediction[0][0]
        pred_y = prediction[1][0]
        pred_w = prediction[2][0]
        pred_h = prediction[3][0]
        
        # Update motion history
        self.motion_history[track_id].append((pred_x, pred_y))
        if len(self.motion_history[track_id]) > self.motion_history_size:
            self.motion_history[track_id].pop(0)
        
        # Enhanced turn detection and compensation
        is_turning, turn_magnitude, turn_direction = self.detect_turn(track_id)
        
        # Update flight pattern
        self.update_flight_pattern(track_id, track)
        
        # Apply flocking behavior if other birds are nearby
        if track_id in self.bird_interactions and self.bird_interactions[track_id]:
            flock_center = self.calculate_flock_center(self.tracks)
            if flock_center is not None:
                # Calculate vector to flock center
                dx = flock_center[0] - pred_x
                dy = flock_center[1] - pred_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > self.bird_separation_threshold:
                    # Apply flocking force
                    force = self.flocking_weight * (distance - self.bird_separation_threshold)
                    pred_x += (dx / distance) * force
                    pred_y += (dy / distance) * force
        
        if is_turning:
            # Apply enhanced turn compensation
            compensation = self.maneuver_compensation * turn_magnitude
            
            # Get velocity and acceleration
            if 'velocity' in track and 'acceleration' in track:
                vx, vy = track['velocity']
                ax, ay = track['acceleration']
                
                # Calculate predicted positions for multiple frames ahead
                for i in range(self.prediction_frames):
                    # Apply compensation with acceleration and turn direction
                    turn_factor = 1 + compensation * (i + 1) * turn_direction
                    
                    # Add pattern-based prediction
                    if track_id in self.turn_sequence and self.turn_sequence[track_id]:
                        pattern = self.turn_sequence[track_id][-1]
                        pattern_factor = pattern['direction'] * pattern['magnitude'] * self.turn_pattern_weight
                        turn_factor *= (1 + pattern_factor)
                    
                    # Apply enhanced prediction model
                    pred_x += (vx * turn_factor + 0.5 * ax * (i + 1)**2) * self.acceleration_weight
                    pred_y += (vy * turn_factor + 0.5 * ay * (i + 1)**2) * self.acceleration_weight
                    
                    # Add additional compensation based on turn history
                    if track_id in self.turn_history and self.turn_history[track_id]:
                        last_turn = self.turn_history[track_id][-1]
                        if last_turn['direction'] == turn_direction:
                            pred_x += vx * last_turn['magnitude'] * 0.7
                            pred_y += vy * last_turn['magnitude'] * 0.7
                
                # Update turn history with more information
                self.turn_history[track_id].append({
                    'frame': self.frame_id,
                    'magnitude': turn_magnitude,
                    'direction': turn_direction,
                    'velocity': [vx, vy],
                    'acceleration': [ax, ay]
                })
                if len(self.turn_history[track_id]) > 5:
                    self.turn_history[track_id].pop(0)
                
                # Update turn patterns
                self.turn_patterns[track_id].append({
                    'magnitude': turn_magnitude,
                    'direction': turn_direction,
                    'velocity': [vx, vy]
                })
                if len(self.turn_patterns[track_id]) > 3:
                    self.turn_patterns[track_id].pop(0)
        
        return [pred_x - pred_w/2, pred_y - pred_h/2, 
                pred_x + pred_w/2, pred_y + pred_h/2]

    def calculate_velocity(self, track, new_bbox):
        if 'last_bbox' not in track:
            track['last_bbox'] = new_bbox
            track['velocity'] = [0, 0]
            track['acceleration'] = [0, 0]
            track['angular_velocity'] = 0
            return
        
        # Calculate center points
        old_x1, old_y1, old_x2, old_y2 = track['last_bbox']
        new_x1, new_y1, new_x2, new_y2 = new_bbox
        
        old_center_x = (old_x1 + old_x2) / 2
        old_center_y = (old_y1 + old_y2) / 2
        new_center_x = (new_x1 + new_x2) / 2
        new_center_y = (new_y1 + new_y2) / 2
        
        # Calculate velocity and acceleration
        vx = new_center_x - old_center_x
        vy = new_center_y - old_center_y
        
        # Enhanced angular velocity calculation
        if 'velocity' in track:
            old_vx, old_vy = track['velocity']
            # Calculate angle between old and new velocity vectors
            old_angle = np.arctan2(old_vy, old_vx)
            new_angle = np.arctan2(vy, vx)
            angle_diff = new_angle - old_angle
            
            # Normalize angle difference to [-pi, pi]
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # Limit angular velocity to prevent extreme values
            angle_diff = np.clip(angle_diff, -self.max_angular_velocity, self.max_angular_velocity)
            track['angular_velocity'] = angle_diff
        else:
            track['angular_velocity'] = 0
        
        # Calculate acceleration with reduced smoothing
        if 'velocity' in track:
            ax = vx - track['velocity'][0]
            ay = vy - track['velocity'][1]
        else:
            ax = ay = 0
        
        # Enhanced adaptive smoothing with reduced smoothing factor
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        acceleration_magnitude = np.sqrt(ax**2 + ay**2)
        angular_velocity_magnitude = abs(track['angular_velocity'])
        
        # More responsive smoothing during turns
        alpha = min(0.95, self.velocity_smoothing + 
                   velocity_magnitude * 0.1 + 
                   acceleration_magnitude * 0.1 + 
                   angular_velocity_magnitude * 0.3)
        
        # Update velocity and acceleration with reduced smoothing
        track['velocity'] = [
            alpha * vx + (1 - alpha) * track['velocity'][0],
            alpha * vy + (1 - alpha) * track['velocity'][1]
        ]
        track['acceleration'] = [
            alpha * ax + (1 - alpha) * track['acceleration'][0],
            alpha * ay + (1 - alpha) * track['acceleration'][1]
        ]
        
        # Store velocity and acceleration history
        track_id = track['id']
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = []
        if track_id not in self.acceleration_history:
            self.acceleration_history[track_id] = []
            
        self.velocity_history[track_id].append(track['velocity'])
        self.acceleration_history[track_id].append(track['acceleration'])
        
        if len(self.velocity_history[track_id]) > self.motion_history_size:
            self.velocity_history[track_id].pop(0)
        if len(self.acceleration_history[track_id]) > self.motion_history_size:
            self.acceleration_history[track_id].pop(0)
        
        track['last_bbox'] = new_bbox

    def recover_track(self, track):
        """Enhanced track recovery with pattern-based prediction"""
        track_id = track['id']
        if track_id not in self.motion_history or not self.motion_history[track_id]:
            return track['bbox']
        
        # Get last known position and motion parameters
        last_pos = self.motion_history[track_id][-1]
        if 'velocity' in track and 'acceleration' in track:
            vx, vy = track['velocity']
            ax, ay = track['acceleration']
            angular_velocity = track['angular_velocity']
        else:
            vx = vy = ax = ay = angular_velocity = 0
        
        # Predict next position using enhanced motion model
        frames_missed = self.frame_id - track['last_seen']
        if frames_missed > 0 and frames_missed <= self.max_recovery_frames:
            # Check track confidence before recovery
            if track_id in self.track_confidences and self.track_confidences[track_id] < self.min_confidence:
                return None  # Return None to indicate track should be terminated
            
            # Enhanced turn detection and compensation
            is_turning, turn_magnitude, turn_direction = self.detect_turn(track_id)
            
            if is_turning:
                # Apply enhanced turn compensation with reduced magnitude
                compensation = self.maneuver_compensation * turn_magnitude * 0.7  # Reduced by 30%
                
                # Use pattern-based prediction if available
                if track_id in self.turn_patterns and self.turn_patterns[track_id]:
                    pattern = self.turn_patterns[track_id][-1]
                    pattern_weight = self.turn_prediction_weight * 0.8  # Reduced by 20%
                    
                    # Combine pattern-based and velocity-based predictions
                    pred_x = last_pos[0] + (vx * frames_missed * (1 + compensation * turn_direction) + 
                                          pattern['velocity'][0] * frames_missed * pattern_weight)
                    pred_y = last_pos[1] + (vy * frames_missed * (1 + compensation * turn_direction) + 
                                          pattern['velocity'][1] * frames_missed * pattern_weight)
                else:
                    # Use standard turn compensation with reduced magnitude
                    pred_x = last_pos[0] + vx * frames_missed * (1 + compensation * turn_direction * 0.8)
                    pred_y = last_pos[1] + vy * frames_missed * (1 + compensation * turn_direction * 0.8)
                
                # Add reduced acceleration and angular velocity compensation
                pred_x += 0.5 * ax * frames_missed**2 * 0.7 + vx * frames_missed * angular_velocity * 0.7
                pred_y += 0.5 * ay * frames_missed**2 * 0.7 + vy * frames_missed * angular_velocity * 0.7
                
                # Add reduced recovery attempt compensation
                recovery_factor = 1 + (self.recovery_attempts[track_id] * 0.05)  # Reduced from 0.1 to 0.05
                pred_x *= recovery_factor
                pred_y *= recovery_factor
                
                # Update recovery attempts
                self.recovery_attempts[track_id] += 1
            else:
                # Use standard quadratic motion model with reduced magnitude
                pred_x = last_pos[0] + vx * frames_missed * 0.8 + 0.5 * ax * frames_missed**2 * 0.7
                pred_y = last_pos[1] + vy * frames_missed * 0.8 + 0.5 * ay * frames_missed**2 * 0.7
            
            # Use last known size
            x1, y1, x2, y2 = track['bbox']
            w = x2 - x1
            h = y2 - y1
            
            return [pred_x - w/2, pred_y - h/2, pred_x + w/2, pred_y + h/2]
        
        # Reset recovery attempts if track is recovered or max frames exceeded
        self.recovery_attempts[track_id] = 0
        return None  # Return None to indicate track should be terminated

    def box_iou_batch(self, boxes1, boxes2):
        """
        Calculate IoU between two sets of boxes
        Args:
            boxes1: (N, 4) first set of boxes
            boxes2: (M, 4) second set of boxes
        Returns:
            iou_matrix: (N, M) IoU values for every pair of boxes
        """
        boxes1 = torch.tensor(boxes1)
        boxes2 = torch.tensor(boxes2)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
        
        wh = (rb - lt).clamp(min=0)  # (N,M,2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N,M)
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        return iou.numpy()

    def update(self, detections: List[Tuple[float, float, float, float, float, int]]):
        self.frame_id += 1
        
        if not detections:
            # Update existing tracks with predictions
            for track in self.tracks:
                if self.frame_id - track['last_seen'] > 0:
                    recovered_bbox = self.recover_track(track)
                    if recovered_bbox is not None:  # Only update if recovery was successful
                        track['bbox'] = recovered_bbox
                    track['age'] += 1
                    # Update confidence for missed detections with faster decay
                    if track['id'] in self.track_confidences:
                        self.track_confidences[track['id']] *= self.confidence_decay
            
            # Remove old tracks based on confidence and age
            self.tracks = [track for track in self.tracks 
                          if (self.frame_id - track['last_seen'] <= self.track_buffer and
                              self.track_confidences.get(track['id'], 0) > self.min_confidence and
                              track['hits'] >= self.min_hits and
                              track['age'] <= self.max_age)]
            return self.tracks
        
        dets = np.array(detections)
        
        # Initialize new tracks
        if not self.tracks:
            for det in dets:
                track_id = self.track_id
                self.tracks.append({
                    'id': track_id,
                    'bbox': det[:4],
                    'conf': det[4],
                    'cls': det[5],
                    'age': 1,
                    'hits': 1,
                    'last_seen': self.frame_id
                })
                self.track_confidences[track_id] = det[4]
                self.track_id += 1
            return self.tracks

        # Update bird interactions
        for track_id, track in enumerate(self.tracks):
            self.update_bird_interactions(track_id, self.tracks)

        # Predict next positions for all tracks
        predicted_boxes = []
        for track in self.tracks:
            if self.frame_id - track['last_seen'] > 0:
                predicted_boxes.append(self.recover_track(track))
            else:
                predicted_boxes.append(self.predict_next_position(track))
        predicted_boxes = np.array(predicted_boxes)
        
        # Calculate IoU between predicted positions and new detections
        det_boxes = dets[:, :4]
        iou_matrix = self.box_iou_batch(predicted_boxes, det_boxes)
        
        # Match tracks with detections
        matched_tracks = set()
        matched_dets = set()
        
        for i in range(len(self.tracks)):
            if i in matched_tracks:
                continue
                
            best_j = -1
            best_iou = self.match_thresh
            
            for j in range(len(dets)):
                if j in matched_dets:
                    continue
                    
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            
            if best_j != -1:
                # Update existing track with new detection
                new_bbox = dets[best_j, :4]
                self.calculate_velocity(self.tracks[i], new_bbox)
                self.tracks[i]['bbox'] = new_bbox
                self.tracks[i]['conf'] = dets[best_j, 4]
                self.tracks[i]['cls'] = dets[best_j, 5]
                self.tracks[i]['age'] += 1
                self.tracks[i]['hits'] += 1
                self.tracks[i]['last_seen'] = self.frame_id
                # Update track confidence
                self.update_track_confidence(self.tracks[i]['id'], dets[best_j, 4])
                matched_tracks.add(i)
                matched_dets.add(best_j)
        
        # Add new tracks for unmatched detections
        for j in range(len(dets)):
            if j not in matched_dets:
                track_id = self.track_id
                self.tracks.append({
                    'id': track_id,
                    'bbox': dets[j, :4],
                    'conf': dets[j, 4],
                    'cls': dets[j, 5],
                    'age': 1,
                    'hits': 1,
                    'last_seen': self.frame_id
                })
                self.track_confidences[track_id] = dets[j, 4]
                self.track_id += 1
        
        # Remove old tracks based on confidence
        self.tracks = [track for track in self.tracks 
                      if (self.frame_id - track['last_seen'] <= self.track_buffer and
                          self.track_confidences.get(track['id'], 0) > self.min_confidence and
                          track['hits'] >= self.min_hits)]
        
        return self.tracks

def main():
    # Initialize YOLOv8n model
    model = YOLO('yolov8m.pt')
    
    # Initialize ByteTracker with adjusted parameters
    tracker = BYTETracker()
    
    # Open video file with explicit path
    video_path = "bird.mp4"
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    
    # Create video writer
    output = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )
    
    # FPS calculation variables
    fps_display = 0
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection with optimized parameters
        results = model(frame, conf=0.25, iou=0.25)  # Slightly lowered thresholds for better recall
        
        # Process detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Only include bird detections (class 14 in COCO dataset)
                if cls == 14:  # Bird class
                    detections.append((x1, y1, x2, y2, conf, cls))
        
        # Update tracker
        tracks = tracker.update(detections)
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            conf = track['conf']
            
            # Draw bounding box with solid color
            cv2.rectangle(frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Draw label with track ID and confidence
            label = f'Bird {track_id} {conf:.2f}'
            cv2.putText(frame, label, 
                       (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
            
            # Draw velocity vector if available
            if 'velocity' in track:
                vx, vy = track['velocity']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                end_x = int(center_x + vx * 15)  # Increased scale for better visibility
                end_y = int(center_y + vy * 15)
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps_display = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(frame, f'FPS: {fps_display:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to output video
        output.write(frame)
        
        # Display the frame
        cv2.imshow('Bird Detection with Tracking', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
