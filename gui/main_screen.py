import bisect
import time
from collections import deque

import numpy as np
import cv2

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout

from utils.config import (
    HYSTERESIS_THRESHOLD,
    ST_CAM_OFFSET,
    FOCUS_PLANE_START,
    WINDOW_SIZE,
    PROFILE_CANVAS_BG_M,
    MASK_SAMPLE_RATIO_TRACKED,
    MASK_SAMPLE_RATIO_UNTRACKED,
)
from hardware.camera import RealSenseCamera
from hardware.motor_controller import MotorController
from vision.object_tracker import DetectionPipeline, SortTracker
from vision.depth_processor import correct_distance, get_lighting_lut


class MainScreen(FloatLayout):
    def __init__(self, lichtbedingung=None, **kwargs):
        super(MainScreen, self).__init__(**kwargs)

        # UI elements
        self.video_image = Image(size_hint=(0.8, 0.8), pos_hint={'center_x': 0.4, 'center_y': 0.5})
        self.add_widget(self.video_image)

        self.profile_image = Image(size_hint=(0.18, 0.8), pos_hint={'right': 1.0, 'center_y': 0.5})
        self.add_widget(self.profile_image)

        #  BoxLayout oben andocken
        self.status_bar = BoxLayout(size_hint=(None, None), size=(40, 30), pos=(20, 10))
        self.bind(size=self._update_status_bar_pos)
        self.fps_label = Label(text="FPS: 0")
        self.status_bar.add_widget(self.fps_label)
        self.add_widget(self.status_bar)

        self.instruction_bar = BoxLayout(size_hint=(None, None), size=(40, 30), pos=(200, 10))
        self.bind(size=self._update_instruction_bar_pos)
        self.intruction_label = Label(text="Antippen, um das Fokussobjekt auszuwählen")
        self.instruction_bar.add_widget(self.intruction_label)
        self.add_widget(self.instruction_bar)

        self.focus_slider = Slider(min=0, max=10, value=0, size_hint=(0.3, 0.05), pos_hint={'x': 0.20, 'y': 0.01})
        self.focus_slider.bind(value=self.on_slider_value_change)
        self.focus_label = Label(
            text=f'Fokusszeit: {self.focus_slider.value:.2f} s',
            size_hint=(0.6, 0.05),
            pos_hint={'x': 0.05, 'y': 0.05},
            color=[1, 1, 1, 1]
        )
        self.add_widget(self.focus_slider)
        self.add_widget(self.focus_label)

        self.reset_button = Button(text="Reset Tracking", size_hint=(None, None), size=(140, 40), pos_hint={'x': 0.01, 'y': 0.02})
        self.reset_button.bind(on_press=self.reset_tracking)
        self.add_widget(self.reset_button)

        # State
        #  Framegröße und ROI erst nach erstem Kameroframe setzen
        self.frame_width = 1280
        self.frame_height = 720
        self.roi_start = None
        self.roi_end = None
        self.dragging = False
        self.selected_corner = None
        self.corner_size = 20

        self.selected_id = None
        self.person_tracks = []
        self.focus_locked_once = False

        self.of_point_selected = False
        self.of_point = ()
        self.of_old_points = None
        self.of_old_gray = None

        self.lighting_condition = get_lighting_lut(lichtbedingung)
        self.last_target_distance = None
        self.white_bar_pos = 0
        self.focus_distance = 0.0
        self.prev_time = time.time()
        self.fps_history = deque(maxlen=10)

        # Components
        self.detector = DetectionPipeline()
        self.tracker = SortTracker()
        self.camera = RealSenseCamera()
        #  ersten Frame holen und ROI aus Kameroframe ableiten
        first_color, _first_depth = self.camera.get_aligned_frames()
        if first_color is not None:
            self.frame_height, self.frame_width = first_color.shape[:2]
            self.roi_start = [int(self.frame_width * 0.37), int(self.frame_height * 0.37)]
            self.roi_end = [int(self.frame_width * 0.61), int(self.frame_height * 0.65)]

        self.motor = MotorController(initial_focus_time=self.focus_slider.value)

        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def on_slider_value_change(self, instance, value):
        self.focus_label.text = f'Fokusszeit: {value:.2f} s'

    #  Leisten oben positionieren
    def _update_status_bar_pos(self, *args):
        self.status_bar.pos = (20, self.height - self.status_bar.height - 10)

    def _update_instruction_bar_pos(self, *args):
        self.instruction_bar.pos = (400, self.height - self.instruction_bar.height - 10)

    def get_image_coordinates(self, touch):
        if not self.video_image.collide_point(*touch.pos):
            return None, None
        tex = self.video_image.texture
        if tex is None:
            return None, None
        img_w, img_h = tex.size
        widget_w, widget_h = self.video_image.size
        scale_x = img_w / widget_w
        scale_y = img_h / widget_h
        x = (touch.x - self.video_image.x) * scale_x
        y = (self.video_image.height - (touch.y - self.video_image.y)) * scale_y
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        return int(x), int(y)

    def on_touch_down(self, touch):
        if self.roi_start is None or self.roi_end is None:
            return super(MainScreen, self).on_touch_down(touch)
        x, y = self.get_image_coordinates(touch)
        if x is None or y is None:
            return super(MainScreen, self).on_touch_down(touch)

        for idx, (cx, cy) in enumerate([
            self.roi_start,
            [self.roi_end[0], self.roi_start[1]],
            [self.roi_start[0], self.roi_end[1]],
            self.roi_end
        ]):
            if abs(x - cx) < self.corner_size and abs(y - cy) < self.corner_size:
                self.dragging = True
                self.selected_corner = idx
                return True

        matching_tracks = []
        for track in self.person_tracks:
            x1, y1, x2, y2, track_id = track
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                matching_tracks.append((track, area))
        if matching_tracks:
            matching_tracks.sort(key=lambda x: x[1])
            selected_track = matching_tracks[0][0]
            self.selected_id = int(selected_track[4])
            self.of_point_selected = False
            self.focus_locked_once = False
            return True

        roi_x1, roi_y1 = self.roi_start
        roi_x2, roi_y2 = self.roi_end
        if roi_x1 <= x < roi_x2 and roi_y1 <= y < roi_y2:
            self.of_point = (int(x) - roi_x1, int(y) - roi_y1)
            self.of_point_selected = True
            self.of_old_points = np.array([[self.of_point]], dtype=np.float32)
            self.selected_id = None
            self.focus_locked_once = False
        else:
            self.of_point_selected = False
        return True

    def on_touch_move(self, touch):
        if not self.dragging or self.selected_corner is None:
            return super(MainScreen, self).on_touch_move(touch)
        x, y = self.get_image_coordinates(touch)
        if x is None or y is None:
            return True
        if self.selected_corner == 0:
            self.roi_start = [x, y]
        elif self.selected_corner == 1:
            self.roi_end[0] = x
            self.roi_start[1] = y
        elif self.selected_corner == 2:
            self.roi_start[0] = x
            self.roi_end[1] = y
        elif self.selected_corner == 3:
            self.roi_end = [x, y]
        self.roi_start[0] = max(0, min(self.roi_end[0] - 50, self.roi_start[0]))
        self.roi_start[1] = max(0, min(self.roi_end[1] - 50, self.roi_start[1]))
        self.roi_end[0] = min(self.frame_width, max(self.roi_start[0] + 50, self.roi_end[0]))
        self.roi_end[1] = min(self.frame_height, max(self.roi_start[1] + 50, self.roi_end[1]))
        return True

    def on_touch_up(self, touch):
        was_dragging = self.dragging
        self.dragging = False
        self.selected_corner = None
        result = super(MainScreen, self).on_touch_up(touch)
        if was_dragging:
            self.of_point_selected = False
            self.of_point = ()
            self.of_old_points = None
            self.of_old_gray = None
        return result

    @staticmethod
    def _get_non_overlapping_crop(frame, tracking_bbox, other_bboxes):
        try:
            x1, y1, x2, y2 = tracking_bbox
            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(frame.shape[1], x2)
            y2_crop = min(frame.shape[0], y2)
            if x2_crop <= x1_crop or y2_crop <= y1_crop:
                return None
            crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            mask = np.ones(crop.shape[:2], dtype=np.uint8)
            for ox1, oy1, ox2, oy2, _ in other_bboxes:
                ox1_rel = max(0, ox1 - x1_crop)
                oy1_rel = max(0, oy1 - y1_crop)
                ox2_rel = min(crop.shape[1], ox2 - x1_crop)
                oy2_rel = min(crop.shape[0], oy2 - y1_crop)
                if ox2_rel > ox1_rel and oy2_rel > oy1_rel:
                    mask[oy1_rel:oy2_rel, ox1_rel:ox2_rel] = 0
            black_ratio = np.mean(mask == 0)
            if black_ratio > 0.5:
                return crop, x1_crop, y1_crop, x2_crop, y2_crop
            crop[mask == 0] = [0, 0, 0]
            return crop, x1_crop, y1_crop, x2_crop, y2_crop
        except Exception:
            return None

    def update(self, dt):
        try:
            color_frame, depth_image = self.camera.get_aligned_frames()
            if color_frame is None or depth_image is None:
                return
            frame = color_frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            roi_x1, roi_y1 = self.roi_start
            roi_x2, roi_y2 = self.roi_end
            roi_x1 = max(0, min(self.frame_width - 10, roi_x1))
            roi_y1 = max(0, min(self.frame_height - 10, roi_y1))
            roi_x2 = max(roi_x1 + 10, min(self.frame_width, roi_x2))
            roi_y2 = max(roi_y1 + 10, min(self.frame_height, roi_y2))
            self.roi_start = [roi_x1, roi_y1]
            self.roi_end = [roi_x2, roi_y2]

            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            if roi_frame.size == 0:
                return

            person_bboxes_roi = self.detector.detect_person_bboxes(roi_frame)
            person_bboxes = []
            for (x1, y1, x2, y2, score) in person_bboxes_roi:
                x1_full = int(x1 + roi_x1)
                y1_full = int(y1 + roi_y1)
                x2_full = int(x2 + roi_x1)
                y2_full = int(y2 + roi_y1)
                #  Nx4 ohne Score an SORT
                person_bboxes.append((x1_full, y1_full, x2_full, y2_full))
            dets_for_sort = np.array(person_bboxes) if person_bboxes else np.empty((0, 5))
            tracks = self.tracker.update(dets_for_sort)
            self.person_tracks = tracks

            # Depth profile canvas
            canvas_height = 900
            scaled_width = int(self.frame_width * 0.18)
            profile_canvas = np.zeros((canvas_height, scaled_width, 3), dtype=np.uint8)
            profile_canvas[:] = [30, 30, 30]
            background_depth = PROFILE_CANVAS_BG_M
            y_scale = (canvas_height - 50) / background_depth
            r = 2

            # Draw tracks and compute focus distance
            for track in tracks:
                x1, y1, x2, y2, track_id = track.astype(int)
                color = (0, 255, 0) if self.selected_id == int(track_id) else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if self.selected_id is not None and self.selected_id == int(track_id):
                    other_bboxes = []
                    for t2 in tracks:
                        x1o, y1o, x2o, y2o, id2 = t2.astype(int)
                        if int(id2) != int(track_id):
                            other_bboxes.append((x1o, y1o, x2o, y2o, id2))

                    crop_info = self._get_non_overlapping_crop(frame, (x1, y1, x2, y2), other_bboxes)
                    if not crop_info:
                        continue
                    person_crop, x1c, y1c, x2c, y2c = crop_info

                    # Face detection in the crop
                    face_results = self.detector.detect_faces(person_crop)
                    found_face = False
                    face_box = None
                    face_text = ""
                    corrected_face_distance = None
                    uncorrected_face_distance = None

                    for (fx1c, fy1c, fx2c, fy2c, fscore) in face_results:
                        found_face = True
                        fx1 = int(fx1c + x1c)
                        fy1 = int(fy1c + y1c)
                        fx2 = int(fx2c + x1c)
                        fy2 = int(fy2c + y1c)
                        face_area = depth_image[fy1:fy2, fx1:fx2]
                        valid = face_area[face_area > 0]
                        if valid.size > 0:
                            uncorrected_face_distance = float(np.mean(valid)) / 1000.0
                        else:
                            uncorrected_face_distance = 0.0
                        corrected_face_distance = correct_distance(uncorrected_face_distance, self.lighting_condition)
                        self.focus_distance = corrected_face_distance + ST_CAM_OFFSET
                        face_box = (fx1, fy1, fx2, fy2)
                        face_text = f"corr: {corrected_face_distance:.2f} uncorr:{uncorrected_face_distance:.2f}"
                        break  # first face

                    # Person segmentation mask for depth sampling
                    mask = self.detector.segment_person(person_crop)
                    if mask is not None:
                        if mask.shape[:2] != person_crop.shape[:2]:
                            mask = cv2.resize(mask, (person_crop.shape[1], person_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask_bin = (mask > 0.5).astype(np.uint8)
                        #  rotes Overlay auf getrackter Person
                        color_mask = np.zeros_like(person_crop, dtype=np.uint8)
                        color_mask[..., 2] = 255
                        person_crop[mask_bin.astype(bool)] = cv2.addWeighted(
                            person_crop[mask_bin.astype(bool)], 0.6, color_mask[mask_bin.astype(bool)], 0.4, 0
                        )
                        frame[y1:y2, x1:x2] = person_crop
                        # ... Tiefen-Sampling wie gehabt ...

                    if face_box is not None:
                        fx1, fy1, fx2, fy2 = face_box
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                        cv2.putText(frame, face_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Untracked persons contribute gray samples
                if self.selected_id != int(track_id):
                    person_crop_nt = frame[y1:y2, x1:x2].copy()
                    mask_nt = self.detector.segment_person(person_crop_nt)
                    if mask_nt is not None:
                        if mask_nt.shape[:2] != person_crop_nt.shape[:2]:
                            mask_nt = cv2.resize(mask_nt, (person_crop_nt.shape[1], person_crop_nt.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask_bin = (mask_nt > 0.5).astype(np.uint8)
                        y_coords, x_coords = np.where(mask_bin)
                        num_points = len(y_coords)
                        if num_points > 0:
                            sample_size = max(1, int(num_points * MASK_SAMPLE_RATIO_UNTRACKED))
                            indices = np.random.choice(num_points, sample_size, replace=False)
                            x_sample = x_coords[indices] + x1
                            y_sample = y_coords[indices] + y1
                            depths_nt = depth_image[y_sample, x_sample] / 1000.0
                            mask_distance_nt = depths_nt[depths_nt > 0]
                            uncorrected_nt = float(np.mean(mask_distance_nt)) if mask_distance_nt.size > 0 else 0.0
                            corrected_nt = correct_distance(uncorrected_nt, self.lighting_condition)
                            diff_nt = uncorrected_nt - corrected_nt
                            x_coords_reduced = (x_sample * (scaled_width / self.frame_width)).astype(int)
                            y_positions = 50 + ((background_depth - (depths_nt - diff_nt + ST_CAM_OFFSET)) * y_scale).astype(int)
                            x_coords_reduced = np.clip(x_coords_reduced, 0, scaled_width - 1)
                            y_positions = np.clip(y_positions, 0, canvas_height - 1)
                            for xx, yy in zip(x_coords_reduced, y_positions):
                                x1b, x2b = max(0, xx - r), min(scaled_width, xx + r + 1)
                                y1b, y2b = max(0, yy - r), min(canvas_height, yy + r + 1)
                                profile_canvas[y1b:y2b, x1b:x2b] = [128, 128, 128]

                label = f"Fokussperson {track_id}: {self.focus_distance:.2f}m" if self.selected_id == int(track_id) else f"Person {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Optical flow within ROI to set focus point if selected
            if self.of_point_selected and self.of_old_points is not None:
                roi_gray = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                if self.of_old_gray is None:
                    self.of_old_gray = roi_gray.copy()
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.of_old_gray, roi_gray, self.of_old_points, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                good_new = new_points[status.flatten() == 1] if new_points is not None else []
                if good_new is not None and good_new.size > 0:
                    good_new = good_new.reshape(-1, 2)
                    mean_x, mean_y = np.mean(good_new, axis=0)

                    all_depths = []
                    for px, py in good_new:
                        x_track = int(px) + roi_x1
                        y_track = int(py) + roi_y1
                        x1w = max(0, x_track - 5)
                        x2w = min(depth_image.shape[1], x_track + 5)
                        y1w = max(0, y_track - 5)
                        y2w = min(depth_image.shape[0], y_track + 5)
                        window = depth_image[y1w:y2w, x1w:x2w] / 1000.0
                        all_depths.extend(window.flatten())
                        cv2.rectangle(frame, (x1w, y1w), (x2w, y2w), (0, 255, 0), 2)

                    depths = [d for d in all_depths if d > 0]
                    uncorrected_of = float(np.median(depths)) if depths else 0.0
                    corrected_of = correct_distance(uncorrected_of, self.lighting_condition)
                    if corrected_of > 0:
                        self.focus_distance = corrected_of + ST_CAM_OFFSET

                    cv2.putText(
                        frame,
                        f"{self.focus_slider.value:.2f} corr: {corrected_of:.2f} uncorr: {uncorrected_of:.2f} fD: {self.focus_distance:.2f}",
                        (int(mean_x) + roi_x1, int(mean_y) + roi_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    self.of_old_points = good_new.reshape(-1, 1, 2)
                else:
                    self.of_point_selected = False
                    self.of_old_points = None
                self.of_old_gray = roi_gray.copy()

            # Motor control
            target_steps = self.motor.distance_to_steps(self.focus_distance)
            steps_diff = abs(target_steps - self.motor.current_steps)
            if steps_diff <= 1 and not self.focus_locked_once:
                self.focus_locked_once = True

            if (self.last_target_distance is None or
                    abs(self.focus_distance - self.last_target_distance) > HYSTERESIS_THRESHOLD):
                if self.focus_locked_once:
                    self.motor.move_to(target_steps, focus_time=0.001)
                else:
                    self.motor.move_to(target_steps, focus_time=self.focus_slider.value)
                self.last_target_distance = self.focus_distance

            # Depth profile decorations
            not_valid = 50 + int((background_depth - FOCUS_PLANE_START) * y_scale)
            cv2.line(profile_canvas, (0, not_valid), (scaled_width, not_valid), (0, 0, 255), 2)
            cv2.line(profile_canvas, (0, profile_canvas.shape[0] - 2), (scaled_width, profile_canvas.shape[0] - 2), (0, 0, 255), 2)
            cv2.line(profile_canvas, (0, profile_canvas.shape[0]), (scaled_width, not_valid), (0, 0, 255), 2)
            cv2.line(profile_canvas, (0, not_valid), (scaled_width, profile_canvas.shape[0]), (0, 0, 255), 2)

            self.white_bar_pos = 50 + int((background_depth - self.focus_distance) * y_scale)
            cv2.line(profile_canvas, (0, self.white_bar_pos), (scaled_width, self.white_bar_pos), (255, 255, 255), 5)

            focus_plane_pos = self.motor.focus_plane_pos(self.motor.current_steps) if self.motor.current_steps != 0 else FOCUS_PLANE_START
            focus_plane_y = 50 + int((background_depth - focus_plane_pos) * y_scale)
            cv2.line(profile_canvas, (0, focus_plane_y), (scaled_width, focus_plane_y), (0, 255, 0), 4)

            for y in range(0, int(background_depth) + 1):
                y_pos = 50 + int((background_depth - y) * y_scale)
                if 0 <= y_pos < canvas_height:
                    cv2.line(profile_canvas, (0, y_pos), (20, y_pos), (255, 255, 255), 1)
                    cv2.putText(profile_canvas, f"{y}m", (25, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Push textures
            buf_profile = cv2.flip(profile_canvas, 0).tobytes()
            texture_profile = Texture.create(size=(profile_canvas.shape[1], profile_canvas.shape[0]), colorfmt='bgr')
            texture_profile.blit_buffer(buf_profile, colorfmt='bgr', bufferfmt='ubyte')
            self.profile_image.texture = texture_profile

            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            corner_color = (0, 255, 255)
            corner_size = 15
            for (cx, cy) in [(roi_x1, roi_y1), (roi_x2, roi_y1), (roi_x1, roi_y2), (roi_x2, roi_y2)]:
                cv2.rectangle(frame, (int(cx) - corner_size, int(cy) - corner_size), (int(cx) + corner_size, int(cy) + corner_size), corner_color, -1)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.video_image.texture = texture

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
            self.prev_time = curr_time
            self.fps_history.append(fps)
            smoothed_fps = float(np.median(self.fps_history))
            self.fps_label.text = f"FPS: {int(smoothed_fps)}"

        except Exception as e:
            print(f"Error in update: {e}")

    def reset_tracking(self, instance):
        self.selected_id = None
        self.of_point_selected = False
        self.of_point = ()
        self.of_old_points = None
        self.of_old_gray = None
        self.white_bar_pos = 0
        self.focus_locked_once = False

    def cleanup(self):
        try:
            if hasattr(self, 'camera') and self.camera:
                self.camera.stop()
            if hasattr(self, 'motor') and self.motor:
                self.motor.stop()
        except Exception as e:
            print(f"Fehler beim Cleanup: {e}")