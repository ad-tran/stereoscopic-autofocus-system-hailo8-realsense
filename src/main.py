import cv2
import time
import numpy as np
import pyrealsense2 as rs
from collections import deque
import bisect
import ctypes

# Für Objekterkennung und -Verfolgung
import degirum as dg
import sys
sys.path.append('/home/amacus/hailo_examples/sort')
from sort import Sort


# Für Kivy GUI
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty, ListProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.slider import Slider
from kivy.uix.stencilview import StencilView
from kivy.graphics import Color, Line
from kivy.properties import NumericProperty
from kivy.graphics.transformation import Matrix
from kivy.uix.widget import Widget
import os

# Für Motorsteuerung
import multiprocessing as mp
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
from multiprocessing import Value

# GUI Fenstergröße, passt zum Display
Window.size = (1280, 700)
Window.maximize()



class CalibrationStep(GridLayout):
    text = StringProperty("")
    checked = BooleanProperty(False)
    dropdown_options = ListProperty([])
    dropdown_value = StringProperty("Auswählen")
    def __init__(self, step_num, text, has_dropdown=False, options=None, font_size='30sp', **kwargs):
        super(CalibrationStep, self).__init__(**kwargs)
        self.has_dropdown = has_dropdown # damit Auswählen Dropdown Menu hat
        self.cols = 3 if has_dropdown else 2
        self.rows = 1
        self.size_hint_y = None
        self.height = 40
        
    
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect) 
        self.selected_option_btn = None # Merkt sich den aktuell markierten Dropdown-Button
        
        # Checkbox und Label
        self.checkbox = CheckBox(
            active=self.checked, 
            size_hint_x=None,
            width=250,
            color=[0,0,0,1],
            )
        
        self.checkbox.bind(active=self.on_checkbox_active)
        self.label = Label(
            text=f"{step_num}. {text}",
            halign='left',
            valign='middle',
            color=[0, 0, 0, 1],
            font_size=font_size,
            size_hint_x=0.8 if has_dropdown else 1
        )
        self.label.bind(size=self.label.setter('text_size'))
        self.add_widget(self.checkbox)
        self.add_widget(self.label)

        if has_dropdown:
            self.dropdown = DropDown()
            for option in options:
                btn = Button(
                    text=option,
                    color=[0, 0, 0, 1],
                    background_normal='',
                    background_color=[1, 1, 1, 1],
                    size_hint_y=None,
                    height=44,
                    size_hint_x=None,
                    width=200
                )
                btn.bind(on_release=lambda btn: self.select_option(btn.text))
                self.dropdown.add_widget(btn)
                

            # Dropdown-Button
            self.dropdown_btn = Button(
                text=self.dropdown_value,
                color=[0, 0, 0, 1],
                background_normal='',
                background_color=[1, 1, 1, 1],
                size_hint_x=None,
                width=200,
                height=50,
                font_size='19sp'
            )


          # Rahmen um den Auswählen-Button
            with self.dropdown_btn.canvas.after:
                self.border_color = Color(0, 0, 0, 1)  # Anfangs sichtbar (alpha=1)
                self.border = Line(rectangle=(0, 0, self.dropdown_btn.width, self.dropdown_btn.height), width=2)
            def update_border(instance, value):
                self.border.rectangle = (920, 0, instance.width, instance.height)
            self.dropdown_btn.bind(size=update_border, pos=update_border)

            def hide_border(instance):
                self.border_color.a = 0  # Rahmen ausblenden

            self.dropdown_btn.bind(on_release=hide_border)
            self.dropdown_btn.bind(on_release=self.dropdown.open)
            self.dropdown.bind(on_select=lambda instance, x: setattr(self, 'dropdown_value', x))
            self.add_widget(self.dropdown_btn)
            
            
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def on_checkbox_active(self, instance, value):
        if not getattr(self, 'has_dropdown', False):
            self.checked = value
        else:
            self.checkbox.active = self.checked

    def select_option(self, value):
        self.dropdown.select(value)
        self.dropdown_btn.text = value
        self.dropdown_value = value
        self.checked = value != "Auswählen"
        self.checkbox.active = self.checked


class CalibrationScreen(BoxLayout):
    def __init__(self, main_app, **kwargs):
        super(CalibrationScreen, self).__init__(**kwargs)
        self.main_app = main_app
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        self.title = Label(
            text="Führen Sie folgende Kalibrierungsschritte aus und kreuzen Sie an, \num zu starten:",
            font_size='40sp',
            color=[0, 0, 0, 1]
        )
    
        # Logo oben rechts
        self.logo = Image(
            source='/home/amacus/hailo_examples/LogoName.png',  
            size_hint=(None, None),
            size=(150, 150),  
            pos_hint={'right': 0.95, 'top': 0.95}
        )
        
        logo_layout = FloatLayout(size_hint=(1, None), height=50)
        logo_layout.add_widget(self.logo)
        self.add_widget(logo_layout)

        self.add_widget(self.title)
        self.scroll = ScrollView()
        self.steps_layout = BoxLayout(
            orientation='vertical',
            spacing=10,
            size_hint_y=None
        )
        self.steps_layout.bind(minimum_height=self.steps_layout.setter('height'))
        with self.steps_layout.canvas.before:
            Color(1, 1, 1, 1)
            self.steps_rect = Rectangle(size=self.steps_layout.size, pos=self.steps_layout.pos)
        self.steps_layout.bind(size=self._update_steps_rect, pos=self._update_steps_rect)
        self.steps = [
            CalibrationStep(1, "Stellen Sie den Fokusring auf den minimalen Fokusabstand ein."),
            CalibrationStep(2, "Bringen Sie den Schrittmotor an den Fokusring und befestigen ihn."),
            CalibrationStep(3, "Wählen Sie die Lichtbedingung Ihrer Szene aus:", True, ["Drinnen - Gutes Licht", "Drinnen - Schlechtes Licht", "Draußen - Gutes Licht", "Draußen - Schlechtes Licht"],
                            size_hint_x=0.89,   # Label schmaler
                            width=250       # Dropdown-Button schmaler
                        )
        ]
        for step in self.steps:
            self.steps_layout.add_widget(step)
        self.scroll.add_widget(self.steps_layout)
        self.add_widget(self.scroll)
        self.start_btn = Button(
            text="Start",
            size_hint_y=None,
            height=50,
            pos_hint={'center': 0.5},
            color=[0, 0, 0, 1],
            background_normal='',
            background_color=[0.9, 0.9, 0.9, 1],
            font_size='30sp',
        )
        self.start_btn.bind(on_press=self.check_calibration)
        self.add_widget(self.start_btn)
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    def _update_steps_rect(self, instance, value):
        self.steps_rect.pos = instance.pos
        self.steps_rect.size = instance.size
    def check_calibration(self, instance):
        all_checked = all(step.checked for step in self.steps)
        if all_checked:
            self.main_app.start_main_program()
        else:
            content = Label(
                text='Bitte alle Kalibrierungsschritte durchführen!',
                color=[1,1,1,1],
                font_size='30sp'
            )
            popup = Popup(
                title='Fehler 006',
                content=content,
                size_hint=(0.5, 0.2))
            with popup.canvas.before:
                Color(1, 1, 1, 1)
                popup.rect = Rectangle(size=popup.size, pos=popup.pos)
            popup.bind(size=self._update_popup_rect, pos=self._update_popup_rect)
            popup.open()
    def _update_popup_rect(self, instance, value):
        instance.rect.pos = instance.pos
        instance.rect.size = instance.size


class MaskedLogo(StencilView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = (800, 800)
        from kivy.core.window import Window
        self.pos = (Window.width / 2 - self.size[0] / 2, Window.height / 2 - self.size[1] / 2)
        image_path = '/home/amacus/hailo_examples/LogoName.png'
        self.logo = Image(source=image_path, size_hint=(None, None), size=self.size, pos=self.pos)
        self.add_widget(self.logo)
        self.progress_value = 0
        with self.canvas:
            self.mask_color = Color(1, 1, 1, 1)
            self.mask_rect = Rectangle(pos=self.pos, size=self.size)
        from kivy.clock import Clock
        Clock.schedule_interval(self.update_progress, 0.03)
    def update_progress(self, dt):
        if self.progress_value < self.size[0]:
            self.progress_value += 30
            self.mask_rect.pos = (self.pos[0] + self.progress_value, self.pos[1])
        else:
            from kivy.clock import Clock
            Clock.unschedule(self.update_progress)
            # Nach dem Laden: LoadingScreen ausblenden
            if hasattr(self, 'parent') and hasattr(self.parent, 'on_loading_finished'):
                self.parent.on_loading_finished()


class LoadingScreen(FloatLayout):
    def __init__(self, on_finished_callback=None, **kwargs):
        super().__init__(**kwargs)
        from kivy.core.window import Window
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.bg = Rectangle(size=Window.size)
        self.mask = MaskedLogo()
        self.add_widget(self.mask)
        self.on_finished_callback = on_finished_callback

    def on_loading_finished(self):
        if self.on_finished_callback:
            # 5 Sekunden warten, dann Callback ausführen
            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: self.on_finished_callback(), 2)

class MainScreen(FloatLayout):
    
    Hysteresis_Threshold = 0.02
    
     
    motor_lut = [
    (0, 0.6), (10, 0.61), (20, 0.62), (30, 0.63), (40, 0.65), (50, 0.67), (60, 0.69), (70, 0.71),
    (75, 0.72), (80, 0.73), (85, 0.74), (90, 0.75), (95, 0.76), (100, 0.77), (105, 0.78),
    (110, 0.79), (115, 0.81), (120, 0.82), (125, 0.83), (130, 0.85), (135, 0.86), (140, 0.87),
    (145, 0.89), (150, 0.91), (155, 0.93), (160, 0.94), (165, 0.97), (170, 0.98), (175, 0.99),
    (180, 1.01), (185, 1.03), (190, 1.04), (195, 1.07), (200, 1.09), (205, 1.11), (210, 1.14),
    (215, 1.16), (220, 1.21), (225, 1.24), (230, 1.28), (235, 1.31), (240, 1.34), (245, 1.37),
    (250, 1.42), (255, 1.45), (260, 1.54), (265, 1.58), (270, 1.63), (275, 1.67), (280, 1.71),
    (285, 1.79), (290, 1.83), (295, 1.92), (300, 1.98), (305, 2.06), (310, 2.15), (312, 2.18),
    (314, 2.23), (316, 2.25), (318, 2.28), (320, 2.34), (322, 2.4), (324, 2.43), (326, 2.52),
    (328, 2.59), (330, 2.64), (332, 2.69), (334, 2.73), (336, 2.8), (338, 2.84), (340, 2.88),
    (342, 3), (344, 3.13), (346, 3.21), (348, 3.28), (350, 3.37), (352, 3.42), (354, 3.52),
    (356, 3.65), (358, 3.81), (360, 3.92), (362, 3.98), (364, 4.07), (366, 4.12), (368, 4.22),
    (370, 4.43), (372, 4.89), (374, 5.25), (376, 5.47), (378, 5.72), (380, 5.92), (382, 5.99),
    (384, 6.05), (386, 6.56), (388, 6.96), (390, 7.51), (392, 7.95), (394, 8.26), (396, 8.64),
    (398, 9.06), (400, 9.79), (402, 10.8), (404, 11.8)
    ]

    @staticmethod
    def distance_to_steps(distance_m):
        steps = [x[0] for x in MainScreen.motor_lut]
        distances = [x[1] for x in MainScreen.motor_lut]
        pos = bisect.bisect_left(distances, distance_m)
        if pos == 0:
            return steps[0]
        elif pos == len(distances):
            return steps[-1]
        else:
            prev_step, prev_dist = MainScreen.motor_lut[pos-1]
            next_step, next_dist = MainScreen.motor_lut[pos]
            alpha = (distance_m - prev_dist) / (next_dist - prev_dist)
            return int(prev_step + alpha * (next_step - prev_step))

     
    @staticmethod
    def motor_worker(queue, stop_event, current_motor_steps, time):
        from adafruit_motorkit import MotorKit
        from adafruit_motor import stepper
        import time

        kit = MotorKit()
        max_speed_delay = 0.001
        homing_speed_delay = 0.01  # Langsamere Geschwindigkeit für Homing

        try:
            while not stop_event.is_set():
                try:
                    target_steps = None
                    focustime = time
                    while not queue.empty():
                        item = queue.get_nowait()
                        if isinstance(item, tuple):
                            target_steps, focustime = item
                        else:
                            target_steps = item
                            focustime = time

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
                        if stop_event.is_set():  # Abbruch wenn Homing aktiviert
                            break
                        if not queue.empty():
                            break

                        kit.stepper1.onestep(direction=direction, style=stepper.INTERLEAVE)
                        with current_motor_steps.get_lock():
                            if direction == stepper.FORWARD:
                                current_motor_steps.value += 1
                            else:
                                current_motor_steps.value -= 1
                        time.sleep(delay)

                except Exception as e:
                    print(f"Motor error: {e}")
                    continue

        finally:
            # Automatisches Homing beim Beenden
            
            with current_motor_steps.get_lock():
                home_steps = -current_motor_steps.value
                current_steps = current_motor_steps.value
            
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
            
            kit.stepper1.release()
            

    
    @staticmethod
    def focus_plane_pos(curr_step):
        """
        Berechnet die Position der Fokusebene basierend auf der aktuellen Schrittzahl.
        """
        steps = [x[0] for x in MainScreen.motor_lut]
        distances = [x[1] for x in MainScreen.motor_lut]
        
        pos = bisect.bisect_left(steps, curr_step)
        
        if pos == 0:
            return distances[0]
        elif pos == len(steps):
            return distances[-1]
        else:
            prev_step, prev_dist = MainScreen.motor_lut[pos-1]
            next_step, next_dist = MainScreen.motor_lut[pos]
            
            alpha = (curr_step - prev_step) / (next_step - prev_step)
            return prev_dist + alpha * (next_dist - prev_dist)
        
    ,  
    # --- LUTs und Korrekturfunktion ---
    inside_good_lighting = [
        0.5, 0.6, 0.7, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.62, 1.73, 1.83, 1.92,
        2.03, 2.13, 2.24, 2.34, 2.44, 2.55, 2.65, 2.76, 2.86, 2.95, 3.08, 3.19, 3.29, 3.38, 3.5,
        3.59, 3.69, 3.82, 3.91, 4.02, 4.13, 4.24, 4.35, 4.45, 4.54, 4.66, 4.78, 4.87, 4.96, 5.07,
        5.23, 5.34, 5.43, 5.55, 5.63, 5.71, 5.81, 5.93, 6.11, 6.21, 6.28, 6.37, 6.47, 6.58, 6.69,
        6.8, 6.9, 7.05, 7.11, 7.26, 7.32, 7.46, 7.55, 7.65, 7.79, 7.91, 8.13, 8.2, 8.34, 8.45,
        8.59, 8.71, 8.74, 8.8, 8.88, 9.0, 9.04, 9.11, 9.22, 9.37, 9.47, 9.68, 9.69, 9.77, 10.1,
        10.21, 10.25, 10.34, 10.89, 10.96, 11.11
    ]
    inside_bad_lighting = [
        0.5, 0.6, 0.7, 0.81, 0.9, 1.01, 1.11, 1.2, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.92,
        2.01, 2.12, 2.23, 2.33, 2.43, 2.53, 2.65, 2.74, 2.85, 2.92, 3.05, 3.16, 3.27, 3.33, 3.47,
        3.58, 3.68, 3.79, 3.88, 3.98, 4.06, 4.2, 4.31, 4.4, 4.52, 4.63, 4.77, 4.85, 4.94, 5.03,
        5.19, 5.29, 5.4, 5.5, 5.59, 5.7, 5.78, 5.9, 6.1, 6.2, 6.31, 6.39, 6.48, 6.59, 6.7, 6.82,
        6.92, 7.08, 7.23, 7.37, 7.45, 7.53, 7.59, 7.65, 7.83, 7.96, 8.07, 8.17, 8.38, 8.57, 8.59,
        8.61, 8.71, 8.76, 8.85, 8.9, 9.06, 9.21, 9.28, 9.5, 9.59, 9.69, 9.73, 9.88, 9.98, 10.27,
        10.33, 10.55, 10.63, 10.9, 11.29
    ]
    outside_good_lighting = [
        0.5, 0.6, 0.7, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.42, 1.52, 1.62, 1.72, 1.83, 1.93,
        2.03, 2.14, 2.25, 2.34, 2.45, 2.56, 2.64, 2.76, 2.88, 2.99, 3.05, 3.18, 3.25, 3.35, 3.45,
        3.56, 3.66, 3.75, 3.87, 4.01, 4.12, 4.2, 4.32, 4.41, 4.54, 4.63, 4.72, 4.8, 4.92, 4.99,
        5.17, 5.27, 5.38, 5.47, 5.54, 5.65, 5.76, 5.89, 6.01, 6.2, 6.27, 6.35, 6.48, 6.55, 6.73,
        6.86, 6.99, 7.05, 7.21, 7.31, 7.42, 7.5, 7.66, 7.8, 7.89, 8.05, 8.14, 8.34, 8.43, 8.55,
        8.69, 8.57, 8.8, 8.92, 9.0, 9.03, 9.09, 9.23, 9.29, 9.5, 9.51, 9.56, 9.87, 10.01, 10.19,
        10.24, 10.48, 10.6, 10.71, 10.71, 10.94
    ]
    outside_bad_lighting = [
        0.5, 0.6, 0.7, 0.81, 0.9, 1.01, 1.11, 1.2, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.92,
        2.01, 2.12, 2.23, 2.33, 2.43, 2.53, 2.65, 2.74, 2.85, 2.92, 3.05, 3.16, 3.27, 3.33, 3.47,
        3.58, 3.68, 3.79, 3.88, 3.98, 4.06, 4.2, 4.31, 4.4, 4.52, 4.63, 4.77, 4.85, 4.94, 5.03,
        5.19, 5.29, 5.4, 5.5, 5.59, 5.7, 5.78, 5.9, 6.1, 6.2, 6.31, 6.39, 6.48, 6.59, 6.7, 6.82,
        6.92, 7.08, 7.23, 7.37, 7.45, 7.53, 7.59, 7.65, 7.83, 7.96, 8.07, 8.17, 8.38, 8.57, 8.59,
        8.61, 8.71, 8.76, 8.85, 8.9, 9.06, 9.21, 9.28, 9.5, 9.59, 9.69, 9.73, 9.88, 9.98, 10.27,
        10.33, 10.55, 10.63, 10.9, 11.29
    ]
    true_distances_m = [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4,
        3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
        5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4,
        6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
        8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4,
        9.5, 9.6, 9.7, 9.8, 9.9, 10.0
    ]
    
    @staticmethod
    def correct_distance(measured_m, lighting_condition):
        pos = bisect.bisect_left(lighting_condition, measured_m) # Bestimme Position in der LUT
        if pos == 0:
            # Alle Werte unterhalb des Bereichs wird nicht korrigiert
            return measured_m
        elif pos == len(lighting_condition):
            # Alle Werte außerhalb des Bereichs wird auf 10m gesetzt
            return 10.0
        else: 
            # Interpolation zwischen zwei Werten
            prev_meas = lighting_condition[pos-1]
            next_meas = lighting_condition[pos]
            prev_true = MainScreen.true_distances_m[pos-1]
            next_true = MainScreen.true_distances_m[pos]
            alpha = (measured_m - prev_meas) / (next_meas - prev_meas)
            return prev_true + alpha * (next_true - prev_true)
    
    
    def __init__(self, lichtbedingung=None, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        
        self.focus_locked_once = False # Variable für schnellste Verfolgung

        # D455 RGB-Bild
        self.video_image = Image(size_hint=(0.8, 0.8), pos_hint={'center_x': 0.4, 'center_y': 0.5})
        self.add_widget(self.video_image)
        
        # Tiefenprofil ganz rechts
        self.profile_image = Image(size_hint=(0.18, 0.8), pos_hint={'right': 1.0, 'center_y': 0.5})
        self.add_widget(self.profile_image)
        
        # fps Leiste
        self.status_bar = BoxLayout(size_hint=(None, None), size=(40, 30), pos=(20, 10))
        self.status_bar.pos = (20, self.height - self.status_bar.height - 10)
        self.bind(size=self._update_status_bar_pos)
        self.fps_label = Label(text="FPS: 0")
        self.status_bar.add_widget(self.fps_label)
        self.add_widget(self.status_bar)
        
        # Anweisungsleiste
        self.instruction_bar = BoxLayout(size_hint=(None, None), size=(40, 30), pos=(200, 10))
        self.instruction_bar.pos = (400, self.height - self.instruction_bar.height - 10)
        self.bind(size=self._update_instruction_bar_pos)
        self.intruction_label = Label(text="Antippen, um das Fokussobjekt auszuwählen")
        self.instruction_bar.add_widget(self.intruction_label)
        self.add_widget(self.instruction_bar)
        
        # Fokusracking-Zeit Slider
        self.focus_slider = Slider(
            min=0, max=10, value=0,
            size_hint=(0.3, 0.05),
            pos_hint={'x': 0.20, 'y': 0.01}
        )
        self.focus_slider.bind(value=self.on_slider_value_change)
        self.focus_label = Label(
            text=f'Fokusszeit: {self.focus_slider.value:.2f} s',
            size_hint=(0.6, 0.05),
            pos_hint={'x': 0.05, 'y': 0.05},
            color=[1, 1, 1, 1]
        )
        self.add_widget(self.focus_slider)
        self.add_widget(self.focus_label)
        
        # Reset-Button zum Zurücksetzen des Trackings
        self.reset_button = Button(
            text="Reset Tracking",
            size_hint=(None, None),
            size=(140, 40),
            pos_hint={'x': 0.01, 'y': 0.02}
        )
        self.reset_button.bind(on_press=self.reset_tracking)
        self.add_widget(self.reset_button)
        
        
        self.frame_width = 1280
        self.frame_height = 720
        
        # Initialisierung für ROI und Dragging
        self.roi_start = None
        self.roi_end = None
        self.dragging = False
        self.selected_corner = None
        self.corner_size = 20
        
        self.selected_id = None # ID der ausgewählten Person
        self.person_tracks = [] # Liste der verfolgten Personen
        self.initialize_components()
        self.person_tracker = Sort() # Sort-Trackers
        
        # --- Initialisierung für Optical Flow ---
        self.of_point_selected = False
        self.of_point = ()
        self.of_old_points = None
        self.of_old_gray = None
        
        # --- LUT Auswahl je nach Lichtbedingung ---
        if lichtbedingung == "Drinnen - Gutes Licht":
            self.lighting_condition = self.inside_good_lighting 
        elif lichtbedingung == "Drinnen - Schlechtes Licht":
            self.lighting_condition = self.inside_bad_lighting
        elif lichtbedingung == "Draußen - Gutes Licht":
            self.lighting_condition = self.outside_good_lighting
        elif lichtbedingung == "Draußen - Schlechtes Licht":
            self.lighting_condition = self.outside_bad_lighting 
        else:
            self.lighting_condition = self.inside_bad_lighting
        
        # --- Initialisierung der Steuerung des Motor ---
        self.motor_queue = mp.Queue()
        self.stop_motor_event = mp.Event()
        self.last_target_distance = None
        self.last_target_steps = None
        self.current_motor_steps = Value('i', 0)
        self.motor_process = mp.Process(target=self.motor_worker, args=(self.motor_queue, self.stop_motor_event,
                                                                        self.current_motor_steps, self.focus_slider.value))
        self.motor_process.start()
        
        # self.focus_distance = 0.6 
        self.stCam_offset = 0.075 # Wegen der Position der Stcam
        self.focus_plane_start = 0.6 #Beim Starten ist die Fokusebene bei 0.6m
        self.white_bar_pos = 0 # Wo fokussiert werden soll
        self.focus_distance = 0
    
        self.prev_time = time.time() # für FPS-Berechnung

        Clock.schedule_interval(self.update, 1.0 / 30.0)
    
    
    def _update_status_bar_pos(self, *args):
        self.status_bar.pos = (20, self.height - self.status_bar.height - 10)
    
    
    def _update_instruction_bar_pos(self, *args):
        self.instruction_bar.pos = (400, self.height - self.instruction_bar.height - 10)
    
    
    def on_slider_value_change(self, instance, value):
        self.focus_label.text = f'Fokusszeit: {value:.2f} s'
    
    
    def initialize_components(self):
        try:
            # 3 Modelle werden verwendet:
            model_name = "yolo11n_silu_coco--640x640_quant_hailort_hailo8_1"
            model_name_face = "yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8_1"
            model_name_seg = "yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8_1"
            
            inference_host_address="@local"
            zoo_url = "models/yolo11n_silu_coco--640x640_quant_hailort_hailo8_1"
            zoo_url_face = "models/yolov8n_relu6_widerface_kpts--640x640_quant_hailort_hailo8_1"
            zoo_url_seg = "models/yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8_1"
            token = ""
            device_type = "HAILORT/HAILO8" # AI HAT
            self.model = dg.load_model(
                model_name=model_name,
                inference_host_address=inference_host_address,
                zoo_url=zoo_url,
                token=token,
                device_type=device_type,
            )
            self.model_face = dg.load_model(
                model_name=model_name_face,
                inference_host_address=inference_host_address,
                zoo_url=zoo_url_face,
                token=token,
                device_type=device_type,
            )
            self.seg_model = dg.load_model(
                model_name=model_name_seg,
                inference_host_address=inference_host_address,
                zoo_url=zoo_url_seg,
                token=token,
                device_type=device_type,
            )

            # Dummy-Aufruf, um Modelle zu initialisieren, ohne das hängt sich es für 1-2s auf,
            # wenn man zum ersten Mal eine Trackingsperson auswählt,
            # da die Modelle erst initialisiert werden müssen (1-2s)
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model_face(dummy)
            self.seg_model(dummy)

            # RealSense D455 initialisieren
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            profile = self.pipeline.start(config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            depth_sensor.set_option(rs.option.laser_power, 360) # Laserleistung
            
            # Hole die RGB- und Tiefenbilder
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            # Ausrichtung der Tiefenbilder auf RGB-Bilder
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            if not color_frame:
                raise Exception("RealSense konnte kein RGB- oder TiefenBild liefern")
            
            
            # Red ROI-Start- und Endpunkte
            self.frame_width = color_frame.get_width()
            self.frame_height = color_frame.get_height()
            # 50mm: (0.37, 0.37) (0.61, 0.65)
            self.roi_start = [int(self.frame_width * 0.37), int(self.frame_height * 0.37)] 
            self.roi_end = [int(self.frame_width * 0.61), int(self.frame_height * 0.65)]
            
            self.fps_history = deque(maxlen = 10) # fps stabisieren
            self.fps = 0.0
            

        except Exception as e:
            # Fehler bei der Initialisierung anzeigen
            error_popup = Popup(title='Initialisierungsfehler',
                              content=Label(text=f'Fehler: {str(e)}'),
                              size_hint=(0.8, 0.4))
            error_popup.open()
            self.cleanup()
    
    # 
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
        # Prüfe, ob eine Ecke angefasst wurde. Für red ROI dragging
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
        # Prüfe, ob auf ein Trackingobjekt geklickt wurde (kleinste Fläche bei Überlappung)
        # Also kleinere BBs hat höhere Priorität, falls die BBs überlappen
        matching_tracks = []
        for track in self.person_tracks:
            x1, y1, x2, y2, track_id = track
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                matching_tracks.append((track, area))
        if matching_tracks:
            matching_tracks.sort(key=lambda x: x[1])
            selected_track = matching_tracks[0][0]
            self.selected_id = selected_track[4]
            self.of_point_selected = False
            self.focus_locked_once = False # Reset Fokus-Status
            return True
        # Prüfe, ob in die ROI geklickt wurde (Optical-Flow-Punkt setzen)
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
        # Begrenzungen wie gehabt
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
            # Nach ROI-Resize Optical-Flow zurücksetzen!
            self.of_point_selected = False
            self.of_point = ()
            self.of_old_points = None
            self.of_old_gray = None
        return result
    
    
    def get_non_overlapping_crop(self, frame, tracking_bbox, other_bboxes):
        """
        Gibt einen Crop der Trackingperson zurück, bei dem überlappende Bereiche mit anderen Personen-BBs ausgeblendet (schwarz) sind.
        """
        try:
            x1, y1, x2, y2 = tracking_bbox
            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(frame.shape[1], x2)
            y2_crop = min(frame.shape[0], y2)
            if x2_crop <= x1_crop or y2_crop <= y1_crop:
                return None, None, None, None, None
            crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            mask = np.ones(crop.shape[:2], dtype=np.uint8)
            for ox1, oy1, ox2, oy2, _ in other_bboxes:
                ox1_rel = max(0, ox1 - x1_crop)
                oy1_rel = max(0, oy1 - y1_crop)
                ox2_rel = min(crop.shape[1], ox2 - x1_crop)
                oy2_rel = min(crop.shape[0], oy2 - y1_crop)
                if ox2_rel > ox1_rel and oy2_rel > oy1_rel:
                    mask[oy1_rel:oy2_rel, ox1_rel:ox2_rel] = 0
            black_ration = np.mean(mask == 0)
            if black_ration > 0.5:
                # Wenn mehr als 50% der Fläche schwarz ist, wird der Crop verworfen
                return crop, x1_crop, y1_crop, x2_crop, y2_crop
            crop[mask == 0] = [0, 0, 0]
            return crop, x1_crop, y1_crop, x2_crop, y2_crop
        except:
            return None, None, None, None, None

    
    def update(self, dt):
        try:
            # prüft, ob das Objekt self das Attribut 'pipeline' hat
            if not hasattr(self, 'pipeline'):
                return
            
            frames = self.pipeline.wait_for_frames() # synchronisierte Frames holen
            aligned_frames = self.align.process(frames) # ausgerichtete Frames holen
            color_frame = aligned_frames.get_color_frame() # RGB-Bild holen
            depth_frame = aligned_frames.get_depth_frame() # Tiefenbild holen
            
            # Prüfe, ob Frames erfolgreich geladen wurden
            if not color_frame or not depth_frame:
                return
            
            # Konvertiere Frames in NumPy-Arrays wegen OpenCV
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # in Graubild konverieren für Optical Flow
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.roi_start is None or self.roi_end is None:
                return
            
            roi_x1, roi_y1 = self.roi_start
            roi_x2, roi_y2 = self.roi_end
            roi_x1 = max(0, min(self.frame_width - 10, roi_x1))
            roi_y1 = max(0, min(self.frame_height - 10, roi_y1))
            roi_x2 = max(roi_x1 + 10, min(self.frame_width, roi_x2))
            roi_y2 = max(roi_y1 + 10, min(self.frame_height, roi_y2))
            self.roi_start = [roi_x1, roi_y1]
            self.roi_end = [roi_x2, roi_y2]
            
            # ROI-Bild für Personenerkennung
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            if roi_frame.size == 0:
                return
            
            person_results = self.model(roi_frame) # Personenerkennung im ROI-Bild
            person_bboxes = [] # Liste für Bounding-Boxes der Personen
            
            # Gehe durch alle erkannten Objekte und filtere nach Personen
            for result in person_results.results:
                label = result['label']
                conf = result['score']
                x1_p, y1_p, x2_p, y2_p = result['bbox']
                if label == "person":
                    x1, y1, x2, y2 = int(x1_p), int(y1_p), int(x2_p), int(y2_p)
                    x1_full = x1 + roi_x1
                    y1_full = y1 + roi_y1
                    x2_full = x2 + roi_x1
                    y2_full = y2 + roi_y1
                    person_bboxes.append((x1_full, y1_full, x2_full, y2_full))
            
            # Die BBs werden in ein NumPy-Array umgewandelt, wenn sie vorhanden sind,
            # sonst wird ein leeres Array mit der Form (0,5) erstellt, da Sort das erwartet      
            dets = np.array(person_bboxes) if person_bboxes else np.empty((0, 5))
            
            # erkannte Personen werden an Tracker übergeben
            tracks = self.person_tracker.update(dets)
            self.person_tracks = tracks

            # --- Tiefenprofil NUR für Trackingperson ---
            canvas_height = 900
            scaled_width = int(self.frame_width * 0.18)
            profile_canvas = np.zeros((canvas_height, scaled_width, 3), dtype=np.uint8)
            profile_canvas[:] = [30, 30, 30]
            background_depth = 10.0
            y_scale = (canvas_height - 50) / background_depth
            sample_ratio = 0.01 # auf Maske der getrackten Person
            r = 2

            
            # --- Anzeige: KI-Tracking oder Optical Flow ---
            for track in tracks:
                x1, y1, x2, y2, track_id = track.astype(int)
                color = (0, 255, 0) if track_id==self.selected_id else  (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if self.selected_id is not None and self.selected_id == track_id:
                    # --- Overlapping-Logik: andere BBs sammeln ---
                    other_bboxes = []
                    for t2 in tracks:
                        x1o, y1o, x2o, y2o, id2 = t2.astype(int)
                        if id2 != track_id:
                            other_bboxes.append((x1o, y1o, x2o, y2o, id2))
                    # --- Crop mit überlappende Bereiche ausgeblendet ---
                    person_crop, x1_crop, y1_crop, x2_crop, y2_crop = self.get_non_overlapping_crop(frame, (x1, y1, x2, y2), other_bboxes)
                    if person_crop is None:
                        continue
                    
                    face_results = self.model_face(person_crop) # Gesichtserkennung im Personencrop

                    found_face = False
                    face_box = None
                    face_text = ""
                    corrected_face_distance = None
                    uncorrected_face_distance = None

                    for result in face_results.results:
                        if result['label'] == "face" and result['score']>0.3:
                            found_face = True
                            fx1, fy1, fx2, fy2 = map(int, result['bbox'])
                            fx1 += x1_crop
                            fy1 += y1_crop
                            fx2 += x1_crop
                            fy2 += y1_crop

                            face_area = depth_image[fy1:fy2, fx1:fx2] # Tiefenwerte im Gesichtsausschnitt
                            valid = face_area[face_area > 0] # nur gültige Tiefenwerte
                            if valid.size > 0:
                                uncorrected_face_distance = np.mean(valid) / 1000
                            else:
                                uncorrected_face_distance = 0.0
                            
                            corrected_face_distance = self.correct_distance(uncorrected_face_distance, self.lighting_condition)

                            self.focus_distance = corrected_face_distance + self.stCam_offset

                            face_box = (fx1, fy1, fx2, fy2)
                            face_text = f"corr: {corrected_face_distance:.2f} uncorr:{uncorrected_face_distance:.2f}"
                    
                    seg_result = self.seg_model(person_crop) # Segmentierung im Personencrop
                    for seg in seg_result.results:
                        if seg['label'] == "person":
                            mask = seg.get('mask')
                            if mask is None:
                                mask = seg.get('segmentation_mask')
                            if mask is not None:
                                if mask.shape[:2] != person_crop.shape[:2]:
                                    mask = cv2.resize(mask, (person_crop.shape[1], person_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                                # --- Maskenüberlagerung
                                mask_bin = (mask > 0.5).astype(np.uint8)
                                color_mask = np.zeros_like(person_crop, dtype=np.uint8)
                                color_mask[:, :, 2] = 255  # Rotkanal
                                mask_indices = mask_bin.astype(bool)
                                person_crop[mask_indices] = cv2.addWeighted(
                                    person_crop[mask_indices], 0.6, color_mask[mask_indices], 0.4, 0
                                )
                                frame[y1:y2, x1:x2] = person_crop
                                # --- Tiefenprofil ---
                                y_coords, x_coords = np.where(mask_bin)
                                num_points = len(y_coords)
                                if num_points > 0:
                                    sample_size = max(1, int(num_points * sample_ratio))
                                    indices = np.random.choice(num_points, sample_size, replace=False)
                                    x_sample = x_coords[indices] + x1
                                    y_sample = y_coords[indices] + y1
                                    depths = depth_image[y_sample, x_sample] / 1000.0
                                    mask_distance = depths[depths > 0]

                                    if mask_distance.size > 0:
                                        uncorrected_mask_distance = np.mean(mask_distance)
                                    else:
                                        uncorrected_mask_distance = 0.0

                                    corrected_mask_distance = self.correct_distance(uncorrected_mask_distance, self.lighting_condition)


                                    if not found_face:
                                        self.focus_distance = corrected_mask_distance + self.stCam_offset

                                    diff = uncorrected_mask_distance - corrected_mask_distance # 
                                    
                                    x_coords_reduced = (x_sample * (scaled_width / self.frame_width)).astype(int)
                                    # Tiefeninformationen für das Tiefenprofil wird nur durch die Diffenz berechnet
                                    # nicht durch correct_distance um Leistung zu sparen
                                    y_positions = 50 + ((background_depth - (depths-diff+self.stCam_offset)) * y_scale).astype(int)
                                    x_coords_reduced = np.clip(x_coords_reduced, 0, scaled_width - 1)
                                    y_positions = np.clip(y_positions, 0, canvas_height - 1)
                                    for x, y in zip(x_coords_reduced, y_positions):
                                        x1b, x2b = max(0, x - r), min(scaled_width, x + r + 1)
                                        y1b, y2b = max(0, y - r), min(canvas_height, y + r + 1)
                                        profile_canvas[y1b:y2b, x1b:x2b] = [0, 255, 0]
                    
                    # BB für erkannte Gesichter zeichnen
                    if found_face and face_box is not None:
                        fx1, fy1, fx2, fy2 = face_box
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                        cv2.putText(frame, face_text, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if self.selected_id != track_id:
                    
                    sample_ratio_not_tracking = 0.001

                    person_crop_not_tracking = frame[y1:y2, x1:x2].copy()
                    
                    seg_result_not_tracking = self.seg_model(person_crop_not_tracking) # Segmentierung im Personencrop
                    for seg in seg_result_not_tracking.results:
                        if seg['label'] == "person":
                            mask = seg.get('mask')
                            if mask is None:
                                mask = seg.get('segmentation_mask')
                            if mask is not None:
                                if mask.shape[:2] != person_crop_not_tracking.shape[:2]:
                                    mask = cv2.resize(mask, (person_crop_not_tracking.shape[1], person_crop_not_tracking.shape[0]), interpolation=cv2.INTER_NEAREST)
                                # --- Maskenüberlagerung
                                mask_bin = (mask > 0.5).astype(np.uint8)
                                
                                frame[y1:y2, x1:x2] = person_crop_not_tracking
                                # --- Tiefenprofil ---
                                y_coords, x_coords = np.where(mask_bin)
                                num_points = len(y_coords)
                                if num_points > 0:
                                    sample_size = max(1, int(num_points * sample_ratio_not_tracking))
                                    indices = np.random.choice(num_points, sample_size, replace=False)
                                    x_sample = x_coords[indices] + x1
                                    y_sample = y_coords[indices] + y1
                                    depths_not_tracking = depth_image[y_sample, x_sample] / 1000.0
                                    mask_distance_not_tracking = depths_not_tracking[depths_not_tracking > 0]

                                    if mask_distance_not_tracking.size > 0:
                                        uncorrected_mask_distance_not_tracking = np.mean(mask_distance_not_tracking)
                                    else:
                                        uncorrected_mask_distance_not_tracking = 0.0

                                    corrected_mask_distance_not_tracking = self.correct_distance(uncorrected_mask_distance_not_tracking, 
                                                                                                 self.lighting_condition)

                                    diff_not_tracking = uncorrected_mask_distance_not_tracking - corrected_mask_distance_not_tracking # 
                                    
                                    x_coords_reduced = (x_sample * (scaled_width / self.frame_width)).astype(int)
                                    # Tiefeninformationen für das Tiefenprofil wird nur durch die Diffenz berechnet
                                    # nicht durch correct_distance um Leistung zu sparen
                                    y_positions = 50 + ((background_depth - (depths_not_tracking-diff_not_tracking+
                                                                             self.stCam_offset)) * y_scale).astype(int)
                                    x_coords_reduced = np.clip(x_coords_reduced, 0, scaled_width - 1)
                                    y_positions = np.clip(y_positions, 0, canvas_height - 1)
                                    for x, y in zip(x_coords_reduced, y_positions):
                                        x1b, x2b = max(0, x - r), min(scaled_width, x + r + 1)
                                        y1b, y2b = max(0, y - r), min(canvas_height, y + r + 1)
                                        profile_canvas[y1b:y2b, x1b:x2b] = [128, 128, 128]
                    
                    label = f"Fokussperson {track_id}: {self.focus_distance:.2f}m" if self.selected_id == track_id else f"Person {track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    
            
            
            # Optical Flow Tracking
            if self.of_point_selected and self.of_old_points is not None:
                roi_gray = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                if self.of_old_gray is None:
                    self.of_old_gray = roi_gray.copy()
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.of_old_gray, roi_gray, self.of_old_points, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                good_new = new_points[status.flatten() == 1]
                anzahl_trackingpunkte = len(good_new)
                if good_new.size > 0:
                    good_new = good_new.reshape(-1, 2)
                    mean_x, mean_y = np.mean(good_new, axis=0)

                    # Focusdistanz berechnen durch eine Fläche von 10x10 um den Trackingspunkt
                    all_depths = []
                    for x, y in good_new:
                        x_track = int(x) + roi_x1
                        y_track = int(y) + roi_y1
                        x1 = max(0, x_track - 5)
                        x2 = min(depth_image.shape[1], x_track + 5)
                        y1 = max(0, y_track - 5)
                        y2 = min(depth_image.shape[0], y_track + 5)
                        window = depth_image[y1:y2, x1:x2] / 1000.0
                        all_depths.extend(window.flatten())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Nur gültige Werte (>0)
                    depths = [d for d in all_depths if d > 0]
                    if depths:
                        uncorrected_OF_distance = np.median(depths)
                    else:
                        uncorrected_OF_distance = 0
                    corrected_OF_distance = self.correct_distance(uncorrected_OF_distance, self.lighting_condition)
                    if corrected_OF_distance > 0:
                        # sonst springt der Fokus auf den Nahpunkt zurück
                        self.focus_distance = corrected_OF_distance + self.stCam_offset
                    cv2.putText(
                        frame,
                        f"{self.focus_slider.value:.2f} corr: {corrected_OF_distance:.2f} uncorr: {uncorrected_OF_distance:.2f} fD: {self.focus_distance:.2f}",
                        (int(mean_x) + roi_x1, int(mean_y) + roi_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    self.of_old_points = good_new.reshape(-1, 1, 2)
                else:
                    self.of_point_selected = False
                    self.of_old_points = None
                self.of_old_gray = roi_gray.copy()

            

            # Motor
            target_steps = self.distance_to_steps(self.focus_distance)
            steps_diff = abs(target_steps - self.current_motor_steps.value)

            if steps_diff <= 1 and not self.focus_locked_once:
                self.focus_locked_once = True

            if (self.last_target_distance is None or 
                abs(self.focus_distance - self.last_target_distance) > self.Hysteresis_Threshold):
                if self.focus_locked_once:
                # Maximale Geschwindigkeit nach erstem Scharfstellen
                    self.motor_queue.put((target_steps, 0.001))
                else:
                    # Slidergeschwindigkeit beim ersten Fokussieren
                    self.motor_queue.put((target_steps, self.focus_slider.value))
                self.last_target_distance = self.focus_distance
                self.last_target_steps = target_steps



            # Der Bereich unterhalb dem Nahpunkt - Tiefensprofil
            not_valid = 50 + int((background_depth - self.focus_plane_start) * y_scale)
            cv2.line(profile_canvas, (0, not_valid), (scaled_width, not_valid), (0, 0, 255), 2)
            cv2.line(profile_canvas, (0, profile_canvas.shape[0]-2), (scaled_width, profile_canvas.shape[0]-2), (0, 0, 255), 2)

            cv2.line(profile_canvas, (0, profile_canvas.shape[0]), (scaled_width, not_valid), (0, 0, 255), 2)
            cv2.line(profile_canvas, (0, not_valid), (scaled_width, profile_canvas.shape[0]), (0, 0, 255), 2)
            
            # Wo fokusiert wird
            self.white_bar_pos = 50 + int((background_depth - self.focus_distance) * y_scale)
            cv2.line(profile_canvas, (0, self.white_bar_pos), (scaled_width, self.white_bar_pos), (255, 255, 255), 5)

            focus_plane_pos = self.focus_plane_pos(self.current_motor_steps.value)

            if self.current_motor_steps.value == 0:
                focus_plane_pos = self.focus_plane_start
            else:
                focus_plane_pos = self.focus_plane_pos(self.current_motor_steps.value)
            
            # Fokus-Ebene zeichnen
            focus_plane_y = 50 + int((background_depth - focus_plane_pos) * y_scale)
            cv2.line(profile_canvas, (0, focus_plane_y), (scaled_width, focus_plane_y), (0, 255, 0), 4)

            

            # Tiefenprofil Achsen zeichnen
            for y in range(0, int(background_depth) + 1):
                y_pos = 50 + int((background_depth - y) * y_scale)
                if 0 <= y_pos < canvas_height:
                    cv2.line(profile_canvas, (0, y_pos), (20, y_pos), (255, 255, 255), 1)
                    cv2.putText(profile_canvas, f"{y}m", (25, y_pos + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        

            # Profilbild in Kivy-Image anzeigen
            buf_profile = cv2.flip(profile_canvas, 0).tobytes()
            texture_profile = Texture.create(size=(profile_canvas.shape[1], profile_canvas.shape[0]), colorfmt='bgr')
            texture_profile.blit_buffer(buf_profile, colorfmt='bgr', bufferfmt='ubyte')
            self.profile_image.texture = texture_profile

            # ROI zeichnen (rot) und Ecken (gelb)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            corner_color = (0, 255, 255)
            corner_size = 15
            for (cx, cy) in [(roi_x1, roi_y1), (roi_x2, roi_y1), (roi_x1, roi_y2), (roi_x2, roi_y2)]:
                cv2.rectangle(frame,
                              (int(cx) - corner_size, int(cy) - corner_size),
                              (int(cx) + corner_size, int(cy) + corner_size),
                              corner_color, -1)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.video_image.texture = texture

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
            self.prev_time = curr_time
            if not hasattr(self, 'fps_history'):
                self.fps_history = deque(maxlen=10)
            self.fps_history.append(fps)
            smoothed_fps = np.median(self.fps_history)
            self.fps_label.text = f"FPS: {int(smoothed_fps)}"
        except Exception as e:
            print(f"Error in update: {e}")

    
    
    def cleanup(self):
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
            if hasattr(self, 'stop_motor_event'):
                self.stop_motor_event.set()
            if hasattr(self, 'motor_process'):
                self.motor_process.join(timeout=2)
        except Exception as e:
            print(f"Fehler beim Cleanup: {e}")
    
    
    def reset_tracking(self, instance):
        self.selected_id = None
        self.of_point_selected = False
        self.of_point = ()
        self.of_old_points = None
        self.of_old_gray = None
        self.white_bar_pos = 0
        self.focus_locked_once = False # Fokus-Status zurücksetzen

# --- Integration in AMACUSApp ---
class AMACUSApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical')
        # Ladebildschirm zuerst anzeigen
        self.loading_screen = LoadingScreen(on_finished_callback=self.show_calibration)
        self.root.add_widget(self.loading_screen)
        return self.root

    def show_calibration(self):
        # Ladebildschirm entfernen, Kalibrierung anzeigen
        self.root.clear_widgets()
        self.calibration_screen = CalibrationScreen(main_app=self)
        self.root.add_widget(self.calibration_screen)

    def start_main_program(self):
        lichtbedingung = self.calibration_screen.steps[2].dropdown_value
        self.root.clear_widgets()
        self.main_screen = MainScreen(lichtbedingung=lichtbedingung)
        self.root.add_widget(self.main_screen)

    def on_stop(self):
        if hasattr(self, 'main_screen'):
            self.main_screen.cleanup()

if __name__ == '__main__':
    AMACUSApp().run()
