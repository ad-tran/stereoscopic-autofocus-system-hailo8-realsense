from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.dropdown import DropDown
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.graphics import Color, Rectangle, Line
from kivy.properties import BooleanProperty, StringProperty, ListProperty
from kivy.core.window import Window

from utils.config import LIGHTING_OPTIONS, LOGO_PATH


class CalibrationStep(GridLayout):
    text = StringProperty("")
    checked = BooleanProperty(False)
    dropdown_options = ListProperty([])
    dropdown_value = StringProperty("Auswählen")

    def __init__(self, step_num, text, has_dropdown=False, options=None, font_size='30sp', **kwargs):
        super(CalibrationStep, self).__init__(**kwargs)
        self.has_dropdown = has_dropdown
        self.cols = 3 if has_dropdown else 2
        self.rows = 1
        self.size_hint_y = None
        self.height = 40

        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        self.selected_option_btn = None

        self.checkbox = CheckBox(active=self.checked, size_hint_x=None, width=250, color=[0, 0, 0, 1])
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
            for option in options or []:
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

            with self.dropdown_btn.canvas.after:
                self.border_color = Color(0, 0, 0, 1)
                self.border = Line(rectangle=(0, 0, self.dropdown_btn.width, self.dropdown_btn.height), width=2)

            def update_border(instance, value):
                self.border.rectangle = (920, 0, instance.width, instance.height)

            self.dropdown_btn.bind(size=update_border, pos=update_border)

            def hide_border(instance):
                self.border_color.a = 0

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
        self.steps_layout = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        self.steps_layout.bind(minimum_height=self.steps_layout.setter('height'))
        with self.steps_layout.canvas.before:
            Color(1, 1, 1, 1)
            self.steps_rect = Rectangle(size=self.steps_layout.size, pos=self.steps_layout.pos)
        self.steps_layout.bind(size=self._update_steps_rect, pos=self._update_steps_rect)

        self.steps = [
            CalibrationStep(1, "Stellen Sie den Fokusring auf den minimalen Fokusabstand ein."),
            CalibrationStep(2, "Bringen Sie den Schrittmotor an den Fokusring und befestigen ihn."),
            CalibrationStep(
                3,
                "Wählen Sie die Lichtbedingung Ihrer Szene aus:",
                True,
                ["Drinnen - Gutes Licht", "Drinnen - Schlechtes Licht", "Draußen - Gutes Licht", "Draußen - Schlechtes Licht"],
                size_hint_x=0.89,
                width=250
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
                color=[1, 1, 1, 1],
                font_size='30sp'
            )
            popup = Popup(title='Fehler 006', content=content, size_hint=(0.5, 0.2))
            with popup.canvas.before:
                Color(1, 1, 1, 1)
                popup.rect = Rectangle(size=popup.size, pos=popup.pos)
            popup.bind(size=self._update_popup_rect, pos=self._update_popup_rect)
            popup.open()

    def _update_popup_rect(self, instance, value):
        instance.rect.pos = instance.pos
        instance.rect.size = instance.size