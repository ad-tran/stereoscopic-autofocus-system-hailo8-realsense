from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window

from utils.config import WINDOW_SIZE
from gui.loading_screen import LoadingScreen
from gui.calibration_screen import CalibrationScreen
from gui.main_screen import MainScreen


class AMACUSApp(App):
    def build(self):
        Window.size = WINDOW_SIZE
        Window.maximize()
        root = BoxLayout(orientation='vertical')
        self.loading_screen = LoadingScreen(on_finished_callback=self.show_calibration)
        root.add_widget(self.loading_screen)
        self.root_widget = root
        return root

    def show_calibration(self):
        self.root_widget.clear_widgets()
        self.calibration_screen = CalibrationScreen(main_app=self)
        self.root_widget.add_widget(self.calibration_screen)

    def start_main_program(self):
        lichtbedingung = self.calibration_screen.steps[2].dropdown_value
        self.root_widget.clear_widgets()
        self.main_screen = MainScreen(lichtbedingung=lichtbedingung)
        self.root_widget.add_widget(self.main_screen)

    def on_stop(self):
        if hasattr(self, 'main_screen'):
            self.main_screen.cleanup()


if __name__ == '__main__':
    AMACUSApp().run()