from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle
from kivy.uix.stencilview import StencilView
from kivy.core.window import Window
from kivy.clock import Clock

from utils.config import LOGO_PATH


class MaskedLogo(StencilView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = (800, 800)
        self.pos = (Window.width / 2 - self.size[0] / 2, Window.height / 2 - self.size[1] / 2)
        self.logo = Image(
            source='/home/amacus/hailo_examples/LogoName.png',
            size_hint=(None, None),
            size=self.size,
            pos=self.pos
        )
        self.add_widget(self.logo)
        self.progress_value = 0
        with self.canvas:
            self.mask_color = Color(1, 1, 1, 1)
            self.mask_rect = Rectangle(pos=self.pos, size=self.size)
        Clock.schedule_interval(self.update_progress, 0.03)

    def update_progress(self, dt):
        if self.progress_value < self.size[0]:
            self.progress_value += 30
            self.mask_rect.pos = (self.pos[0] + self.progress_value, self.pos[1])
        else:
            Clock.unschedule(self.update_progress)
            if hasattr(self, 'parent') and hasattr(self.parent, 'on_loading_finished'):
                self.parent.on_loading_finished()


class LoadingScreen(FloatLayout):
    def __init__(self, on_finished_callback=None, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.bg = Rectangle(size=Window.size)
        self.mask = MaskedLogo()
        self.add_widget(self.mask)
        self.on_finished_callback = on_finished_callback

    def on_loading_finished(self):
        if self.on_finished_callback:
            Clock.schedule_once(lambda dt: self.on_finished_callback(), 2)