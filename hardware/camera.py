import numpy as np
import pyrealsense2 as rs  #  hartes Import wie im Original

class RealSenseCamera:
    def __init__(self):
        #  feste Streams wie im Original
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.laser_power, 360)  #  ohne try/except

        #  Align auf Farbstream
        self.align = rs.align(rs.stream.color)
        self.started = True

    def get_aligned_frames(self):
        #  Frames holen und auf Color alignen, dann in NumPy wandeln (funktional wie Original)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return frame, depth_image

    def stop(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()