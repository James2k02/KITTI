import pyrealsense2 as rs
import numpy as np
import cv2

def capture_stereo_ir():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            ir_left = frames.get_infrared_frame(1)
            ir_right = frames.get_infrared_frame(2)

            left_img = np.asanyarray(ir_left.get_data())
            right_img = np.asanyarray(ir_right.get_data())

            cv2.imshow("Left IR", left_img)
            cv2.imshow("Right IR", right_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

capture_stereo_ir()
