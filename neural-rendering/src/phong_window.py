import os.path

import json
import moderngl
import numpy as np
import random
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.param_buffer = []

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def on_render(self, time: float, frame_time: float):
        if self.frame >= self.frame_count:
            with open(os.path.join(self.output_path, "dataset.json"), "w") as f:
                json.dump(self.param_buffer, f, indent=2)

            self.wnd.close()
            return

        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        model_translation = [
            random.uniform(-5.0, 5.0),
            random.uniform(-5.0, 3.0),
            random.uniform(-15.0, 5.0)
        ]
        material_diffuse = [
            random.randint(0, 255) / 255.0, 
            random.randint(0, 255) / 255.0,
            random.randint(0, 255) / 255.0
        ]
        material_shininess = random.uniform(3.0, 20.0)
        light_position = [
            random.uniform(-20.0, 20.0),
            random.uniform(-20.0, 20.0),
            random.uniform(-20.0, 20.0)
        ]

        camera_position = [5.0, 5.0, 15.0]
        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_position,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        camera_relative_translation = np.array(model_translation) - np.array(camera_position)
        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()
        if self.output_path:
            filename = f"image_{self.frame:04}"
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(self.output_path, f'images/{filename}.png'))

            params = {
                "image_filename": f"{filename}.png",
                "model_translation": model_translation,
                "material_diffuse": material_diffuse,
                "material_shininess": material_shininess,
                "light_position": light_position,
                "camera_position": camera_position,
                "frame": self.frame
            }
            self.param_buffer.append(params)
            self.frame += 1
