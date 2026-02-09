import copy
import glob
import json
import math
import os
import platform
import sys
import tkinter as tk
import tkinter.filedialog as fd
from threading import Thread as thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
import time

from Meshtrics.photometric_metrics import sim_view_point, solve_pnp

isMacOS = (platform.system() == "Darwin")


def pairs(lst):
    i = iter(lst)
    first = prev = item = next(i)
    for item in i:
        yield prev, item
        prev = item
    yield item, first


def unproject_point(point2d, depth, intrinsics, extrinsics):
    x_d = point2d[0]
    y_d = point2d[1]
    fx_d, fy_d, cx_d, cy_d = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    x = (x_d - cx_d) * depth / fx_d
    y = (y_d - cy_d) * depth / fy_d
    z = depth

    x, y, z, pad = np.dot(extrinsics, [x, y, z, 1])

    return np.asarray([x, y, z])


curr_img = []
curr_depth = []


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [1.0, 1.0, 1.0, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [1.0, 1.0, 1.0, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.UNLIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_SHOW_TOOLS = 23
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Unlit", "Lit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.UNLIT, Settings.LIT, Settings.NORMALS, Settings.DEPTH
    ]

    JSON_DICT = {
        "Geometry": {
            "Orthogonality": {
                "Saved": False,
                "Annotation": []
            },
            "Planarity": {
                "Saved": False,
                "Annotation": []
            }
        },
        "Texture": {
            "NoReference": {
                "Saved": False,
                "Annotation": []
            },
            "FullReference": {
                "Saved": False,
                "Annotation": []
            }
        }
    }

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.title_font = gui.Application.instance.add_font(
            gui.FontDescription(typeface='monospace', style=gui.FontStyle.BOLD, point_size=18))

        self.window = gui.Application.instance.create_window(
            "Meshtrics", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        self.picked_points_2D = []
        self.polygon_vertices = []
        self.polygon_vertices_2D = []
        self.labels = []
        self.selecting_polygon = False
        self.model = None

        self.colors = plt.cm.tab10.colors
        self.color_idx = 0

        self._scene.set_on_mouse(self._on_mouse_widget3d)
        self._scene.set_on_key(self._on_key_widget3d)

        frame = self._scene.frame
        fovy = math.radians(60)
        f = frame.height / (2 * math.tan(fovy / 2))
        cx = frame.width / 2
        cy = frame.height / 2
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(frame.width, frame.height, f, f, cx, cy)

        o3d_extrinsics = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(self.o3d_intrinsics, o3d_extrinsics, bounds)
        self.K = np.zeros((3, 3))
        self.K[2, 2] = 1
        self.K[0, 0] = f
        self.K[1, 1] = f
        self.K[0, 2] = cx
        self.K[1, 2] = cy
        self._scene.scene.camera.set_projection(self.K, 0.2, 1000, frame.width, frame.height)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)

        self._tools_panel = gui.Vert(
            0, gui.Margins(0.75 * em, 0.25 * em, 0.75 * em, 0.25 * em))

        metrics_label = gui.Label("Annotation Tools")
        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(metrics_label)
        h.add_stretch()
        self._tools_panel.add_child(h)

        self._tree = gui.TreeView()
        mesh_id = self._tree.add_text_item(self._tree.get_root_item(), "Geometric")
        default_pick = self._tree.add_text_item(mesh_id, "Orthogonality")
        self._tree.add_text_item(mesh_id, "Local Planarity")
        mesh_id = self._tree.add_text_item(self._tree.get_root_item(), "Photometric")
        self._tree.add_text_item(mesh_id, "Full Reference")
        self._tree.add_text_item(mesh_id, "No Reference")
        self._tree.can_select_items_with_children = False
        self._tree.set_on_selection_changed(self._on_tree)
        self._tree.selected_item = default_pick
        self._tools_panel.add_child(self._tree)

        ###################################################
        # Instructions

        # ORTHO

        self._ortho_instuct = gui.Vert(0.25 * em, gui.Margins(0.75 * em, 0.75 * em, 0.75 * em, 0 * em))

        ortho_title = gui.Label("Select planes")
        ortho_title.font_id = self.title_font
        self._ortho_instuct.add_child(ortho_title)

        ortho_label = gui.Label("Please choose planes to test orthogonality"
                                "\nbetween them. Use ctrl+click to pick at"
                                "\nleast 3 points, then hit 'Extract'.")
        self._ortho_instuct.add_child(ortho_label)

        self._extract_plane_button = gui.Button("Extract")
        self._extract_plane_button.horizontal_padding_em = 0.5
        self._extract_plane_button.vertical_padding_em = 0
        self._extract_plane_button.set_on_clicked(self._on_menu_extract_plane)

        self._save_plane_button = gui.Button("Save plane")
        self._save_plane_button.horizontal_padding_em = 0.5
        self._save_plane_button.vertical_padding_em = 0
        self._save_plane_button.set_on_clicked(self._on_menu_save_plane)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._extract_plane_button)
        h.add_child(self._save_plane_button)
        h.add_stretch()
        self._ortho_instuct.add_child(h)

        # PLAN

        self._plan_instuct = gui.Vert(0.25 * em, gui.Margins(0.75 * em, 0.75 * em, 0.75 * em, 1.2 * em))

        plan_title = gui.Label("Select planar region")
        plan_title.font_id = self.title_font
        self._plan_instuct.add_child(plan_title)

        plan_label = gui.Label("Please choose a ROI to test planarity in."
                               "\nUse ctrl+click to draw a polygon"
                               ", then hit 'Extract'.")
        self._plan_instuct.add_child(plan_label)

        self._extract_poly_button = gui.Button("Extract")
        self._extract_poly_button.horizontal_padding_em = 0.5
        self._extract_poly_button.vertical_padding_em = 0
        self._extract_poly_button.set_on_clicked(self._on_menu_extract_poly)

        self._save_poly_button = gui.Button("Save ROI")
        self._save_poly_button.horizontal_padding_em = 0.5
        self._save_poly_button.vertical_padding_em = 0
        self._save_poly_button.set_on_clicked(self._on_menu_save_poly)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._extract_poly_button)
        h.add_child(self._save_poly_button)
        h.add_stretch()
        self._plan_instuct.add_child(h)

        self._plan_instuct.visible = False

        # FULL REF

        self._ref_instuct = gui.Vert(0.25 * em, gui.Margins(0.75 * em, 0.75 * em, 0.75 * em, 0.75 * em))

        ref_title = gui.Label("Reference registration")
        ref_title.font_id = self.title_font
        self._ref_instuct.add_child(ref_title)

        ref_label = gui.Label("Please \"import\" groudtruth photos and"
                              "\nmanually adjust the camera to initial"
                              "\nposition, then click \"refine\".")
        self._ref_instuct.add_child(ref_label)

        self._import_image_button = gui.Button("Import")
        self._import_image_button.horizontal_padding_em = 0.5
        self._import_image_button.vertical_padding_em = 0
        self._import_image_button.set_on_clicked(self._on_menu_import_image)

        h = gui.Horiz(0.25 * em)
        h.add_child(self._import_image_button)
        h.add_stretch()
        self._ref_instuct.add_child(h)

        # Add a list of items
        self.lv = gui.ListView()
        # self.lv.set_items(("Ground", "Trees", "Buildings", "Cars", "People"))
        # self.lv.selected_index = 3 # initially is -1, so now 1
        self.lv.set_on_selection_changed(self._on_list)
        self._ref_instuct.add_child(self.lv)

        self._refine_button = gui.Button("Refine")
        self._refine_button.horizontal_padding_em = 0.5
        self._refine_button.vertical_padding_em = 0
        self._refine_button.set_on_clicked(self._on_menu_refine)

        self._save_corres_button = gui.Button("Save")
        self._save_corres_button.horizontal_padding_em = 0.5
        self._save_corres_button.vertical_padding_em = 0
        self._save_corres_button.set_on_clicked(self._on_menu_save_corres)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._refine_button)
        h.add_child(self._save_corres_button)
        h.add_stretch()
        self._ref_instuct.add_child(h)

        self._ref_instuct.visible = False

        # Add image to half the screen
        # self._image2D = gui.ImageWidget(o3d.io.read_image("../assets/icons/image.png"))
        self._image2D = gui.SceneWidget()
        self._image2D.scene = rendering.Open3DScene(w.renderer)
        self._image2D.scene.set_background([45 / 255, 45 / 255, 45 / 255, 1.0],
                                           o3d.io.read_image("../assets/icons/image.png"))
        self._image2D.visible = False
        self._current_2Dbackground = []

        # Add image for loading mesh progress
        self.logo = cv2.imread("../meshtrics_logo.png", cv2.IMREAD_UNCHANGED)
        self._loading_image = gui.ImageWidget(o3d.geometry.Image(self.logo))
        self._loading_image.visible = False

        # NO REF

        self._no_ref_instuct = gui.Vert(0.25 * em, gui.Margins(0.75 * em, 0.75 * em, 0.75 * em, 0.75 * em))

        no_ref_title = gui.Label("Select texture region")
        no_ref_title.font_id = self.title_font
        self._no_ref_instuct.add_child(no_ref_title)

        no_ref_label = gui.Label("Please choose a ROI to test texture entropy"
                                 "in. Use ctrl+click to draw a polygon"
                                 ", then hit 'Extract'.")
        self._no_ref_instuct.add_child(no_ref_label)

        self._extract_color_button = gui.Button("Extract")
        self._extract_color_button.horizontal_padding_em = 0.5
        self._extract_color_button.vertical_padding_em = 0
        self._extract_color_button.set_on_clicked(self._on_menu_extract_color)

        self._save_color_button = gui.Button("Save ROI")
        self._save_color_button.horizontal_padding_em = 0.5
        self._save_color_button.vertical_padding_em = 0
        self._save_color_button.set_on_clicked(self._on_menu_save_color)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._extract_color_button)
        h.add_child(self._save_color_button)
        h.add_stretch()
        self._no_ref_instuct.add_child(h)

        self._no_ref_instuct.visible = False

        ###################################################
        # IO

        IO = gui.Vert(0, gui.Margins(0, 0, 0, 0))

        IO_label = gui.Label("Input | Output")

        IO.add_child(IO_label)

        self._save_json_button = gui.Button("Export")
        self._save_json_button.horizontal_padding_em = 0.5
        self._save_json_button.vertical_padding_em = 0
        self._save_json_button.set_on_clicked(self._on_menu_save_json)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._save_json_button)
        h.add_stretch()
        IO.add_child(h)

        # self._ortho_instuct.add_child(IO)

        ###################################################

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsable vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(False)

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path + "/*_ibl.ktx"):
            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        material_settings.set_is_open(False)

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)

        self._settings_panel.visible = False
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._tools_panel)
        w.add_child(self._settings_panel)
        w.add_child(self._plan_instuct)
        w.add_child(self._ortho_instuct)
        w.add_child(self._ref_instuct)
        w.add_child(self._image2D)
        w.add_child(self._loading_image)
        w.add_child(self._no_ref_instuct)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, False)
            tools_menu = gui.Menu()
            tools_menu.add_item("Annotation Tools",
                                AppWindow.MENU_SHOW_TOOLS)
            tools_menu.set_checked(AppWindow.MENU_SHOW_TOOLS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Metrics", tools_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Metrics", tools_menu)
                menu.add_menu("Settings", settings_menu)
                # menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_TOOLS,
                                     self._on_menu_toggle_tools_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
                self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        em = self.window.theme.font_size
        r = self.window.content_rect

        if self._image2D.visible:
            self._image2D.frame = gui.Rect(r.x, r.y, r.width / 2, r.height)

            if self._loading_image.visible:
                self._loading_image.frame = gui.Rect(r.x + r.width / 2, r.y, r.width / 2, r.height)
            else:
                self._scene.frame = gui.Rect(r.x + r.width / 2, r.y, r.width / 2, r.height)
        else:
            if self._loading_image.visible:
                self._loading_image.frame = r
            else:
                self._scene.frame = r

        # width = 17 * layout_context.theme.font_size

        frame = self._scene.frame
        fovy = math.radians(60)
        f = frame.height / (2 * math.tan(fovy / 2))
        cx = frame.width / 2
        cy = frame.height / 2
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(frame.width, frame.height, f, f, cx, cy)

        cam_matrix = self._scene.scene.camera.get_view_matrix()
        cam_matrix[1, :] = -cam_matrix[1, :]
        cam_matrix[2, :] = -cam_matrix[2, :]

        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(self.o3d_intrinsics, cam_matrix, bounds)
        self.K = np.zeros((3, 3))
        self.K[2, 2] = 1
        self.K[0, 0] = f
        self.K[1, 1] = f
        self.K[0, 2] = cx
        self.K[1, 2] = cy
        self._scene.scene.camera.set_projection(self.K, 0.2, 1000, frame.width, frame.height)


        ###############################
        ## metrics
        preferred_height = self._tools_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height

        # prefered_width = self._tools_panel.calc_preferred_size(
        #                 layout_context, gui.Widget.Constraints()).width

        prefered_width = self._ortho_instuct.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width

        tools_height = min(
            r.height,
            preferred_height - 10000 + 7.2 * em)

        self._tools_panel.frame = gui.Rect(r.get_right() - prefered_width, r.y, prefered_width,
                                           tools_height)

        if self._tools_panel.visible:
            self._on_tree(self._tree.selected_item)

        ###############################
        # ortho
        preferred_height = self._ortho_instuct.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height

        self._ortho_instuct.frame = gui.Rect(r.get_right() - prefered_width, tools_height + em * 1.1, prefered_width,
                                             preferred_height + em)
        ###############################
        # plan
        preferred_height = self._plan_instuct.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height

        self._plan_instuct.frame = gui.Rect(r.get_right() - prefered_width, tools_height + em * 1.1, prefered_width,
                                            preferred_height + em)
        ###############################
        # ref
        preferred_height = self._ref_instuct.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height - 10000 + 7.5 * em

        self._ref_instuct.frame = gui.Rect(r.get_right() - prefered_width, tools_height + em * 1.1, prefered_width,
                                           preferred_height + em)
        ###############################
        # noref
        preferred_height = self._no_ref_instuct.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height

        self._no_ref_instuct.frame = gui.Rect(r.get_right() - prefered_width, tools_height + em * 1.1, prefered_width,
                                              preferred_height + em * 2.7)
        ###############################
        # settings
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)

        if self._tools_panel.visible:
            self._settings_panel.frame = gui.Rect(r.get_right() - prefered_width, r.y + tools_height, prefered_width,
                                                  height)
        else:
            self._settings_panel.frame = gui.Rect(r.get_right() - prefered_width, r.y, prefered_width,
                                                  height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):

        def get_model_path(self):
            root = tk.Tk()
            root.withdraw()
            filename = fd.askopenfilename(title='Choose a file',
                                          filetypes=(("Triangle Mesh", "*.ply;*.stl;*.fbx;*.obj;*.off;*.gltf;*.glb"),
                                                     ("Point Cloud", "*.xyz;*.xyzn;*.xyzrgb;*.ply;*.pcd;*.pts"),
                                                     ("All files", "*.*")))
            root.update()
            root.destroy()
            if filename:
                self.load(filename)

        # Use thread for tk dialog. root window destroy in main seems to close app
        ts = thread(target=get_model_path, args=(self,))
        ts.start()

    def _on_menu_export(self):
        cam_matrix = self._scene.scene.camera.get_view_matrix()
        intrinsics_file = open("extrinsics.txt", "w")
        for row_idx in range(cam_matrix.shape[0]):
            for col_idx in range(cam_matrix.shape[1]):
                intrinsics_file.write(str(cam_matrix[row_idx][col_idx]))
                if col_idx < cam_matrix.shape[1] - 1:
                    intrinsics_file.write(' ')
            if row_idx < cam_matrix.shape[0] - 1:
                intrinsics_file.write('\n')
        intrinsics_file.close()

        cam_intrinsics = self.o3d_intrinsics.intrinsic_matrix
        intrinsics_file = open("intrinsics.txt", "w")
        for row_idx in range(cam_intrinsics.shape[0]):
            for col_idx in range(cam_intrinsics.shape[1]):
                intrinsics_file.write(str(cam_intrinsics[row_idx][col_idx]))
                if col_idx < cam_intrinsics.shape[1] - 1:
                    intrinsics_file.write(' ')
            if row_idx < cam_intrinsics.shape[0] - 1:
                intrinsics_file.write('\n')
        intrinsics_file.close()

        frame = self._scene.frame
        self.export_image("snapshot.png", frame.width, frame.height)

    def _on_menu_save_json(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".json", "JSON files (.json)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_save_json_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_json_dialog_done(self, filename):
        self.window.close_dialog()
        self.export_json(filename)

    def _on_menu_extract_plane(self):
        print("Extracting plane...")

    def _on_menu_save_plane(self):
        print("Saving plane annotation to JSON dict...")

    def _on_menu_extract_poly(self):
        print("Extracting polygon...")

    def _on_menu_save_poly(self):
        print("Saving geometry ROI annotation to JSON dict...")

    def _on_menu_extract_color(self):
        print("Extracting region texture...")

    def _on_menu_save_color(self):
        print("Saving texture ROI annotation to JSON dict...")

    def _on_menu_save_corres(self):
        print("Saving correspondences to JSON dict...")

    def _refine_view(self):

        original_photo = copy.deepcopy(self._current_2Dbackground)

        cam_matrix = self._scene.scene.camera.get_view_matrix()
        cam_matrix = np.linalg.inv(cam_matrix)

        cam_matrix[3,:] = [0, 0, 0, 1]

        width = original_photo.shape[1]
        height = original_photo.shape[0]
        K = np.zeros((3, 3))
        K[2, 2] = 1
        K[0, 0] = height/2
        K[1, 1] = height/2
        K[0, 2] = width/2
        K[1, 2] = height/2
        est_color, est_depth = sim_view_point(self.trimesh, cam_matrix,
                                              K,
                                              width, height)

        print("Finding matching features between photo and generated viewpoint...")
        query_img = cv2.cvtColor(original_photo[:, :, :3], cv2.COLOR_BGR2GRAY)
        train_img = cv2.cvtColor(est_color[:, :, :3], cv2.COLOR_BGR2GRAY)

        # cv2.imshow("query", query_img)
        # cv2.imshow("train", train_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # detector = cv2.ORB_create(edgeThreshold=31, patchSize=31, fastThreshold=1000)
        # detector = cv2.xfeatures2d.SURF_create(hessianThreshold=100)
        detector = cv2.SIFT_create(nOctaveLayers=6, contrastThreshold=0.01, edgeThreshold=30, sigma=0.8)
        query_keypoints, query_descriptors = detector.detectAndCompute(query_img, None)
        train_keypoints, train_descriptors = detector.detectAndCompute(train_img, None)

        # Matching descriptor vectors with a FLANN based matcher
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(query_descriptors, train_descriptors, 2)

        # Filter matches using the Lowe's ratio test
        ratio_thresh = 0.6
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        img_matches = np.empty(
            (max(query_img.shape[0], train_img.shape[0]), query_img.shape[1] + train_img.shape[1], 3),
            dtype=np.uint8)
        cv2.drawMatches(query_img, query_keypoints, cv2.cvtColor(train_img, cv2.COLOR_RGB2BGR),
                        train_keypoints, good_matches, img_matches,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.namedWindow('Good Matches', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Good Matches', cv2.WINDOW_FULLSCREEN, 1)
        cv2.imshow('Good Matches', img_matches)
        cv2.waitKey()
        cv2.destroyAllWindows()

        model_points = []
        photo_pixels = []
        # Take the pixels and use the depth image to get 3D coordinates
        for match in good_matches:
            query_pixel = (
                round(query_keypoints[match.queryIdx].pt[1]), round(query_keypoints[match.queryIdx].pt[0]))
            train_pixel = (
                round(train_keypoints[match.trainIdx].pt[1]), round(train_keypoints[match.trainIdx].pt[0]))

            y, x = train_pixel
            depth = est_depth[train_pixel]

            matrix = np.array(self._scene.scene.camera.get_view_matrix())
            matrix[1, :] = -matrix[1, :]
            matrix[2, :] = -matrix[2, :]
            matrix = np.linalg.inv(matrix)

            point_pos = unproject_point([x, y], depth, K, matrix)

            model_points.append(point_pos)
            # To use OpenCV's SolvePnP the col and row must be switched
            photo_pixels.append((query_pixel[1], query_pixel[0]))

        model_points = np.array(model_points).reshape(len(model_points), -1, 3)
        photo_pixels = np.array(photo_pixels).reshape(len(photo_pixels), -1, 2)

        est_cam_matrix = solve_pnp(model_points.astype(np.float32), photo_pixels.astype(np.float32), self.K)

        est_color, est_depth = sim_view_point(self.trimesh, est_cam_matrix,
                                              self.K,
                                              original_photo.shape[1], original_photo.shape[0])
        cv2.imwrite("est_color.png", cv2.cvtColor(est_color, cv2.COLOR_RGB2BGR))

        if est_cam_matrix is None:
            print("\n[ERROR] Could not estimate camera transformation.")
        else:
            est_cam_matrix = np.linalg.inv(est_cam_matrix)
            est_cam_matrix[1, :] = -est_cam_matrix[1, :]
            est_cam_matrix[2, :] = -est_cam_matrix[2, :]

            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(self.o3d_intrinsics, est_cam_matrix, bounds)
            self._new_val_extrinsics[self._new_val] = est_cam_matrix

            # Apply metrics... TODO: separate to other script
            # cv2.imwrite('/fullref_output/' + self._new_val + '_original_photo' + '.png',
            #             original_photo)
            # cv2.imwrite('/fullref_output/' + self._new_val + '_simulated_photo' + '.png',
            #             cv2.cvtColor(est_color, cv2.COLOR_RGBA2BGRA))
            #
            # # Elements are False where image is transparent
            # mask = np.ma.masked_not_equal(est_color[:, :, 3], 0).mask
            # # cv2.imwrite('mask.png', (mask*255).astype(np.uint8))
            # if not mask.any():
            #     mask = None
            #
            # compare_images(cv2.cvtColor(original_photo, cv2.COLOR_BGRA2RGBA), est_color, self._new_val, mask)

    def _on_menu_refine(self):
        print("\nRefining viewpoint...")

        gui.Application.instance.post_to_main_thread(self.window, self._refine_view)

        # ts = thread(target=self._refine_view)
        # ts.start()

    def _on_menu_import_image(self):

        def get_image_paths(self):
            root = tk.Tk()
            root.withdraw()
            files = fd.askopenfilenames(title='Choose a file',
                                        filetypes=(("Images", "*.jpg; *.png"), ("All files", "*.*")))
            root.update()
            root.destroy()
            if files:
                self.lv.set_items([os.path.basename(file) for file in files])
                self._photos_dir = os.path.dirname(files[0])
                self._new_val_extrinsics = {}

        # Use thread for tk dialog. root window destroy in main seems to close app
        ts = thread(target=get_image_paths, args=(self,))
        ts.start()

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)
        w = self.window
        w.set_needs_layout()

    def _on_menu_toggle_tools_panel(self):
        self._tools_panel.visible = not self._tools_panel.visible
        if not self._tools_panel.visible:
            self._ortho_instuct.visible = False
            self._plan_instuct.visible = False
            self._ref_instuct.visible = False
            self._no_ref_instuct.visible = False
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_TOOLS, self._tools_panel.visible)
        w = self.window
        w.set_needs_layout()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def draw_loading(self):
        tmp_logo = np.asarray(self.logo * (abs(self.i - 40) / (40)), dtype=np.uint8)
        tmp_logo = cv2.copyMakeBorder(tmp_logo, 0, 0, 700, 700, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        self._loading_image.update_image(o3d.geometry.Image(tmp_logo))
        self.i = (self.i + 1) % 80
        self.frame_ready = True

    def load(self, path):
        self._loading_image.visible = True
        w = self.window
        w.set_needs_layout()

        self.frame_ready = True

        def update_animation(self):
            self.i = 1
            while self._loading_image.visible:
                time.sleep(0.1)
                if self.frame_ready:
                    self.frame_ready = False
                    gui.Application.instance.post_to_main_thread(self.window, self.draw_loading)

        ts = thread(target=update_animation, args=(self,))
        ts.start()

        self._scene.scene.clear_geometry()

        geometry = None

        geometry_type = o3d.io.read_file_geometry_type(path)

        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            self.model = o3d.io.read_triangle_model(path)

        if self.model is not None:
            geometry = self.model
            for mesh in geometry.meshes:
                if len(mesh.mesh.triangles) == 0:
                    print(
                        "[WARNING] Contains 0 triangles, will read as point cloud")
                    mesh.mesh = None
                else:
                    mesh.mesh.compute_vertex_normals()
                    if len(mesh.mesh.vertex_colors) == 0:
                        mesh.mesh.paint_uniform_color([1, 1, 1])
                # Make sure the mesh has texture coordinates
                if not mesh.mesh.has_triangle_uvs():
                    uv = np.array([[0.0, 0.0]] * (3 * len(mesh.mesh.triangles)))
                    mesh.mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)

                m = geometry.materials[mesh.material_idx]
                m.shader = "defaultUnlit"
                m.base_color = [1.0, 1.0, 1.0, 1.0]
                try:
                    self._scene.scene.add_geometry(mesh.mesh_name, mesh.mesh, m)
                except Exception as e:
                    print(e)
        else:
            print("[Info]", path, "appears to be a point cloud")

            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
                try:
                    self._scene.scene.add_geometry("__model__", geometry,
                                                   self.settings.material)
                except Exception as e:
                    print(e)
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            # Setup camera
            # bounds = self._scene.scene.bounding_box
            # self._scene.setup_camera(60, bounds, bounds.get_center())

            frame = self._scene.frame
            fovy = math.radians(60)
            f = frame.height / (2 * math.tan(fovy / 2))
            cx = frame.width / 2
            cy = frame.height / 2
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(frame.width, frame.height, f, f, cx, cy)

            o3d_extrinsics = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(self.o3d_intrinsics, o3d_extrinsics, bounds)
            self.K = np.zeros((3, 3))
            self.K[2, 2] = 1
            self.K[0, 0] = f
            self.K[1, 1] = f
            self.K[0, 2] = cx
            self.K[1, 2] = cy
            self._scene.scene.camera.set_projection(self.K, 0.2, 1000, frame.width, frame.height)

        self.trimesh = trimesh.load(path)

        self._loading_image.visible = False
        w = self.window
        w.set_needs_layout()

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def export_json(self, path):

        with open(path, 'w') as outfile:
            json.dump(AppWindow.JSON_DICT, outfile, indent=4)

    def _on_tree(self, new_item_id):
        if new_item_id == 2:
            self._ortho_instuct.visible = True
            self._plan_instuct.visible = False
            self._ref_instuct.visible = False
            self._image2D.visible = False
            self._no_ref_instuct.visible = False
        if new_item_id == 3:
            self._ortho_instuct.visible = False
            self._plan_instuct.visible = True
            self._ref_instuct.visible = False
            self._image2D.visible = False
            self._no_ref_instuct.visible = False
        if new_item_id == 5:
            self._ortho_instuct.visible = False
            self._plan_instuct.visible = False
            self._ref_instuct.visible = True
            self._image2D.visible = True
            self._no_ref_instuct.visible = False
        if new_item_id == 6:
            self._ortho_instuct.visible = False
            self._plan_instuct.visible = False
            self._ref_instuct.visible = False
            self._image2D.visible = False
            self._no_ref_instuct.visible = True

        w = self.window
        w.set_needs_layout()

    def _on_list(self, new_val, is_dbl_click):
        # background hack to pick2d points
        self._new_val = new_val
        self._current_2Dbackground = cv2.cvtColor(cv2.imread(os.path.join(self._photos_dir, new_val)),
                                                  cv2.COLOR_BGR2RGB)
        self._image2D.scene.set_background([45 / 255, 45 / 255, 45 / 255, 1.0],
                                           o3d.geometry.Image(self._current_2Dbackground))
        self._image2D.set_on_mouse(self._on_mouse_image2d)

        # # get camera intrinsics for reference photos from txt file or use default
        # intrinsics_path = os.path.join(self._photos_dir, 'intrinsics.txt')
        # if os.path.exists(intrinsics_path):
        #     with open(intrinsics_path, 'r') as f:
        #         self.K = np.array([[float(num) for num in line.split()] for line in f])[:3, :3]
            
        #     self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        #         width=self._current_2Dbackground.shape[1],
        #         height=self._current_2Dbackground.shape[0],
        #         fx=self.K[0, 0], fy=self.K[1, 1],
        #         cx=self.K[0, 2], cy=self.K[1, 2]
        #     )


        # extrinsics
        if new_val in self._new_val_extrinsics:
            cam_matrix = self._new_val_extrinsics[new_val]
        else:
            cam_matrix = self._scene.scene.camera.get_view_matrix()
            cam_matrix[1, :] = -cam_matrix[1, :]
            cam_matrix[2, :] = -cam_matrix[2, :]

        # bounds = self._scene.scene.bounding_box
        # self._scene.setup_camera(self.o3d_intrinsics, cam_matrix, bounds)

        self.window.set_needs_layout()

        # print(self.lv.selected_value)

    def _on_key_widget3d(self, event):

        if event.key == gui.KeyName.SPACE:
            if event.type == gui.KeyEvent.UP:

                def mask_callback(depth_image):

                    # reset point colors
                    self.color_idx = 0

                    if self.selecting_polygon:
                        self.selecting_polygon = False

                        if len(self.polygon_vertices) >= 3:

                            depth = np.asarray(depth_image)

                            points = np.array(self.polygon_vertices_2D, np.int32)
                            points = points.reshape((-1, 1, 2))

                            mask = np.zeros((self._scene.frame.height, self._scene.frame.width), np.uint8)
                            mask = cv2.fillPoly(mask, [points], 1)

                            masked_depth = cv2.bitwise_and(depth, depth, mask=mask)

                            cv2.imshow("depth", masked_depth)
                            cv2.waitKey(0)

                        else:
                            print("Not enough vertices have been selected!")

                    else:
                        self.selecting_polygon = True
                        self.polygon_vertices = []
                        self.polygon_vertices_2D = []
                        for label in self.labels:
                            self._scene.remove_3d_label(label)
                        self._scene.scene.clear_geometry()
                        for mi in self.model.meshes:
                            m = self.model.materials[mi.material_idx]
                            m.shader = "defaultUnlit"
                            m.base_color = [1.0, 1.0, 1.0, 1.0]
                            self._scene.scene.add_geometry(mi.mesh_name, mi.mesh, m)
                        print("\nPlease select the polygon vertices")

                if self._tree.selected_item == 6:
                    self._scene.scene.scene.render_to_image(mask_callback)
                else:
                    self._scene.scene.scene.render_to_depth_image(mask_callback)

                return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse_image2d(self, event):

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):

            tmp_img = copy.deepcopy(self._current_2Dbackground)

            if gui.MouseEvent.is_button_down(event, gui.MouseButton.LEFT):
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y

                max_y, max_x, _ = tmp_img.shape

                img_ratio = max_x / max_y

                # If the window aspect ratio changes, keep using only the space of the image
                frame_height = min(self._scene.frame.height, self._scene.frame.width * (1 / img_ratio))
                frame_width = min(self._scene.frame.width, self._scene.frame.height * img_ratio)

                # X is reversed for some reason in the window
                x = (self._scene.frame.width + x - abs(self._scene.frame.width - frame_width) / 2) * max_x / frame_width
                y = (y - abs(self._scene.frame.height - frame_height) / 2) * max_y / frame_height

                # If (x, y) falls within the image
                if 0 <= x < max_x and 0 <= y < max_y:
                    self.picked_points_2D.append((int(x), int(y)))
                    print("[INFO] Picked 2D point (" + str(x) + ", " + str(y) + ") to add in queue.")

            if gui.MouseEvent.is_button_down(event, gui.MouseButton.RIGHT):
                if len(self.picked_points_2D) > 0:
                    x, y = self.picked_points_2D.pop()
                    print("[INFO] Removed point (" + str(x) + ", " + str(y) + ") from pick queue.")

            if len(self.picked_points_2D) > 0:
                for idx, point in enumerate(self.picked_points_2D):
                    color = np.array(plt.cm.tab10.colors[idx % 10][:3]) * 1.6 * 255 
                    cv2.circle(tmp_img, point, 15, color, -1)

            self._image2D.scene.set_background([45 / 255, 45 / 255, 45 / 255, 1.0],
                                               o3d.geometry.Image(tmp_img))

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse_widget3d(self, event):

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y

                depth = np.asarray(depth_image)[y, x]

                if depth != 1.0:  # clicked on nothing (i.e. the far plane)

                    frame = self._scene.frame
                    fovy = math.radians(60)
                    f = frame.height / (2 * math.tan(fovy / 2))
                    cx = frame.width / 2
                    cy = frame.height / 2

                    self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(frame.width, frame.height, f, f, cx, cy)

                    matrix = np.array(self._scene.scene.camera.get_view_matrix())
                    matrix[1, :] = -matrix[1, :]
                    matrix[2, :] = -matrix[2, :]
                    matrix = np.linalg.inv(matrix)

                    z_near = self._scene.scene.camera.get_near()
                    z_far = self._scene.scene.camera.get_far()
                    depth = 2.0 * z_near * z_far / (z_far + z_near - (2.0 * depth - 1.0) * (z_far - z_near))

                    point_pos = unproject_point([x, y], depth, self.o3d_intrinsics.intrinsic_matrix, matrix)

                    # distance, vertex_id = mesh.nearest.vertex([depth])
                    # point_pos = mesh.vertices[vertex_id][0]

                    tmp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self._scene.scene.bounding_box.get_max_extent()/150)
                    tmp_sphere.compute_triangle_normals()
                    tmp_sphere.paint_uniform_color(self.colors[self.color_idx][:3])
                    self.color_idx = (self.color_idx + 1) % len(self.colors)
                    tmp_sphere.translate(point_pos)
                    self._scene.scene.add_geometry(str(point_pos), tmp_sphere, self.settings.material)

                    if self.selecting_polygon:
                        text = "({:.3f}, {:.3f}, {:.3f})".format(
                            point_pos[0], point_pos[1], point_pos[2])
                        print("[Open3D INFO] Added point " + text + " to polygon")
                        self.polygon_vertices.append(point_pos)
                        self.polygon_vertices_2D.append((x, y))
                        idx = len(self.polygon_vertices) - 1
                        self.labels.append(self._scene.add_3d_label(point_pos, str(idx)))
                        if len(self.polygon_vertices) > 1:
                            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(
                                [point_pos, self.polygon_vertices[idx - 1]]),
                                lines=o3d.utility.Vector2iVector([[0, 1]]))
                            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
                            self._scene.scene.add_geometry("line_" + str(idx), line_set, self.settings.material)

            self._scene.scene.scene.render_to_depth_image(depth_callback)

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
