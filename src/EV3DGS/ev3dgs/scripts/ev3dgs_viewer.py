import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchvision

import viser
import viser.transforms as vtf

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.viewer.utils import CameraState
from nerfstudio.utils.colormaps import ColormapOptions

from nerfstudio.viewer.viewer_elements import ViewerCheckbox, ViewerSlider
from nerfstudio.viewer_legacy.server import viewer_utils
from nerfstudio.viewer.render_state_machine import RenderStateMachine, RenderAction

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ev3dgs.utils.ev3dgs_render_panel import populate_render_tab


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0

class EV3DGSControlPanel:
    """Minimal control panel with only EV3DGS controls."""
    
    def __init__(self, viser_server, rerender_cb):
        self.server = viser_server

        self._time_enabled = False
        
        with self.server.gui.add_folder("Render Options"):
            self._max_res = ViewerSlider(
                "Max res",
                512,
                64,
                2048,
                100,
                cb_hook=lambda _: rerender_cb(),
                hint="Maximum resolution to render in viewport",
            )
            self._max_res.install(self.server)
        
        with self.server.gui.add_folder("EV3DGS Filtering"):
            self._two_pass = ViewerCheckbox(
                name="Two-Pass Filter",
                default_value=False,
                cb_hook=lambda _: rerender_cb(),
                hint="Enable two-pass filtering",
            )
            self._two_pass.install(self.server)
            
            self._xg_filter = ViewerSlider(
                name="xg threshold",
                default_value=0.01,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                cb_hook=lambda _: rerender_cb(),
                hint="Filtering threshold for uncertain areas.",
            )
            self._xg_filter.install(self.server)
            
            self._cc_filter = ViewerSlider(
                name="gaussian/pixel ratio",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                cb_hook=lambda _: rerender_cb(),
                hint="Filtering threshold for Gaussians per pixel.",
            )
            self._cc_filter.install(self.server)
    
    # ========== Properties for created controls ==========
    @property
    def max_res(self) -> int:
        return self._max_res.value
    
    @property
    def two_pass(self) -> bool:
        return self._two_pass.value
    
    @property
    def xg_filter(self) -> float:
        return self._xg_filter.value
    
    @property
    def cc_filter(self) -> float:
        return self._cc_filter.value
    
    # ========== Fixed values for default controls ==========
    # (No GUI elements, just return the values the viewer expects)
    
    @property
    def output_render(self) -> str:
        return "rgb"  # Always rgb, no dropdown needed
    
    @property
    def layer_depth(self) -> bool:
        return False  # Always False, no checkbox needed
    
    @property
    def split(self) -> bool:
        return False
    
    @property
    def split_percentage(self) -> float:
        return 0.5
    
    @property
    def split_output_render(self) -> str:
        return "rgb"
    
    @property
    def crop_viewport(self) -> bool:
        return False
    
    @property
    def crop_obb(self):
        return None
    
    @property
    def background_color(self):
        return (38, 42, 55)
    
    @property
    def time(self) -> float:
        return 0.0
    
    @property
    def colormap_options(self):
        return ColormapOptions(colormap="default", normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
    
    @property
    def split_colormap_options(self):
        return ColormapOptions(colormap="default", normalize=True, colormap_min=0.0, colormap_max=1.0, invert=False)
    
    # ========== Stub methods that may be called ==========
    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        pass
    
    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        pass
    
    def update_output_options(self, new_options):
        pass
    
    def update_control_panel(self) -> None:
        pass
    
    

class EV3DGSViewer:
    """Standalone viewer for EV3DGS with minimal controls."""
    
    def __init__(
        self,
        config,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        trainer=None,
        train_lock=None,
        share: bool = False,
    ):
        
        self.ready = False
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        self.include_time = self.pipeline.datamanager.includes_time

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state = "completed"
        self._prev_train_state = "completed"
        self.last_move_time = 0

        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port)
        
        share_url = None
        if share:
            share_url = self.viser_server.request_share_url()

        if share_url is not None:
            self.viewer_info = [f"Viewer at: http://localhost:{websocket_port} or {share_url}"]
        elif config.websocket_host == "0.0.0.0":
            self.viewer_info = [f"Viewer running locally at: http://localhost:{websocket_port} (listening on 0.0.0.0)"]
        else:
            self.viewer_info = [f"Viewer running locally at: http://{config.websocket_host}:{websocket_port}"]

        self.viser_server.configure_theme(
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        self.render_statemachines: Dict[int, RenderStateMachine] = {}
        self.viser_server.on_client_disconnect(self.handle_disconnect)
        self.viser_server.on_client_connect(self.handle_new_client)

        # Camera visibility buttons
        self.hide_images = self.viser_server.gui.add_button(
            label="Hide Train Cams", disabled=False, icon=viser.Icon.EYE_OFF
        )
        self.hide_images.on_click(lambda _: self.set_camera_visibility(False))
        self.hide_images.on_click(lambda _: self.toggle_cameravis_button())
        
        self.show_images = self.viser_server.gui.add_button(
            label="Show Train Cams", disabled=False, icon=viser.Icon.EYE
        )
        self.show_images.on_click(lambda _: self.set_camera_visibility(True))
        self.show_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images.visible = False

        mkdown = self.make_stats_markdown(0, "0x0px")
        self.stats_markdown = self.viser_server.gui.add_markdown(mkdown)

        # CREATE TAB GROUP
        tabs = self.viser_server.gui.add_tab_group()

        # EV3DGS CONTROL PANEL
        control_tab = tabs.add_tab("Control", viser.Icon.SETTINGS)
        with control_tab:
            self.control_panel = EV3DGSControlPanel(
                self.viser_server,
                self._trigger_rerender,
            )

        # Render tab (from nerfstudio)
        config_path = self.log_filename.parents[0] / "config.yml"
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                self.viser_server, config_path, self.datapath, self.control_panel
            )

        self.include_time = False
        self.ready = True

    # === Required methods (copied from Viewer) ===
    
    def toggle_cameravis_button(self) -> None:
        self.hide_images.visible = not self.hide_images.visible
        self.show_images.visible = not self.show_images.visible

    def make_stats_markdown(self, step: Optional[int], res: Optional[str]) -> str:
        return f"Step: {step}  \nResolution: {res}"

    def update_step(self, step):
        self.stats_markdown.content = self.make_stats_markdown(step, None)

    def get_camera_state(self, client: viser.ClientHandle):
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(client.camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        return CameraState(
            fov=client.camera.fov,
            aspect=client.camera.aspect,
            c2w=c2w,
            camera_type=CameraType.PERSPECTIVE,
        )

    def handle_disconnect(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id].running = False
        self.render_statemachines.pop(client.client_id)

    def handle_new_client(self, client: viser.ClientHandle) -> None:
        self.render_statemachines[client.client_id] = RenderStateMachine(self, VISER_NERFSTUDIO_SCALE_RATIO, client)
        self.render_statemachines[client.client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            if not self.ready:
                return
            self.last_move_time = time.time()
            with self.viser_server.atomic():
                camera_state = self.get_camera_state(client)
                self.render_statemachines[client.client_id].action(RenderAction("move", camera_state))

    def set_camera_visibility(self, visible: bool) -> None:
        with self.viser_server.atomic():
            for idx in self.camera_handles:
                self.camera_handles[idx].visible = visible

    def _trigger_rerender(self) -> None:
        if not self.ready:
            return
        clients = self.viser_server.get_clients()
        for id in clients:
            camera_state = self.get_camera_state(clients[id])
            self.render_statemachines[id].action(RenderAction("move", camera_state))

    def init_scene(self, train_dataset, train_state, eval_dataset=None):
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}
        
        num_images = len(train_dataset)
        if self.config.max_num_display_images > 0:
            num_images = min(num_images, self.config.max_num_display_images)
        image_indices = np.linspace(0, len(train_dataset) - 1, num_images, dtype=np.int32).tolist()
        
        for idx in image_indices:
            image = train_dataset[idx]["image"]
            camera = train_dataset.cameras[idx]
            image_uint8 = (image * 255).detach().type(torch.uint8)
            image_uint8 = image_uint8.permute(2, 0, 1)
            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias=None)
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            c2w = camera.camera_to_worlds.cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.scene.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan((camera.cx / camera.fx[0]).cpu())),
                scale=self.config.camera_frustum_scale,
                aspect=float((camera.cx[0] / camera.cy[0]).cpu()),
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
            )
            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2w

        self.train_state = train_state
        self.train_util = 0.9

    def update_scene(self, step: int, num_rays_per_batch=None):
        self.step = step

    def get_model(self):
        return self.pipeline.model
    
    # === EV3DGS properties (delegate to control_panel) ===
    @property
    def two_pass(self) -> bool:
        return self.control_panel.two_pass
    
    @property
    def xg_filter(self) -> float:
        return self.control_panel.xg_filter
    
    @property
    def cc_filter(self) -> float:
        return self.control_panel.cc_filter
    
    # ========== Stub methods that may be called ==========
    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        pass
    
    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        pass
    
    def update_output_options(self, new_options):
        pass
    
    def update_control_panel(self) -> None:
        pass