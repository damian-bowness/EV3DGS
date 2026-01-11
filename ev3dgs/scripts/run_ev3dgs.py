from dataclasses import dataclass, field, fields
from pathlib import Path
import tyro
import types
import time

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ev3dgs_viewer import EV3DGSViewer
from ev3dgs.utils.utils import get_render_output

def get_output_splatfacto_new(self, camera):
    ''' reimplementation of get_output function from models because of lack of proper interface to outputs dict'''

    outputs ={}
    raster_pkg = get_render_output(self, camera)
    outputs["rgb"] = raster_pkg.view(3,camera.height.item(),camera.width.item()).permute(1,2,0).clamp(max=1) # raster_pkg["rgb"] 

    return outputs

@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""
    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewerEV3DGS:
    """Load a checkpoint and start the viewer."""
    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""

    def main(self) -> None:
        """Main function."""
        from nerfstudio.utils.eval_utils import eval_setup
        
        config, pipeline, _, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        
        self.device = pipeline.device
        
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = "viewer"
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk
        
        self._start_viewer(config, pipeline, step)

    def _start_viewer(self, config: TrainerConfig, pipeline: Pipeline, step: int):
        """Starts the viewer."""
        base_dir = config.get_base_dir()
        viewer_log_path = base_dir / config.viewer.relative_log_filename
        
        model = pipeline.model
        
        # Setup EV3DGS-specific model attributes
        model.filter3D       = True
        model.two_pass       = False
        model.rastFlag       = True
        model.filter3D_scale = 0.2
        model.xg_thresh      = 0.01
        model.cc_thresh      = 0.5
        
        # Swap in custom get_outputs
        model.get_outputs = types.MethodType(get_output_splatfacto_new, model)
        
        # Create viewer
        viewer = EV3DGSViewer(
            config=config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            trainer=None,
            train_lock=None,
            share=False,
        )
        
        # Setup logging
        config.logging.local_writer.enable = False
        writer.setup_local_writer(
            config.logging, 
            max_iter=config.max_num_iterations, 
            banner_messages=viewer.viewer_info
        )
        
        assert pipeline.datamanager.train_dataset is not None
        
        # Initialize the scene
        viewer.init_scene(
            train_dataset=pipeline.datamanager.train_dataset,
            train_state="completed",
        )
        viewer.update_scene(step=step)
        
        # Main loop
        while True:
            time.sleep(0.01)
            pipeline.model.two_pass  = viewer.two_pass
            pipeline.model.xg_thresh = viewer.xg_filter
            pipeline.model.cc_thresh = viewer.cc_filter


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RunViewerEV3DGS).main()


if __name__ == "__main__":
    entrypoint()