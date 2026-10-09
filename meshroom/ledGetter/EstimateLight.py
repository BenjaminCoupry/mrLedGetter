"""
"""


__version__ = "1.0"


import os
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class EstimateLight(desc.Node):
    """
    """


    category = "Photometric Stereo"


    documentation = """
    """

    inputs = [
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level for logging.",
            values=VERBOSE_LEVEL,
            value="info",
            exclusive=True,
        ),
        
        desc.File(
            name="inputScenario",
            label="Input Scenario",
            description="Path to the MVPS scenario.",
            value="",
        ),
        
        
        desc.IntParam(
            name="decimation",
            label="Image Decimation",
            description="Decimation of input dimension",
            value=10,
            range=(1, 30, 1),
        ),

        
        desc.IntParam(
            name="iterations",
            label="Iterations",
            description="Iterations for light estimation",
            value=10000,
            range=(1000, 30000, 100),
        ),
        
    
        desc.ChoiceParam(
            name="backend",
            label="Backend",
            description="Backend for heavy computations",
            values=["cpu", "gpu"],
            value="gpu",
            exclusive=True,
        ),

        desc.FloatParam(
            name="learningRate",
            label="Learning Rate",
            description="Learning rate",
            value=0.001,
            range=(0.001, 0.1, 0.001),
            advanced=True
        ),
        
        desc.FloatParam(
            name="delta",
            label="Huber Loss Delta",
            description="Robustness",
            value=0.001,
            range=(0.01, 0.1, 0.01),
            advanced=True
        ),
        
    ]

    outputs = [
        desc.File(
            name="outPath",
            label="Output Folder",
            description="Path to the processed output",
            value="{nodeCacheFolder}/output",
        ),
    ]

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            
            import sys
            import json
            import shlex
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            

            chunk.logger.info(f"Executable: {sys.executable}")
            chunk.logger.info(f"Python: {sys.version}" )
            
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            chunk.logger.info(f"Project: {project_dir}" )
            
            scenario_path = chunk.node.inputScenario.value
            with open(scenario_path, 'r') as f:
                scenario = json.load(f)
            chunk.logger.info(f"Found scenario {scenario_path}.")
            
            for pose_id in scenario:
                chunk.logger.info(f"Step {pose_id}.")
                
                
                lights = scenario[pose_id]['lights']
                pose_path = scenario[pose_id]['pose_path']
                geometry_path = scenario[pose_id]['geometry_path']
                out_path = os.path.join(chunk.node.outPath.value, pose_id)
                           
                learning_rate = chunk.node.learningRate.value
                delta = chunk.node.delta.value
                decimation = chunk.node.decimation.value
                iterations = chunk.node.iterations.value
                backend = chunk.node.backend.value
                
                #env_string = '"' + str(os.path.join(project_dir, "venv_compute", "bin", "python")) + '"'

                #launch_string = '"' + str(os.path.join(project_dir, "LED-getter", "estimate.py")) + '"'

                pinhole_images = False
                images_args = ['--ps_images_path']
                for light_name in lights:
                    images_args.append(lights[light_name]['images_paths'][0])
                    pinhole_images = pinhole_images or lights[light_name]['pinhole_images']

                scenario_args = ['--pose_path', pose_path, '--geometry_path', geometry_path, '--out_path', out_path]
                if pinhole_images:
                    scenario_args.append('--not_apply_geometry_undisto')

                node_args = ['--learning_rate', learning_rate, '--delta', delta, '--step', decimation, '--iterations', iterations, '--backend', backend]

                default_args = ['--parent_image_name', 0, '--tqdm_refresh', int(iterations / 10), '--pattern', 'directional', 'punctual', 'LED', '--slice_i', 0]

                # argv items must be strings, no quoting needed since no shell is involved
                args = [str(a) for a in images_args + scenario_args + node_args + default_args]

                chunk.logger.info(f"Light estimation. {shlex.join(args)}")

                import estimate
                sys.argv = ['estimate.py'] + args
                estimate.main()

                # # Meshroom's python variables would break the venv_compute python
                # env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "LD_LIBRARY_PATH")}
                
                # with subprocess.Popen(shlex.split(command_string), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as process:
                #     for line in process.stdout:  # type: ignore
                #         chunk.logger.info(line.rstrip())
                # if process.returncode != 0:
                #     raise subprocess.CalledProcessError(process.returncode, command_string)

            chunk.logger.info("Done.")
        finally:

            chunk.logManager.end()
