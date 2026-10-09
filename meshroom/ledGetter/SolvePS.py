"""
"""


__version__ = "1.0"


import os
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class SolvePS(desc.Node):
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
        
        desc.File(
            name="inputSfm",
            label="Input SfMData",
            description="SfMData JSON file with multi-lighting views "
                        "grouped by poseId.",
            value="",
        ),
        
        desc.File(
            name="inputLightFolder",
            label="Input Light Folder",
            description="Folder to the estimated light",
            value="",
        ),
        
        desc.IntParam(
            name="decimation",
            label="Image Decimation",
            description="Decimation of input dimension",
            value=2,
            range=(1, 5, 1),
        ),

        
        desc.IntParam(
            name="iterations",
            label="Iterations",
            description="Iterations for PS estimation",
            value=1000,
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
        
        desc.File(
            name="outputSfmDataNormal",
            label="SfMData Normal",
            description="Output SfMData file referencing normal maps.",
            value="{nodeCacheFolder}/normalMaps.sfm",
            group="",
        ),
        
        desc.File(
            name="outputSfmDataAlbedo",
            label="SfMData Albedo",
            description="Output SfMData file referencing albedo maps.",
            value="{nodeCacheFolder}/albedoMaps.sfm",
            group="",
        ),
        
    ]

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            
            import sys
            import json
            import shlex
            import copy
            import os
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
            
            output_sfm_normals_path = chunk.node.outputSfmDataNormal.value
            output_sfm_albedos_path = chunk.node.outputSfmDataAlbedo.value
            input_sfm_path = chunk.node.inputSfm.value
            with open(input_sfm_path, 'r') as f:
                input_sfm = json.load(f)
            
            output_sfm_normals = copy.deepcopy(input_sfm)
            output_sfm_albedos = copy.deepcopy(input_sfm)
            output_sfm_normals['views'] = []
            output_sfm_albedos['views'] = []
            
            for pose_id in scenario:
                chunk.logger.info(f"Step {pose_id}.")
                
                lights = scenario[pose_id]['lights']
                pose_path = scenario[pose_id]['pose_path']
                geometry_path = scenario[pose_id]['geometry_path']
                out_path = os.path.join(chunk.node.outPath.value, 'compute', pose_id)
                           
                learning_rate = chunk.node.learningRate.value
                delta = chunk.node.delta.value
                decimation = chunk.node.decimation.value
                iterations = chunk.node.iterations.value
                backend = chunk.node.backend.value
                
                pinhole_images = False
                images_args = ['--ps_images_path']
                for light_name in lights:
                    images_args.append(lights[light_name]['images_paths'][0])
                    pinhole_images = pinhole_images or lights[light_name]['pinhole_images']

                scenario_args = ['--pose_path', pose_path, '--geometry_path', geometry_path, '--out_path', out_path]
                if pinhole_images:
                    scenario_args.append('--not_apply_geometry_undisto')

                loaded_light_folder = os.path.join(chunk.node.inputLightFolder.value, pose_id, 'LED', 'slice_00000')

                node_args = ['--learning_rate', learning_rate, '--delta', delta, '--step', decimation, '--iterations', iterations, '--ps_backend', backend, '--loaded_light_folder', loaded_light_folder]

                default_args = ['--parent_image_name', 0, '--tqdm_refresh', int(iterations / 10), '--pattern', 'PS', '--slice_i', -1, '--ps_chunck_number', 100, '--backend', 'cpu', '--skip_export', 'images', 'lightmaps', 'light', 'misc']

                # argv items must be strings, no quoting needed since no shell is involved
                args = [str(a) for a in images_args + scenario_args + node_args + default_args]

                chunk.logger.info(f"Light estimation. {shlex.join(args)}")

                import estimate
                sys.argv = ['estimate.py'] + args
                estimate.main()

                
                src_albedos = os.path.join(chunk.node.outPath.value, 'compute', pose_id, 'full_slices_PS' ,'merged', 'values', 'images', 'albedo.png')
                dest_albedos = os.path.join(chunk.node.outPath.value, 'albedos', f'{pose_id}.png')
                src_normals = os.path.join(chunk.node.outPath.value, 'compute', pose_id, 'full_slices_PS' ,'merged', 'values', 'images', 'geometry_normals.png')
                dest_normals = os.path.join(chunk.node.outPath.value, 'normals', f'{pose_id}.png')

                for src, dest in ((src_albedos, dest_albedos), (src_normals, dest_normals)):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    if os.path.lexists(dest):
                        os.remove(dest)
                    os.symlink(src, dest)
                    chunk.logger.info(f"Linked {dest} -> {src}")

                view = dict()
                for v in input_sfm['views']:
                    if v['poseId'] == pose_id:
                        view = v
                
                view['viewId'] = pose_id
                view_normals = copy.deepcopy(view)
                view_albedos = copy.deepcopy(view)
                view_normals['path'] = src_normals
                view_albedos['path'] = src_albedos
                output_sfm_normals['views'].append(view_normals)
                output_sfm_albedos['views'].append(view_albedos)

            with open(output_sfm_normals_path, 'w') as f:
                json.dump(output_sfm_normals, f, indent=4)
            
            with open(output_sfm_albedos_path, 'w') as f:
                json.dump(output_sfm_albedos, f, indent=4)

            chunk.logger.info("Done.")
        finally:

            chunk.logManager.end()