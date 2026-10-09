"""
"""


__version__ = "1.0"


import os
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class PrepareMVPS(desc.Node):
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
            name="poses",
            label="Input Poses",
            description="Path to the input cameras.sfm.",
            value="",
        ),
        
        desc.File(
            name="inputMesh",
            label="Input Mesh",
            description="Path to the input mesh.",
            value="",
        ),
        
        desc.File(
            name="imagesFolder",
            label="Input Images Folder",
            description="Path to the input images folder.",
            value="",
        ),
        
        desc.BoolParam(
            name="identifyLights",
            label="Identify Lights",
            description="If enabled, the file names are assumed to be in the form <LIGHT_UID>-*.png. "
                        "Otherwise, all lights are considered different",
            value=True,
        ),
        
        desc.BoolParam(
            name="lightsInWorldFrame",
            label="Lights in World Frame",
            description="If enabled, lights are attached in the world frame. "
                        "Otherwise, lights are attached in the camera frame",
            value=True,
        ),
        
        desc.BoolParam(
            name="pinholeImages",
            label="Pinhole Images",
            description="If enabled, input images are supposed to be pinhole. "
                        "Otherwise, undisto parameters are to be considered",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        desc.File(
            name="outCameras",
            label="Output Cameras Folder",
            description="Path to the cameras files",
            value="{nodeCacheFolder}/output/cameras",
        ),
        desc.File(
            name="outScenario",
            label="Output Scenario",
            description="Path to the MVPS scenario",
            value="{nodeCacheFolder}/output/scenario.json",
        ),
    ]

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            
            import sys
            import json

            chunk.logger.info(f"Executable: {sys.executable}")
            chunk.logger.info(f"Python: {sys.version}" )
            
            def build_camera_dict(intrinsic, pose):
                width = float(intrinsic['width'])
                height = float(intrinsic['height'])
                sensor_width = float(intrinsic['sensorWidth'])
                sensor_height = float(intrinsic['sensorHeight'])
                focal_length = float(intrinsic['focalLength'])
                fx = focal_length * width / sensor_width
                fy = focal_length * height / sensor_height
                cx = width / 2 + float(intrinsic['principalPoint'][0])
                cy = height / 2 + float(intrinsic['principalPoint'][1])
                distorsion = [float(d) for d in intrinsic['distortionParams']]
                R = [[float(pose['pose']['transform']['rotation'][3 * i + j]) for j in range(3)] for i in range(3)]
                t = [float(c) for c in pose['pose']['transform']['center'] ]
                K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
                camera_dict = {'K':K, 'R':R, 't':t, 'width':width, 'height':height, 'distorsion' : distorsion}
                return camera_dict
            pinhole_images = chunk.node.pinholeImages.value
            input_cameras_path = chunk.node.poses.value
            with open(input_cameras_path, 'r') as f:
                cameras_dict = json.load(f)
            chunk.logger.info(f"Found cameras {input_cameras_path}.")
            
            poses = []
            for pose in cameras_dict['poses']:
                view_count = 0
                for view in cameras_dict['views']:
                    if view['poseId'] == pose['poseId']:
                        view_count += 1
                if view_count > 1:
                    poses.append(pose)
                    
            chunk.logger.info(f"Found {len(poses)} poses associated with multiple views.")

            geometry_path = chunk.node.inputMesh.value
            chunk.logger.info(f"Found geometry {geometry_path}.")
            os.makedirs(chunk.node.outCameras.value, exist_ok=True)
            scenario_path = chunk.node.outScenario.value
            scenario_dict = {}
            for pose in poses:
                pose_id = pose['poseId']
                chunk.logger.info(f"Pose {pose_id}.")
                
                views = []
                for view in cameras_dict['views']:
                    if view['poseId'] == pose_id:
                        views.append(view)
                chunk.logger.info(f"Found {len(views)} views for this pose.")
                
                
                intrinsic = None
                for intrinsic_i in cameras_dict['intrinsics']:
                    for view in views:
                        if intrinsic_i['intrinsicId'] == view['intrinsicId']:
                            intrinsic = intrinsic_i
                intrinsic_id = intrinsic['intrinsicId'] # type: ignore
        
                chunk.logger.info(f"Intrinsic {intrinsic_id}.")
                
                chunk.logger.info(f"Building camera file")
                pose_path = os.path.join(chunk.node.outCameras.value, f'{pose_id}.json')
                camera_dict = build_camera_dict(intrinsic, pose)
                
                images_folder = chunk.node.imagesFolder.value
                if images_folder != "":
                    chunk.logger.info(f"Developped images from folder {images_folder}.")
                else:
                    chunk.logger.info(f"Source images.")
                
                
                
                lights = {}
                world_frame = chunk.node.lightsInWorldFrame.value
                identify_lights = chunk.node.identifyLights.value
                for view in views:
                    view_id = view['viewId']
                    source_image_path = view['path']
                    if identify_lights:
                        light_name = str(os.path.basename(source_image_path)).split('-')[0]
                    else:
                        light_name = view_id
                    if images_folder != "":
                        image_path = os.path.join(images_folder, f'{view_id}.png')
                    else:
                        image_path = source_image_path
                    if light_name not in lights:
                        lights[light_name] = {'images_paths': [], 'world_frame' : world_frame, 'pinhole_images': pinhole_images}
                    lights[light_name]['images_paths'].append(image_path)
                    chunk.logger.info(f"Added {image_path} to light {light_name}.")
                
                scenario_dict[pose_id] = {'lights':lights, 'pose_path' : pose_path, 'geometry_path' : geometry_path}

                chunk.logger.info(f"Writing camera file at {pose_path}")
                with open(pose_path, 'w') as f:
                    json.dump(camera_dict, f, indent=4)
            
            chunk.logger.info(f"Writing scenario at {scenario_path}")
            with open(scenario_path, 'w') as f:
                json.dump(scenario_dict, f, indent=4)
            
            chunk.logger.info("Done.")
        finally:

            chunk.logManager.end()
