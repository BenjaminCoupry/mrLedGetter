"""Node descriptions of the LED-getter plugin: metadata, attributes, ranges (CPU only)."""
import inspect
import re

import pytest

import helpers

from meshroom.core import desc

NODES = [("LedGetterLights", "LedGetterLights")]
METHOD = ("Self-calibrated Near-light Photometric Stereo using a Geometric Proxy (B. Coupry, F. Lauze, A. Laurent, "
          "J. Mélou, Y. Quéau, J.-D. Durou, SSVM 2025)")


def _node(moduleName, className):
    module = helpers.loadModule(moduleName)
    return module, getattr(module, className)


@pytest.mark.parametrize("moduleName,className", NODES)
def test_node_metadata(moduleName, className):
    """Author, license and version are shown in the node info by Meshroom (desc.Node.getNodeInfo)."""
    module, node = _node(moduleName, className)
    assert module.__author__ == "Benjamin Coupry"
    assert module.__license__ == "MIT"
    assert re.match(r"^\d+\.\d+$", module.__version__)
    info = dict(node.__nodeInfo__)
    assert info["method"] == METHOD
    assert info["methodLicense"] == "MIT"
    assert info["methodRepository"] == "https://github.com/BenjaminCoupry/LED-getter"


@pytest.mark.parametrize("moduleName,className", NODES)
def test_node_settings(moduleName, className):
    module, node = _node(moduleName, className)
    assert node.category == "Photometric Stereo"
    assert node.gpu == desc.Level.INTENSIVE
    assert "linear" in node.documentation.lower() and "outputFileType=exr" in node.documentation
    # the deprecated "group" parameter of the attributes is replaced by "commandLineGroup"
    assert not re.search(r"(?<!commandLine)[gG]roup\s*=", inspect.getsource(module))


@pytest.mark.parametrize("moduleName,className", NODES)
def test_node_attribute_defaults_in_range(moduleName, className):
    _, node = _node(moduleName, className)
    names = [attribute.name for attribute in node.inputs + node.outputs]
    assert len(names) == len(set(names))
    for attribute in node.inputs:
        if isinstance(attribute, (desc.IntParam, desc.FloatParam)) and attribute.range:
            low, high, _ = attribute.range
            assert low <= attribute.value <= high, attribute.name


def test_lights_node_attributes():
    _, node = _node("LedGetterLights", "LedGetterLights")
    inputs = {attribute.name: attribute for attribute in node.inputs}
    for name in ("inputSfm", "inputMesh", "meshAliceVisionAxes", "maskFolder", "downscale", "nbImages",
                 "samplingStep", "iterations", "learningRate", "delta", "exportImages", "minViewsPerPose",
                 "excludePoseReferenceImage", "imageSelection", "seed", "linearizeInput", "maskThreshold",
                 "maskVoteThreshold", "maskPoseReferenceOnly", "maskRemoveBorderComponents", "maskUseGlobalFile",
                 "failurePolicy", "useGpu", "verboseLevel"):
        assert name in inputs, name
    for name in ("outputFormat", "normalConvention", "keepLandmarks"):  # no map output
        assert name not in inputs
    assert inputs["delta"].value == pytest.approx(0.01)  # LED-getter default
    assert inputs["learningRate"].value == pytest.approx(0.001)
    assert inputs["iterations"].value == 10000 and inputs["samplingStep"].value == 10
    assert inputs["meshAliceVisionAxes"].value is True
    assert [attribute.name for attribute in node.outputs] == ["outputFolder", "outputLights"]
