from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def _joint_limits(path: Path, joint_names: list[str]) -> dict[str, tuple[str, str]]:
    root = ET.parse(path).getroot()
    out = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in joint_names:
            continue
        limit = joint.find("limit")
        assert limit is not None
        out[name] = (limit.attrib["lower"], limit.attrib["upper"])
    return out


def test_custom_right_four_finger_mcp2_limits_match_quest_effective_search() -> None:
    limits = _joint_limits(
        Path("assets/custom_right/URDF_R.urdf"),
        ["F2-R-MCP2", "F3-R-MCP2", "F4-R-MCP2", "F5-R-MCP2"],
    )

    assert limits == {
        "F2-R-MCP2": ("-0.380431", "0.087049"),
        "F3-R-MCP2": ("-0.242572", "0.204627"),
        "F4-R-MCP2": ("-0.498692", "0.477526"),
        "F5-R-MCP2": ("-0.440455", "0.61"),
    }


def test_custom_left_four_finger_mcp2_limits_match_quest_effective_search() -> None:
    limits = _joint_limits(
        Path("assets/custom_left/URDF_L.urdf"),
        ["F2-L-MCP2", "F3-L-MCP2", "F4-L-MCP2", "F5-L-MCP2"],
    )

    assert limits == {
        "F2-L-MCP2": ("-0.380431", "0.087049"),
        "F3-L-MCP2": ("-0.242572", "0.204627"),
        "F4-L-MCP2": ("-0.498692", "0.477526"),
        "F5-L-MCP2": ("-0.440455", "0.61"),
    }
