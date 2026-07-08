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
        "F2-R-MCP2": ("-0.176905", "0.35"),
        "F3-R-MCP2": ("-0.228696", "0.334765"),
        "F4-R-MCP2": ("-0.299815", "0.283408"),
        "F5-R-MCP2": ("-0.270761", "0.25782"),
    }


def test_custom_left_four_finger_mcp2_limits_match_quest_effective_search() -> None:
    limits = _joint_limits(
        Path("assets/custom_left/URDF_L.urdf"),
        ["F2-L-MCP2", "F3-L-MCP2", "F4-L-MCP2", "F5-L-MCP2"],
    )

    assert limits == {
        "F2-L-MCP2": ("-0.3", "0.344806"),
        "F3-L-MCP2": ("-0.166002", "0.124275"),
        "F4-L-MCP2": ("-0.3", "0.201738"),
        "F5-L-MCP2": ("-0.243689", "0.240955"),
    }
