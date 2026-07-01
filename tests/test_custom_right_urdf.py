import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


class CustomRightUrdfTest(unittest.TestCase):
    def test_pinky_aa_limit_is_expanded_for_opposition(self):
        root = ET.parse(Path("assets/custom_right/URDF_R.urdf")).getroot()
        joint = root.find("./joint[@name='F5-R-MCP2']")

        self.assertIsNotNone(joint)
        limit = joint.find("limit")
        self.assertIsNotNone(limit)
        self.assertEqual(float(limit.attrib["lower"]), -0.30)
        self.assertEqual(float(limit.attrib["upper"]), 0.35)


if __name__ == "__main__":
    unittest.main()
