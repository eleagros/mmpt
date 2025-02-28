import unittest
import subprocess
import os

class TestMathOperations(unittest.TestCase):
    
    def test_script_runs(self):
        script_path = os.path.join(os.path.dirname(__file__), "test_script.py")
        result = subprocess.run(["python", script_path, str(os.path.dirname(__file__))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")

if __name__ == "__main__":
    unittest.main()
