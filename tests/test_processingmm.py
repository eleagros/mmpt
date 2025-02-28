import unittest
import subprocess

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
    
    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(subtract(0, 1), -1)
        self.assertEqual(subtract(10, 10), 0)
    
    def test_script_runs(self):
        result = subprocess.run(["python", "test_script.py"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")

if __name__ == "__main__":
    unittest.main()
