import unittest
import subprocess
import mmpt
import os
import sys

class TestMMPT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test environment, including log file path."""
        base_path = os.path.dirname(os.path.abspath(mmpt.__file__))
        cls.path_test_folder = os.path.join(base_path, "..", "..", "tests")
        cls.log_file = os.path.join(cls.path_test_folder, "logfile.log")

        # Remove the log file if it already exists
        if os.path.exists(cls.log_file):
            os.remove(cls.log_file)

        print(f"Log file: {cls.log_file}")

    def test_dependencies(self):
        """Test if dependencies are listed correctly."""
        from mmpt import libmpMuelMat
        libmpMuelMat.list_Dependencies()
        
    def run_subprocess(self, command):
        """Helper function to run subprocess and check its success."""
        with open(self.log_file, 'a') as log:
            result = subprocess.run(command, stdout=log, stderr=log, shell=(os.name == "nt"))
            self.assertEqual(result.returncode, 0, f"Command {command} failed with exit code {result.returncode}")

    def test_align_wavelengths(self):
        """Test running align_wls.py."""
        script_path = os.path.join(self.path_test_folder, "align_wls.py")
        self.run_subprocess([sys.executable, script_path])

    def test_processing_mm_without_pddn(self):
        """Test running process_MM.py with 'no' argument."""
        self.run_subprocess(['python', f"{self.path_test_folder}/process_MM.py", 'no', 'IMP'])
        
    def test_processing_mm_impv2(self):
        """Test running process_MM.py with 'no' argument."""
        self.run_subprocess(['python', f"{self.path_test_folder}/process_MM.py", 'no', 'IMPv2'])
        
if __name__ == '__main__':
    unittest.main()