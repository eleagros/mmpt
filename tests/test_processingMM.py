import unittest
import subprocess
import processingmm
import os

class TestProcessingMM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test environment, including log file path."""
        cls.path_test_folder = f"{processingmm.__file__.split('src/processingmm')[0]}tests"
        cls.log_file = f"{cls.path_test_folder}/logfile.log"

        # Remove the log file if it already exists
        if os.path.exists(cls.log_file):
            os.remove(cls.log_file)

        print(f"Log file: {cls.log_file}")

    def test_dependencies(self):
        """Test if dependencies are listed correctly."""
        from processingmm import libmpMuelMat
        libmpMuelMat.list_Dependencies()
        
    def run_subprocess(self, command):
        """Helper function to run subprocess and check its success."""
        with open(self.log_file, 'a') as log:
            result = subprocess.run(command, stdout=log, stderr=log)
            self.assertEqual(result.returncode, 0, f"Command {command} failed with exit code {result.returncode}")

    def test_processing_mm_without_pddn(self):
        """Test running process_MM.py with 'no' argument."""
        self.run_subprocess(['python', f"{self.path_test_folder}/process_MM.py", 'no'])

    def test_align_wavelengths(self):
        """Test running align_wls.py."""
        self.run_subprocess(['python', f"{self.path_test_folder}/align_wls.py"])

if __name__ == '__main__':
    unittest.main()
