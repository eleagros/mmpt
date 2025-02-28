import unittest
import subprocess
import processingmm
import os

class TestProcessingMM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test environment, including log file path."""
        cls.path_test_folder = f"{processingmm.__file__.split('src/processingmm')[0]}tests"
        cls.log_file = f"{cls.path_test_folder}/logfile_PDDN.log"

        # Remove the log file if it already exists
        if os.path.exists(cls.log_file):
            os.remove(cls.log_file)

        print(f"Log file: {cls.log_file}")

    def test_dependencies(self):
        """Test if dependencies are listed correctly."""
        from processingmm import libmpMuelMat
        libmpMuelMat.list_Dependencies()

    def test_pddn_models_existence(self):
        """Check if required PDDN models exist."""
        path_models = f"{processingmm.__file__.split('__init__')[0]}PDDN_model/"
        models = ['PDDN_model_550_Fresh_HB.pt', 'PDDN_model_600_Fresh_HB.pt']
        
        missing_models = []
        for model in models:
            if not os.path.exists(path_models + model):
                missing_models.append(model)

        self.assertFalse(missing_models, f"Missing models: {missing_models}")

    def run_subprocess(self, command):
        """Helper function to run subprocess and check its success."""
        with open(self.log_file, 'a') as log:
            result = subprocess.run(command, stdout=log, stderr=log)
            self.assertEqual(result.returncode, 0, f"Command {command} failed with exit code {result.returncode}")

    def test_processing_mm_with_pddn(self):
        """Test running process_MM.py with 'pddn' argument."""
        self.run_subprocess(['python', f"{self.path_test_folder}/process_MM.py", 'pddn'])

if __name__ == '__main__':
    unittest.main()
