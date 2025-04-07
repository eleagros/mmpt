import unittest
import subprocess
import mmpt
import os
import importlib.util
import torch

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
    
    def test_torchvision_installed(self):
        """Check if torchvision is installed."""
        torchvision_spec = importlib.util.find_spec("torchvision")
        self.assertIsNotNone(torchvision_spec, "torchvision is not installed")
    
    def test_torch_with_cuda(self):
        """Check if torch is installed and CUDA is available."""
        self.assertTrue(torch.cuda.is_available(), "Torch is installed but CUDA is not available")

    def test_dependencies(self):
        """Test if dependencies are listed correctly."""
        from mmpt import libmpMuelMat
        libmpMuelMat.list_Dependencies()

    def test_pddn_models_existence(self):
        """Check if required PDDN models exist."""
        path_models = f"{mmpt.__file__.split('__init__')[0]}PDDN_model/"
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
        self.run_subprocess(['python', f"{self.path_test_folder}/process_MM.py", 'pddn', 'IMP'])
        
    # def test_processing_mm_with_pddn(self):
        # """Test running process_MM.py with 'pddn' argument."""
        # self.run_subprocess(['python', f"{self.path_test_folder}/process_MM.py", 'pddn', 'IMPv2'])

if __name__ == '__main__':
    unittest.main()