#!/usr/bin/env python
import unittest
import sys
import os

# Add the parent directory to the path so we can import the tts package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import all test modules
from tts.tests.test_base import TestBaseModel
from tts.tests.test_interpolate import TestInterpolate
from tts.tests.test_kokoro_model import TestSanitizeLSTMWeights, TestKokoroModel
from tts.tests.test_kokoro_pipeline import TestKokoroPipeline
from tts.tests.test_models import TestModels

if __name__ == "__main__":
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestBaseModel))
    test_suite.addTest(unittest.makeSuite(TestInterpolate))
    test_suite.addTest(unittest.makeSuite(TestSanitizeLSTMWeights))
    test_suite.addTest(unittest.makeSuite(TestKokoroModel))
    test_suite.addTest(unittest.makeSuite(TestKokoroPipeline))
    test_suite.addTest(unittest.makeSuite(TestModels))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())