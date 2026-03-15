import lit.formats
import os

config.name = "AdaptiveMatmul-Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']

# Inject the current build directory into the PATH so lit can find 'adaptive-opt'
config.environment['PATH'] = os.getcwd() + os.pathsep + os.environ.get('PATH', '')
