import sys
import types

# The paper code imports `tensorflow.python.training.training_util.global_step` without using it; the local CPU test
# env has no tensorflow (IAS does). Stub it so the module imports.
try:
    import tensorflow  # noqa: F401
except ImportError:
    for name in ("tensorflow", "tensorflow.python", "tensorflow.python.training", "tensorflow.python.training.training_util"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["tensorflow.python.training.training_util"].global_step = None
