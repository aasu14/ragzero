"""Minimal stdlib test runner. Pytest isn't installed in this sandbox, so
this script imports each test module and runs every `test_*` function,
invoking conftest fixtures by hand. Production users will run `pytest` normally.
"""
import contextlib
import importlib
import inspect
import sys
import traceback
import types
from pathlib import Path


# Install a minimal pytest shim so test files' `import pytest` works.
def _install_pytest_shim():
    try:
        import pytest  # noqa: F401
        return
    except ImportError:
        pass

    @contextlib.contextmanager
    def _raises(exc_type):
        class _Info:
            value: BaseException | None = None
        info = _Info()
        try:
            yield info
        except exc_type as e:
            info.value = e
            return
        except BaseException as e:
            raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}")
        raise AssertionError(f"Expected {exc_type.__name__}, none raised")

    def _fixture(fn=None, **_):
        if fn is None:
            return lambda f: f
        return fn

    class _Approx:
        """Minimal pytest.approx replacement — float comparison with tolerance."""
        def __init__(self, value, rel=None, abs=None):
            self.value = value
            self.rel = rel if rel is not None else 1e-6
            self.abs = abs if abs is not None else 1e-12
        def __eq__(self, other):
            if isinstance(other, (list, tuple)):
                if not isinstance(self.value, (list, tuple)) or len(other) != len(self.value):
                    return False
                return all(_Approx(v, self.rel, self.abs) == o
                           for v, o in zip(self.value, other))
            try:
                diff = abs(float(other) - float(self.value))
                tol = max(self.abs, self.rel * abs(float(self.value)))
                return diff <= tol
            except (TypeError, ValueError):
                return False
        def __repr__(self):
            return f"approx({self.value})"

    shim = types.ModuleType("pytest")
    shim.raises = _raises
    shim.fixture = _fixture
    shim.approx = _Approx
    sys.modules["pytest"] = shim


_install_pytest_shim()

sys.path.insert(0, str(Path(__file__).parent))

# Load conftest fixtures
from tests import conftest as _conftest


_fixture_cache: dict = {}

def get_fixture(name):
    if name in _fixture_cache:
        return _fixture_cache[name]
    fn = getattr(_conftest, name, None)
    if fn is None:
        raise RuntimeError(f"Unknown fixture: {name}")
    sig = inspect.signature(fn)
    kwargs = {p: get_fixture(p) for p in sig.parameters}
    value = fn(**kwargs)
    _fixture_cache[name] = value
    return value


def run_module(mod_name):
    print(f"\n--- {mod_name} ---")
    mod = importlib.import_module(mod_name)
    tests = [(name, fn) for name, fn in vars(mod).items()
             if name.startswith("test_") and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        _fixture_cache.clear()  # function-scoped fixtures
        sig = inspect.signature(fn)
        try:
            kwargs = {}
            for pname, param in sig.parameters.items():
                if pname == "tmp_path":
                    import tempfile
                    kwargs["tmp_path"] = Path(tempfile.mkdtemp())
                else:
                    kwargs[pname] = get_fixture(pname)
            fn(**kwargs)
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            failed += 1
    return passed, failed


def main():
    modules = [
        "tests.test_ingest",
        "tests.test_retrieval",
        "tests.test_confidence",
        "tests.test_generation_and_fallback",
        "tests.test_cache_and_observability",
        "tests.test_pipeline_e2e",
        "tests.test_config",
        "tests.test_graph",
        "tests.test_multilingual",
        "tests.test_strategies",
        "tests.test_new_providers",
        "tests.test_citation_fallback",
        "tests.test_jobs",
        "tests.test_vector_stores_external",
    ]
    total_pass, total_fail = 0, 0
    for m in modules:
        p, f = run_module(m)
        total_pass += p
        total_fail += f
    print(f"\n{'='*60}\nTotal: {total_pass} passed, {total_fail} failed")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
