"""Persist real unittest results, including failures, rather than a claimed count."""
import io
from pathlib import Path
import unittest
from storage import save_json


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.append({"test": test.id(), "status": "failed", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.results.append({"test": test.id(), "status": "error", "detail": self._exc_info_to_string(err, test)})


if __name__ == "__main__":
    root = Path(__file__).parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    report = dict(tests_run=result.testsRun, passed=sum(r["status"] == "passed" for r in result.results),
                  failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  successful=result.wasSuccessful(), results=result.results)
    save_json(root / "reports/test_results.json", report)
    (root / "reports/test_output.txt").write_text(stream.getvalue())
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
