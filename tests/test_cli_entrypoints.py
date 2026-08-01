import io
import subprocess
import sys
import unittest
from pathlib import Path

from faceledger.cli import main


class CliApplicationTests(unittest.TestCase):
    def test_top_level_help_is_available(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        status = main(["--help"], stdout=stdout, stderr=stderr)

        self.assertEqual(status, 0)
        self.assertIn("usage: faceledger", stdout.getvalue())
        self.assertIn("candidate matches", stdout.getvalue())
        self.assertIn("command", stdout.getvalue())
        self.assertNotIn("{}", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_top_level_version_comes_from_installed_package_metadata(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        status = main(["--version"], stdout=stdout, stderr=stderr)

        self.assertEqual(status, 0)
        self.assertEqual(stdout.getvalue(), "faceledger 0.5.0\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_invocation_without_a_command_uses_argparse_status_two(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        status = main([], stdout=stdout, stderr=stderr)

        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(
            "the following arguments are required: command", stderr.getvalue()
        )


class CliProcessEntryPointTests(unittest.TestCase):
    def test_module_execution_calls_the_shared_application(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "faceledger", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "faceledger 0.5.0\n")
        self.assertEqual(completed.stderr, "")

    def test_installed_launcher_calls_the_shared_application(self) -> None:
        launcher = Path(sys.executable).with_name("faceledger")

        completed = subprocess.run(
            [launcher, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "faceledger 0.5.0\n")
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
