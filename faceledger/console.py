"""Terminal-specific live operation presentation."""

from typing import TextIO

from faceledger.comparison import ComparisonOutcome, Diagnostic, ProgressNotification
from faceledger.presentation import render_comparison_result, render_diagnostic


class ConsolePresentationFailure(RuntimeError):
    """Signal that a terminal stream could not present operation feedback."""


class ComparisonConsole:
    """Present one comparison run without replaying streamed diagnostics."""

    def __init__(
        self,
        stdout: TextIO,
        stderr: TextIO,
        *,
        show_progress: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._show_progress = show_progress
        self._progress_width = 0

    @staticmethod
    def _write(stream: TextIO, text: str) -> None:
        """Write and flush console text or raise a presentation failure."""

        try:
            stream.write(text)
            stream.flush()
        except Exception as error:
            raise ConsolePresentationFailure(str(error)) from error

    def _clear_progress(self) -> None:
        """Erase the currently visible transient progress line."""

        if self._progress_width:
            self._write(self._stderr, f"\r{' ' * self._progress_width}\r")
            self._progress_width = 0

    def diagnostic(self, diagnostic: Diagnostic) -> None:
        """Render one diagnostic notification as it arises."""

        self._clear_progress()
        self._write(self._stderr, render_diagnostic(diagnostic))

    def progress(self, notification: ProgressNotification) -> None:
        """Replace the transient completed-count and current-path line."""

        if not self._show_progress:
            return
        self._clear_progress()
        text = f"Completed {notification.completed_items}: {notification.path}"
        self._progress_width = len(text)
        self._write(self._stderr, f"\r{text}")

    def present(self, outcome: ComparisonOutcome) -> int:
        """Render final result and warning summary after live notifications."""

        self._clear_progress()
        if not outcome.successful:
            return 1

        self._write(self._stdout, render_comparison_result(outcome))
        warning_count = sum(
            diagnostic.severity == "warning" for diagnostic in outcome.diagnostics
        )
        if warning_count:
            warning_label = "warning" if warning_count == 1 else "warnings"
            compared_count = outcome.target_identities_compared
            compared_label = (
                "target identity" if compared_count == 1 else "target identities"
            )
            match_count = len(outcome.matches)
            match_label = "candidate match" if match_count == 1 else "candidate matches"
            self._write(
                self._stderr,
                f"WARNING SUMMARY: {warning_count} {warning_label}; "
                f"{compared_count} {compared_label} compared; "
                f"{match_count} {match_label}.\n",
            )
        return 0

    def report_presentation_failure(
        self,
        error: ConsolePresentationFailure,
    ) -> int:
        """Best-effort report a broken console callback and return failure."""

        progress_width = self._progress_width
        self._progress_width = 0
        clear = f"\r{' ' * progress_width}\r" if progress_width else ""
        try:
            self._stderr.write(
                f"{clear}ERROR [presentation:presentation-failure]: {error}\n"
            )
            self._stderr.flush()
        except Exception:  # noqa: BLE001
            return 1
        return 1
