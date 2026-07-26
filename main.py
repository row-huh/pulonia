from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ollama import AsyncClient
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Footer, Header, Input, RichLog, Static


# ---------------------------------------------------------------------------
# Agent tools
#
# All tools are sandboxed to AGENT_WORKDIR (default: current directory) so a
# model can never read/write outside the project folder, even if it tries a
# path like "../../etc/passwd".
# ---------------------------------------------------------------------------

BASE_DIR = Path(os.getenv("AGENT_WORKDIR", ".")).resolve()
MAX_TOOL_TURNS = 6  # hard cap so a tool-call loop can't run forever


def _safe_path(file_name: str) -> Path:
    candidate = (BASE_DIR / file_name).resolve()
    if BASE_DIR not in candidate.parents and candidate != BASE_DIR:
        raise ValueError(f"'{file_name}' resolves outside the project directory")
    return candidate


def read_file(fileName: str) -> str:
    path = _safe_path(fileName)
    if not path.is_file():
        return f"error: '{fileName}' does not exist or is not a file"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"error: '{fileName}' is not a text file"


def write_file(fileName: str, content: str) -> str:
    path = _safe_path(fileName)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to '{fileName}'"


def get_project_context(fileName: str = "") -> str:
    """Lightweight project map: directory tree plus, if fileName is given,
    a peek at that file's neighbors (siblings in the same directory)."""
    lines: list[str] = []
    for path in sorted(BASE_DIR.rglob("*")):
        if any(part.startswith(".") for part in path.relative_to(BASE_DIR).parts):
            continue
        if "__pycache__" in path.parts or "node_modules" in path.parts:
            continue
        rel = path.relative_to(BASE_DIR)
        marker = "/" if path.is_dir() else ""
        lines.append(f"{rel}{marker}")

    tree = "\n".join(lines) if lines else "(empty project directory)"

    if not fileName:
        return tree

    target = _safe_path(fileName)
    siblings = sorted(p.name for p in target.parent.glob("*") if p.is_file())
    return f"{tree}\n\n--- siblings of {fileName} ---\n" + "\n".join(siblings)


TOOL_IMPL = {
    "ReadFiles": lambda args: read_file(args["fileName"]),
    "WriteInFiles": lambda args: write_file(args["fileName"], args.get("content", "")),
    "GetProjectContext": lambda args: get_project_context(args.get("fileName", "")),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ReadFiles",
            "description": "Read the full contents of a file in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fileName": {
                        "type": "string",
                        "description": "Path to the file, relative to the project root.",
                    }
                },
                "required": ["fileName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WriteInFiles",
            "description": "Write (create or overwrite) a file in the project directory with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fileName": {
                        "type": "string",
                        "description": "Path to the file, relative to the project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file.",
                    },
                },
                "required": ["fileName", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GetProjectContext",
            "description": (
                "Get a map of the project's file/directory structure. If fileName is "
                "given, also lists sibling files in that file's directory for context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fileName": {
                        "type": "string",
                        "description": "Optional file path to get extra context around.",
                    }
                },
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You are Pulonia, a coding agent running locally against project files.

Project root: {BASE_DIR}
You may only access files inside this root; anything outside it is unavailable to you.

Tools available to you:
- ReadFiles(fileName): read a file's full contents.
- WriteInFiles(fileName, content): create or overwrite a file with the given content.
- GetProjectContext(fileName?): see the project's directory tree, and optionally the
  sibling files around a given file, to orient yourself before reading/writing.

Guidelines:
- Call GetProjectContext first if you don't already know the project layout.
- Read a file with ReadFiles before overwriting it with WriteInFiles, unless you are
  intentionally creating a new file from scratch.
- WriteInFiles overwrites the entire file. When editing an existing file, read it,
  make your change in full, and write back the complete new content.
- Don't call a tool if you already have the information you need in the conversation.
- Be direct and terse in your responses. Don't narrate every tool call in prose;
  just do the work and report the result.
"""


def extract_chunk_content(chunk: Any) -> str:
    """Extract streamed text from an Ollama chat chunk."""
    if isinstance(chunk, dict):
        message = chunk.get("message") or {}
        return message.get("content", "") or ""

    message = getattr(chunk, "message", None)
    if message is None:
        return ""

    return getattr(message, "content", "") or ""


def extract_tool_calls(chunk: Any) -> list[Any]:
    """Extract tool_calls from an Ollama chat chunk, if any."""
    if isinstance(chunk, dict):
        message = chunk.get("message") or {}
        return message.get("tool_calls") or []

    message = getattr(chunk, "message", None)
    if message is None:
        return []

    return getattr(message, "tool_calls", None) or []


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Pulonia(App):
    """Minimal Textual chat UI that streams Ollama responses, with file tools."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #messages {
        height: 1fr;
        border: round $primary;
        border-title-color: $primary;
        padding: 1 2;
        background: $panel;
    }

    #stream {
        height: auto;
        min-height: 3;
        border: round $accent;
        border-title-color: $accent;
        padding: 1 2;
        background: $panel-darken-1;
        color: $text;
    }

    #status_bar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $panel-darken-2;
    }

    #status {
        width: 1fr;
        content-align: left middle;
    }

    #timer {
        width: auto;
        content-align: right middle;
        color: $success;
        text-style: bold;
    }

    #input {
        dock: bottom;
        border: round $primary;
    }

    #input:focus {
        border: round $accent;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        self._ollama = AsyncClient()
        self._messages_widget: RichLog | None = None
        self._stream_widget: Static | None = None
        self._input_widget: Input | None = None
        self._status_widget: Static | None = None
        self._timer_widget: Static | None = None
        self._spinner_timer = None
        self._spinner_frame = 0
        self._request_start: float = 0.0

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            messages = RichLog(id="messages", wrap=True, highlight=False, markup=True)
            messages.border_title = "Chat"
            yield messages
            stream = Static("", id="stream")
            stream.border_title = "Live"
            yield stream
            with Horizontal(id="status_bar"):
                yield Static("Ready", id="status")
                yield Static("", id="timer")
            yield Input(placeholder="Type a message and press Enter", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._messages_widget = self.query_one("#messages", RichLog)
        self._stream_widget = self.query_one("#stream", Static)
        self._input_widget = self.query_one("#input", Input)
        self._status_widget = self.query_one("#status", Static)
        self._timer_widget = self.query_one("#timer", Static)
        self._input_widget.focus()
        self._messages_widget.write(f"[bold cyan]Model:[/] {self.model}")
        self._messages_widget.write(f"[bold cyan]Project root:[/] {BASE_DIR}")

    def action_clear_chat(self) -> None:
        self.messages = [self.messages[0]]
        if self._messages_widget is not None:
            self._messages_widget.clear()
            self._messages_widget.write(f"[bold cyan]Model:[/] {self.model}")
        if self._stream_widget is not None:
            self._stream_widget.update("")
        self._set_status("Ready")
        if self._timer_widget is not None:
            self._timer_widget.update("")

    def _set_status(self, text: str) -> None:
        if self._status_widget is not None:
            self._status_widget.update(text)

    def _tick_spinner(self, label: str = "thinking") -> None:
        if self._status_widget is None:
            return
        frame = SPINNER_FRAMES[self._spinner_frame % len(SPINNER_FRAMES)]
        self._spinner_frame += 1
        elapsed = time.perf_counter() - self._request_start
        self._status_widget.update(f"{frame} {label}...")
        if self._timer_widget is not None:
            self._timer_widget.update(f"{elapsed:0.1f}s")

    def _start_spinner(self) -> None:
        self._spinner_frame = 0
        self._request_start = time.perf_counter()
        self._spinner_timer = self.set_interval(0.08, self._tick_spinner)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _run_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Execute one tool call and return a `role: tool` message for it."""
        if isinstance(tool_call, dict):
            fn = tool_call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
        else:
            fn = getattr(tool_call, "function", None)
            name = getattr(fn, "name", "")
            raw_args = getattr(fn, "arguments", {})

        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")

        impl = TOOL_IMPL.get(name)
        if impl is None:
            result = f"error: unknown tool '{name}'"
        else:
            try:
                result = impl(args)
            except Exception as exc:  # noqa: BLE001 - surface any tool error to the model
                result = f"error running {name}: {exc}"

        self._messages_widget.write(f"[dim]→ {name}({args}) => {str(result)[:200]}[/]")
        return {"role": "tool", "name": name, "content": str(result)}

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        assert self._messages_widget is not None
        assert self._stream_widget is not None
        assert self._input_widget is not None

        event.input.value = ""
        self._input_widget.disabled = True
        self._messages_widget.write(f"\n[bold green]You:[/] {prompt}")
        self.messages.append({"role": "user", "content": prompt})

        self._start_spinner()

        try:
            for turn in range(MAX_TOOL_TURNS):
                self._stream_widget.update("[bold magenta]Pulonia:[/] ")
                chunks: list[str] = []
                pending_tool_calls: list[Any] = []

                stream = await self._ollama.chat(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    stream=True,
                )

                first_token = True
                async for chunk in stream:
                    content = extract_chunk_content(chunk)
                    if content:
                        if first_token:
                            first_token = False
                            self._set_status("streaming...")
                        chunks.append(content)
                        self._stream_widget.update(
                            "[bold magenta]Pulonia:[/] " + "".join(chunks)
                        )

                    tool_calls = extract_tool_calls(chunk)
                    if tool_calls:
                        pending_tool_calls.extend(tool_calls)

                answer = "".join(chunks).strip()

                if pending_tool_calls:
                    # Record the assistant's tool-call turn, then run each tool
                    # and feed results back for the next iteration.
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "tool_calls": pending_tool_calls,
                        }
                    )
                    self._set_status(f"running {len(pending_tool_calls)} tool(s)...")
                    for tool_call in pending_tool_calls:
                        self.messages.append(self._run_tool_call(tool_call))
                    continue  # loop back into the model with tool results

                # No tool calls: this is the final answer for this turn.
                if answer:
                    self.messages.append({"role": "assistant", "content": answer})
                break
            else:
                self._stream_widget.update(
                    self._stream_widget.renderable
                    if hasattr(self._stream_widget, "renderable")
                    else ""
                )
                self._messages_widget.write(
                    "[bold yellow]⚠ hit max tool-call turns without a final answer[/]"
                )

        except Exception as exc:
            self._stream_widget.update(f"[bold red]Pulonia: error:[/] {exc}")
            self._set_status("error")
        else:
            elapsed = time.perf_counter() - self._request_start
            self._set_status("Ready")
            if self._timer_widget is not None:
                self._timer_widget.update(f"responded in {elapsed:0.2f}s")
        finally:
            self._stop_spinner()
            self._input_widget.disabled = False
            self._input_widget.focus()


if __name__ == "__main__":
    Pulonia().run()