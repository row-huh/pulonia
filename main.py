from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from ollama import AsyncClient
from pydantic import BaseModel
from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.events import Resize
from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

# Pillow & textual-imageview support for image rendering in terminal
from PIL import Image as PILImage

try:
    from textual_imageview.widgets import ImageView

    HAS_IMAGE_VIEW = True
except ImportError:
    HAS_IMAGE_VIEW = False

# ---------------------------------------------------------------------------
# Responsive ASCII Art Banners
# ---------------------------------------------------------------------------

BIG_ASCII = r"""
  _____       _             _          _____          _      
 |  __ \     | |           (_)        / ____|        | |     
 | |__) |   _| | ___  _ __  _  __ _  | |     ___   __| | ___ 
 |  ___/ | | | |/ _ \| '_ \| |/ _` | | |    / _ \ / _` |/ _ \
 | |   | |_| | | (_) | | | | | (_| | | |___| (_) | (_| |  __/
 |_|    \__,_|_|\___/|_| |_|_|\__,_|  \_____\___/ \__,_|\___|
"""


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

BASE_DIR = Path(os.getenv("AGENT_WORKDIR", ".")).resolve()
# Directory the script itself lives in — used for locating bundled assets
# like pulonia.png, independent of whatever cwd the user launches from.
SCRIPT_DIR = Path(__file__).resolve().parent
MAX_TOOL_TURNS = 6


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


def edit_text(
    file_name: str,
    target: str,
    content: str,
    mode: Literal["replace", "before", "after"] = "replace",
) -> str:
    path = _safe_path(file_name)

    if not path.exists():
        raise FileNotFoundError(file_name)

    text = path.read_text(encoding="utf-8")

    if target not in text:
        raise ValueError("Target text not found.")

    if mode == "replace":
        updated = text.replace(target, content, 1)
    elif mode == "before":
        updated = text.replace(target, content + target, 1)
    elif mode == "after":
        updated = text.replace(target, target + content, 1)
    else:
        raise ValueError(f"Unknown edit mode: {mode}")

    path.write_text(updated, encoding="utf-8")
    return "Edited successfully."


def replace_text(file_name: str, old: str, new: str) -> str:
    path = _safe_path(file_name)

    if not path.exists():
        raise FileNotFoundError(file_name)

    text = path.read_text(encoding="utf-8")

    if old not in text:
        raise ValueError("Target text not found.")

    updated = text.replace(old, new, 1)
    path.write_text(updated, encoding="utf-8")
    return "Edited successfully."


def get_project_context(fileName: str = "") -> str:
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
    "EditFile": lambda args: edit_text(
        args["fileName"], args["target"], args["newText"], args["mode"]
    ),
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
    {
        "type": "function",
        "function": {
            "name": "EditText",
            "description": (
                "Edit an existing file by replacing text or inserting new text "
                "before or after a target string. The target must exist in the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fileName": {
                        "type": "string",
                        "description": "Path to the file, relative to the project root.",
                    },
                    "target": {
                        "type": "string",
                        "description": "The existing text to search for in the file.",
                    },
                    "newText": {
                        "type": "string",
                        "description": "The text to insert or use as the replacement.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "before", "after"],
                        "description": "How to apply the edit.",
                    },
                },
                "required": ["fileName", "target", "newText", "mode"],
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
    if isinstance(chunk, dict):
        message = chunk.get("message") or {}
        return message.get("content", "") or ""

    message = getattr(chunk, "message", None)
    if message is None:
        return ""

    return getattr(message, "content", "") or ""


def extract_tool_calls(chunk: Any) -> list[Any]:
    if isinstance(chunk, dict):
        message = chunk.get("message") or {}
        return message.get("tool_calls") or []

    message = getattr(chunk, "message", None)
    if message is None:
        return []

    return getattr(message, "tool_calls", None) or []


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Pulonia(App):
    """Minimal Textual chat UI: centered hero (image + ascii + input) on landing,
    switches to full chat workspace once the first message is sent."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    /* Landing Container */
    #landing_container {
        width: 100%;
        height: 1fr;
        align: center middle;
    }

    #hero_box {
        width: auto;
        height: auto;
        align: center middle;
    }

    #hero_image {
        width: 40;
        height: 15;
        margin-bottom: 1;
    }

    #hero_fallback {
        width: 100%;
        text-align: center;
        color: $accent;
        margin-bottom: 1;
    }

    /* Active Chat Container (hidden at start) */
    #chat_container {
        display: none;
        height: 1fr;
        layout: vertical;
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

    /* Input: centered/inline while on the landing hero, docked once chat starts */
    #input {
        width: 60;
        border: round $primary;
        margin-top: 1;
    }

    #input.docked {
        dock: bottom;
        width: 100%;
        margin-top: 0;
    }

    #input:focus {
        border: round $accent;
    }

    #autocomplete {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
    ]

    COMMANDS = {
        "/model": "Switch or view Ollama models",
        "/clear": "Clear current chat history",
        "/session": "View cached chat sessions",
    }

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
        self._has_started_chat = False

    def compose(self) -> ComposeResult:
        yield Header()

        # 1. Centered Hero Landing Screen — image + ascii text + input, stacked
        #    together as one centered block (not either/or, both together).
        with Center(id="landing_container"):
            with Vertical(id="hero_box"):
                # Resolve against the script's own directory, not cwd — this was
                # the actual reason pulonia.png silently failed to load before.
                image_path = SCRIPT_DIR / "pulonia.png"
                if HAS_IMAGE_VIEW and image_path.exists():
                    try:
                        pil_img = PILImage.open(image_path)
                        yield ImageView(pil_img, id="hero_image")
                    except Exception as exc:
                        yield Static(
                            f"[dim red](image failed to load: {exc})[/]",
                            id="hero_image_error",
                        )
                elif not HAS_IMAGE_VIEW:
                    yield Static(
                        "[dim](textual-imageview not installed — pip install textual-imageview)[/]",
                        id="hero_image_error",
                    )
                elif not image_path.exists():
                    yield Static(
                        f"[dim](pulonia.png not found at {image_path})[/]",
                        id="hero_image_error",
                    )

                yield Static(f"[bold magenta]{BIG_ASCII}[/]", id="hero_fallback")

                yield Input(
                    placeholder="Ask Pulonia something or type / for commands...",
                    id="input",
                )

        # 2. Main Chat Workspace Screen (Hidden initially)
        with Vertical(id="chat_container"):
            messages = RichLog(id="messages", wrap=True, highlight=False, markup=True)
            messages.border_title = "Chat"
            yield messages

            stream = Static("", id="stream")
            stream.border_title = "Live"
            yield stream

            with Horizontal(id="status_bar"):
                yield Static("Ready", id="status")
                yield Static("", id="timer")

        yield OptionList(id="autocomplete")
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
        self._update_ascii_art(self.size.width)

    def on_resize(self, event: Resize) -> None:
        """Dynamically adapts the ASCII banner according to terminal width."""
        self._update_ascii_art(event.size.width)

    def _update_ascii_art(self, width: int) -> None:
        try:
            fallback_widget = self.query_one("#hero_fallback", Static)
        except Exception:
            return


        fallback_widget.update(f"[bold magenta]{BIG_ASCII}[/]")

    async def _switch_to_chat_view(self) -> None:
        """Transitions the UI from the central hero view to the chat interface,
        re-parenting the single Input widget from the hero block down to a
        bottom-docked position."""
        if self._has_started_chat:
            return

        self.query_one("#landing_container").styles.display = "none"
        self.query_one("#chat_container").styles.display = "block"

        assert self._input_widget is not None
        await self._input_widget.remove()
        await self.mount(self._input_widget)
        self._input_widget.add_class("docked")
        self._input_widget.focus()

        self._has_started_chat = True

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

    def on_input_changed(self, event: Input.Changed) -> None:
        value = event.value
        autocomplete = self.query_one("#autocomplete", OptionList)

        if value.startswith("/"):
            query = value.lower()
            matches = [
                (cmd, desc)
                for cmd, desc in self.COMMANDS.items()
                if cmd.lower().startswith(query)
            ]

            if matches:
                autocomplete.clear_options()
                for cmd, desc in matches:
                    autocomplete.add_option(
                        Option(f"[bold cyan]{cmd}[/] [dim]- {desc}[/]", id=cmd)
                    )
                autocomplete.display = True
                return

        autocomplete.display = False

    def on_key(self, event) -> None:
        autocomplete = self.query_one("#autocomplete", OptionList)

        if autocomplete.display:
            if event.key == "down":
                autocomplete.action_cursor_down()
                event.prevent_default()
            elif event.key == "up":
                autocomplete.action_cursor_up()
                event.prevent_default()
            elif event.key == "tab" or (
                event.key == "enter" and autocomplete.highlighted is not None
            ):
                if autocomplete.highlighted_at is not None:
                    option = autocomplete.get_option_at_index(
                        autocomplete.highlighted_at
                    )
                    if option and option.id:
                        input_widget = self.query_one("#input", Input)
                        input_widget.value = f"{option.id} "
                        input_widget.cursor_position = len(input_widget.value)
                        autocomplete.display = False
                        event.prevent_default()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        input_widget = self.query_one("#input", Input)
        autocomplete = self.query_one("#autocomplete", OptionList)

        if event.option.id:
            input_widget.value = f"{event.option.id} "
            input_widget.focus()
            input_widget.cursor_position = len(input_widget.value)

        autocomplete.display = False

    def _start_spinner(self) -> None:
        self._spinner_frame = 0
        self._request_start = time.perf_counter()
        self._spinner_timer = self.set_interval(0.08, self._tick_spinner)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _run_tool_call(self, tool_call: Any) -> dict[str, Any]:
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
            except Exception as exc:
                result = f"error running {name}: {exc}"

        if self._messages_widget:
            self._messages_widget.write(
                f"[dim]→ {name}({args}) => {str(result)[:200]}[/]"
            )
        return {"role": "tool", "name": name, "content": str(result)}

    async def _handle_command(self, raw: str) -> None:
        parts = raw[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "model":
            await self._cmd_model(arg)
        elif cmd == "clear":
            self.action_clear_chat()
        else:
            if self._messages_widget:
                self._messages_widget.write(f"[bold yellow]unknown command: /{cmd}[/]")

    async def _cmd_model(self, arg: str) -> None:
        try:
            resp = await self._ollama.list()
        except Exception as exc:
            if self._messages_widget:
                self._messages_widget.write(
                    f"[bold red]could not list models: {exc}[/]"
                )
            return

        raw_models = (
            resp.get("models")
            if isinstance(resp, dict)
            else getattr(resp, "models", [])
        )
        names = [
            m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            for m in raw_models
            if m
        ]
        names = [n for n in names if n]

        if not names:
            if self._messages_widget:
                self._messages_widget.write("[bold yellow]no local models found[/]")
            return

        if not arg:
            lines = [
                f"  [{i}] {n}" + ("  [dim](current)[/]" if n == self.model else "")
                for i, n in enumerate(names)
            ]
            if self._messages_widget:
                self._messages_widget.write(
                    "[bold cyan]local models:[/]\n" + "\n".join(lines)
                )
                self._messages_widget.write("[dim]/model <number|name> to switch[/]")
            return

        picked = None
        if arg.isdigit() and 0 <= int(arg) < len(names):
            picked = names[int(arg)]
        else:
            matches = [n for n in names if n == arg or n.startswith(arg)]
            if matches:
                picked = matches[0]

        if picked is None:
            if self._messages_widget:
                self._messages_widget.write(f"[bold red]no match for '{arg}'[/]")
            return

        self.model = picked
        if self._messages_widget:
            self._messages_widget.write(f"[bold green]switched model →[/] {picked}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        event.input.value = ""

        # Transition layout from landing hero to full chat interface
        await self._switch_to_chat_view()

        if prompt.startswith("/"):
            await self._handle_command(prompt)
            return

        assert self._messages_widget is not None
        assert self._stream_widget is not None
        assert self._input_widget is not None

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
                    continue

                if answer:
                    self.messages.append({"role": "assistant", "content": answer})
                break
            else:
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