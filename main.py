from __future__ import annotations

import os
from typing import Any

from ollama import AsyncClient
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static


def extract_chunk_content(chunk: Any) -> str:
    """Extract streamed text from an Ollama chat chunk."""
    if isinstance(chunk, dict):
        message = chunk.get("message") or {}
        return message.get("content", "") or ""

    message = getattr(chunk, "message", None)
    if message is None:
        return ""

    return getattr(message, "content", "") or ""


class OllamaChatApp(App):
    """Minimal Textual chat UI that streams Ollama responses."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #messages {
        height: 1fr;
        border: round $primary;
        padding: 1;
    }

    #stream {
        height: auto;
        min-height: 3;
        border: round $accent;
        padding: 1;
    }

    #input {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ]
        self._ollama = AsyncClient()
        self._messages_widget: RichLog | None = None
        self._stream_widget: Static | None = None
        self._input_widget: Input | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield RichLog(id="messages", wrap=True, highlight=False, markup=False)
            yield Static("", id="stream")
            yield Input(placeholder="Type a message and press Enter", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._messages_widget = self.query_one("#messages", RichLog)
        self._stream_widget = self.query_one("#stream", Static)
        self._input_widget = self.query_one("#input", Input)
        self._input_widget.focus()
        self._messages_widget.write(f"Model: {self.model}")

    def action_clear_chat(self) -> None:
        self.messages = [self.messages[0]]
        if self._messages_widget is not None:
            self._messages_widget.clear()
            self._messages_widget.write(f"Model: {self.model}")
        if self._stream_widget is not None:
            self._stream_widget.update("")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        assert self._messages_widget is not None
        assert self._stream_widget is not None
        assert self._input_widget is not None

        event.input.value = ""
        self._input_widget.disabled = True
        self._messages_widget.write(f"\nYou: {prompt}")
        self.messages.append({"role": "user", "content": prompt})

        self._stream_widget.update("Assistant: ")
        chunks: list[str] = []

        try:
            stream = await self._ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
            )

            async for chunk in stream:
                content = extract_chunk_content(chunk)
                if not content:
                    continue
                chunks.append(content)
                self._stream_widget.update("Assistant: " + "".join(chunks))

        except Exception as exc:
            self._stream_widget.update(f"Assistant: error: {exc}")
        else:
            answer = "".join(chunks).strip()
            if answer:
                self.messages.append({"role": "assistant", "content": answer})
        finally:
            self._input_widget.disabled = False
            self._input_widget.focus()


if __name__ == "__main__":
    OllamaChatApp().run()
