import sys
import threading
import time
from typing import List


class SpinnerAnimation:
    """
    A class to handle spinner animations in the CLI.
    """

    def __init__(self):
        self.spinner_active = False
        self.spinner_thread = None

    def get_spinner_frames(self) -> List[str]:
        """
        Returns different spinner frame options.
        You can customize these or add more styles.
        """
        # Classic spinning dots
        classic = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # Magic wand style for wizard theme
        magic = ["🪄 ✨", "🪄  ✨", "🪄   ✨", "🪄    ✨", "🪄     ✨", "🪄      ✨"]

        # Crystal ball style
        crystal = ["🔮", "✨🔮", "✨🔮✨", "🔮✨", "🔮"]

        # Simple dots
        dots = ["   ", ".  ", ".. ", "..."]

        return classic  # You can change this to any style above

    def get_spinner_messages(self) -> List[str]:
        """
        Returns fun messages to display with the spinner.
        """
        messages = [
            "🧙‍♂️ Conjuring your response",
            "✨ Weaving magic words",
            "🌟 Brewing the perfect answer",
            "🎭 Channeling AI wisdom",
            "🔬 Analyzing your request",
            "🚀 Optimizing response quality",
        ]
        return messages

    def start_spinner(self, message: str = None):
        """
        Start the spinner animation with optional custom message.

        Args:
            message: Custom message to display with spinner
        """
        if self.spinner_active:
            return

        self.spinner_active = True
        frames = self.get_spinner_frames()
        messages = self.get_spinner_messages()

        # Use provided message or pick a random one
        if message is None:
            import random

            display_message = random.choice(messages)
        else:
            display_message = message

        def animate():
            frame_idx = 0
            while self.spinner_active:
                frame = frames[frame_idx % len(frames)]
                # Clear the line and print spinner with message
                sys.stdout.write(f"\r{frame} {display_message}...")
                sys.stdout.flush()
                time.sleep(0.1)
                frame_idx += 1

        self.spinner_thread = threading.Thread(target=animate, daemon=True)
        self.spinner_thread.start()

    def stop_spinner(self):
        """
        Stop the spinner animation and clear the line.
        """
        if not self.spinner_active:
            return

        self.spinner_active = False
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.2)

        # Clear the spinner line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()


# Global spinner instance
spinner = SpinnerAnimation()


def start_thinking_animation(message: str = None):
    """
    Start the thinking/processing animation.

    Args:
        message: Optional custom message to display
    """
    spinner.start_spinner(message)


def stop_thinking_animation():
    """
    Stop the thinking animation and clear the line.
    """
    spinner.stop_spinner()
