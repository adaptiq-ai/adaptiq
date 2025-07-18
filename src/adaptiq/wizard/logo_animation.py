import time
import os
from typing import List


def clear_screen():
    """Clear the terminal screen"""
    os.system("cls" if os.name == "nt" else "clear")


def get_adaptiq_logo() -> List[str]:
    """
    Returns the AdaptiQ ASCII art logo as a list of strings.
    """
    logo = [
        "╔════════════════════════════════════════════════════════════════════════╗",
        "║                                                                        ║",
        "║     █████╗  ██████╗  █████╗ ██████╗ ████████╗ ██╗ ██████╗              ║",
        "║    ██╔══██╗██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝ ██║██╔═══██╗             ║",
        "║    ███████║██║   ██║███████║██████╔╝   ██║    ██║██║   ██║             ║",
        "║    ██╔══██║██║   ██║██╔══██║██╔═══╝    ██║    ██║██║▄▄ ██║             ║",
        "║    ██║  ██║╚██████╔╝██║  ██║██║        ██║    ██║╚██████╔╝             ║",
        "║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝        ╚═╝    ╚═╝ ╚══▀▀═╝              ║",
        "║                                                                        ║",
        "║               🧙‍♂️  Prompt Optimization Wizard 🔮                     ║",
        "║                                                                        ║",
        "╚════════════════════════════════════════════════════════════════════════╝",
    ]
    return logo


# You can print the logo to see the result
if __name__ == "__main__":
    for line in get_adaptiq_logo():
        print(line)


def get_gradient_colors() -> List[str]:
    """
    Returns ANSI color codes for a nice gradient effect.
    Using cyan to blue gradient.
    """
    return [
        "\033[96m",  # Bright cyan
        "\033[36m",  # Cyan
        "\033[94m",  # Bright blue
        "\033[34m",  # Blue
        "\033[35m",  # Magenta
        "\033[95m",  # Bright magenta
    ]


def display_logo_animated():
    """
    Display the AdaptiQ logo with a smooth animated effect.
    """
    clear_screen()
    logo_lines = get_adaptiq_logo()
    colors = get_gradient_colors()
    reset_color = "\033[0m"

    print("\n" * 2)  # Add some top padding

    # Animate each line appearing with a slight delay and color
    for i, line in enumerate(logo_lines):
        color = colors[i % len(colors)]
        print(f"{color}{line.center(80)}{reset_color}")
        time.sleep(0.1)  # Small delay for animation effect

    # Add some breathing space
    print("\n")

    # Animated welcome message
    welcome_parts = [
        "✨ Initializing AdaptiQ Wizard...",
        "🔧 Loading optimization tools...",
        "🚀 Ready to optimize your prompts!",
    ]

    for part in welcome_parts:
        print(f"\033[92m{part.center(80)}\033[0m")  # Green color
        time.sleep(0.5)

    print("\n" + "=" * 80 + "\n")


def display_simple_logo():
    """
    Display a simpler version without animation for faster startup.
    """
    clear_screen()
    logo_lines = get_adaptiq_logo()
    reset_color = "\033[0m"
    logo_color = "\033[96m"  # Bright cyan

    print("\n" * 2)
    for line in logo_lines:
        print(f"{logo_color}{line.center(80)}{reset_color}")

    print(f"\n\033[92m{'🚀 AdaptiQ Wizard Ready!'.center(80)}\033[0m\n")
    print("=" * 80 + "\n")
