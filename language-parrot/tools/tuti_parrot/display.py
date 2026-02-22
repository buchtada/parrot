"""
Display utilities for formatting and presenting content in the terminal
"""

from typing import List, Optional
from .lesson_loader import Phrase, Lesson


class Display:
    """Handles formatted display of lessons and phrases"""

    # Color codes for terminal
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'red': '\033[91m',
    }

    @classmethod
    def color(cls, text: str, color: str) -> str:
        """Wrap text in color codes"""
        return f"{cls.COLORS.get(color, '')}{text}{cls.COLORS['reset']}"

    @classmethod
    def header(cls, text: str) -> None:
        """Print a header"""
        print(f"\n{cls.color('=' * 60, 'cyan')}")
        print(cls.color(text.center(60), 'bold'))
        print(f"{cls.color('=' * 60, 'cyan')}\n")

    @classmethod
    def subheader(cls, text: str) -> None:
        """Print a subheader"""
        print(f"\n{cls.color(text, 'yellow')}")
        print(cls.color('-' * len(text), 'yellow'))

    @classmethod
    def lesson_overview(cls, lesson: Lesson) -> None:
        """Display lesson overview"""
        cls.header(f"📚 {lesson.title}")

        print(cls.color("Level:", 'bold'), lesson.level.upper())
        print(cls.color("Description:", 'bold'), lesson.description)

        if lesson.learning_objectives:
            print(f"\n{cls.color('Learning Objectives:', 'bold')}")
            for obj in lesson.learning_objectives:
                print(f"  • {obj}")

        print(f"\n{cls.color('Phrases:', 'bold')} {len(lesson.phrases)}")
        print(f"{cls.color('Exercises:', 'bold')} {len(lesson.parrot_exercises)}")

    @classmethod
    def phrase_full(cls, phrase: Phrase, show_number: Optional[int] = None) -> None:
        """Display a phrase with all details"""

        # Header with number if provided
        if show_number is not None:
            print(f"\n{cls.color(f'━━━ Phrase {show_number} ━━━', 'cyan')}")
        else:
            print(f"\n{cls.color('━━━━━━━━━━━━━━━━', 'cyan')}")

        # Main phrase
        print(f"\n  {cls.color(phrase.native, 'green')} ({cls.color(phrase.formality, 'dim')})")
        print(f"  {cls.color(phrase.transliteration, 'yellow')}")
        print(f"  → {cls.color(phrase.translation, 'bold')}")

        # Literal translation
        if phrase.literal_translation:
            print(f"  💭 Literally: {cls.color(phrase.literal_translation, 'dim')}")

        # Frequency
        if phrase.frequency_rank:
            print(f"  📊 Frequency: #{phrase.frequency_rank}")

        # Cultural notes
        if phrase.cultural_notes:
            print(f"\n  {cls.color('🎭 Cultural Context:', 'magenta')}")
            print(f"  {phrase.cultural_notes}")

        # Usage context
        if phrase.usage_context:
            print(f"\n  {cls.color('💬 When to use:', 'blue')}")
            for context in phrase.usage_context:
                print(f"    • {context}")

        # Patterns
        if phrase.patterns:
            print(f"\n  {cls.color('🔍 Language Patterns:', 'cyan')}")
            for pattern in phrase.patterns:
                print(f"    • {pattern}")

        # Breakdown
        if phrase.breakdown:
            print(f"\n  {cls.color('📝 Word Breakdown:', 'yellow')}")
            for word in phrase.breakdown:
                print(f"    {word.word} ({word.transliteration}) = {word.meaning} [{word.part_of_speech}]")

        # Parrot method
        if phrase.parrot_method:
            print(f"\n  {cls.color('🦜 Parroting Instructions:', 'green')}")
            print(f"    Repetitions: {phrase.parrot_method.repetitions_suggested}")
            print(f"    Speed: {phrase.parrot_method.speed}")
            if phrase.parrot_method.focus_sounds:
                print(f"    Focus on these sounds:")
                for sound in phrase.parrot_method.focus_sounds:
                    print(f"      • {sound}")

    @classmethod
    def phrase_compact(cls, phrase: Phrase, index: int) -> None:
        """Display phrase in compact format for lists"""
        print(f"{cls.color(f'{index}.', 'dim')} {cls.color(phrase.native, 'green')} "
              f"({phrase.transliteration}) → {phrase.translation}")

    @classmethod
    def exercise_instructions(cls, exercise_type: str, instructions: str) -> None:
        """Display exercise instructions"""
        cls.subheader(f"🎯 Exercise: {exercise_type.upper()}")
        print(f"\n{instructions}\n")

    @classmethod
    def progress_indicator(cls, current: int, total: int) -> None:
        """Show progress bar"""
        percentage = (current / total) * 100 if total > 0 else 0
        filled = int(percentage / 5)
        bar = '█' * filled + '░' * (20 - filled)
        print(f"\n{cls.color('Progress:', 'bold')} [{bar}] {percentage:.0f}% ({current}/{total})")

    @classmethod
    def prompt(cls, text: str) -> str:
        """Display a prompt and get user input"""
        return input(f"\n{cls.color(text, 'yellow')} ")

    @classmethod
    def success(cls, text: str) -> None:
        """Display success message"""
        print(f"\n{cls.color('✓', 'green')} {text}")

    @classmethod
    def info(cls, text: str) -> None:
        """Display info message"""
        print(f"\n{cls.color('ℹ', 'blue')} {text}")

    @classmethod
    def warning(cls, text: str) -> None:
        """Display warning message"""
        print(f"\n{cls.color('⚠', 'yellow')} {text}")

    @classmethod
    def error(cls, text: str) -> None:
        """Display error message"""
        print(f"\n{cls.color('✗', 'red')} {text}")

    @classmethod
    def menu(cls, title: str, options: List[str]) -> None:
        """Display a menu"""
        cls.subheader(title)
        for i, option in enumerate(options, 1):
            print(f"  {cls.color(f'{i}.', 'cyan')} {option}")

    @classmethod
    def clear_screen(cls) -> None:
        """Clear the terminal screen"""
        print("\033[2J\033[H", end='')
