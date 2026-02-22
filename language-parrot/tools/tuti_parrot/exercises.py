"""
Parroting Exercise Implementations
Each exercise type guides learners through different repetition methods
"""

import time
from typing import List
from .lesson_loader import Phrase
from .display import Display


class ExerciseRunner:
    """Runs different types of parroting exercises"""

    def __init__(self):
        self.display = Display()

    def listen_repeat(self, phrases: List[Phrase]) -> None:
        """
        Listen-Repeat Exercise
        Listen to native audio, then repeat aloud multiple times
        """
        self.display.exercise_instructions(
            "listen-repeat",
            "Listen to each phrase carefully, then repeat it aloud.\n"
            "Focus on mimicking the exact sounds, rhythm, and intonation."
        )

        for i, phrase in enumerate(phrases, 1):
            self.display.progress_indicator(i - 1, len(phrases))
            self.display.phrase_full(phrase, show_number=i)

            reps = phrase.parrot_method.repetitions_suggested if phrase.parrot_method else 10

            self.display.info(f"Repeat this phrase {reps} times out loud")

            if phrase.audio_file:
                self.display.warning(f"🔊 Audio file: {phrase.audio_file}")
                self.display.info("(Audio playback not yet implemented - read and speak!)")

            # Wait for user to practice
            self.display.prompt("Press Enter when you've practiced this phrase...")

        self.display.success("Listen-Repeat exercise complete!")

    def shadowing(self, phrases: List[Phrase]) -> None:
        """
        Shadowing Exercise
        Speak simultaneously with native speaker audio
        """
        self.display.exercise_instructions(
            "shadowing",
            "Speak along WITH the audio as it plays.\n"
            "Try to match the speaker's pace, tone, and emotion exactly.\n"
            "Don't wait for them to finish - speak simultaneously!"
        )

        for i, phrase in enumerate(phrases, 1):
            self.display.progress_indicator(i - 1, len(phrases))
            print(f"\n{self.display.color(f'Phrase {i}:', 'bold')}")
            print(f"  {self.display.color(phrase.native, 'green')}")
            print(f"  {phrase.transliteration}")
            print(f"  → {phrase.translation}")

            if phrase.audio_file:
                self.display.warning(f"🔊 Audio: {phrase.audio_file}")
                self.display.info("(Play audio and speak along)")
            else:
                self.display.info("Read aloud with emotion and natural flow")

            self.display.prompt("Press Enter when done shadowing...")

        self.display.success("Shadowing exercise complete!")

    def pattern_drill(self, phrases: List[Phrase]) -> None:
        """
        Pattern Drill Exercise
        Focus on grammatical patterns and variations
        """
        self.display.exercise_instructions(
            "pattern-drill",
            "Notice the PATTERNS in these phrases.\n"
            "Look for repeating structures, word endings, or grammar rules.\n"
            "Repeat variations to internalize the pattern."
        )

        # Show all phrases first to see patterns
        self.display.subheader("Study these related phrases:")
        for i, phrase in enumerate(phrases, 1):
            self.display.phrase_compact(phrase, i)

        self.display.prompt("\nPress Enter to continue...")

        # Now drill each one
        for i, phrase in enumerate(phrases, 1):
            self.display.progress_indicator(i - 1, len(phrases))
            self.display.phrase_full(phrase, show_number=i)

            if phrase.patterns:
                self.display.info("Notice these patterns and repeat 8 times")

            self.display.prompt("Press Enter when you've drilled this phrase...")

        self.display.success("Pattern drill complete!")

    def progressive_speed(self, phrases: List[Phrase]) -> None:
        """
        Progressive Speed Exercise
        Start slow, gradually increase to conversational speed
        """
        self.display.exercise_instructions(
            "progressive-speed",
            "Start VERY slowly, then gradually speed up.\n"
            "Do 5 repetitions, getting faster each time:\n"
            "  1. Very slow (exaggerated)\n"
            "  2. Slow (careful)\n"
            "  3. Medium (deliberate)\n"
            "  4. Normal (conversational)\n"
            "  5. Fast (natural flow)"
        )

        for i, phrase in enumerate(phrases, 1):
            self.display.progress_indicator(i - 1, len(phrases))
            print(f"\n{self.display.color(f'Phrase {i}:', 'bold')}")
            print(f"  {self.display.color(phrase.native, 'green')}")
            print(f"  {phrase.transliteration}")
            print(f"  → {phrase.translation}")

            speeds = ["Very Slow", "Slow", "Medium", "Normal", "Fast"]
            for speed in speeds:
                print(f"\n  {self.display.color(f'→ {speed}:', 'yellow')}", end=' ')
                time.sleep(0.5)
                print(phrase.transliteration)

            self.display.prompt("\nPress Enter after completing all 5 speeds...")

        self.display.success("Progressive speed exercise complete!")

    def memory_recall(self, phrases: List[Phrase]) -> None:
        """
        Memory Recall Exercise
        Try to remember and say phrases from memory
        """
        self.display.exercise_instructions(
            "memory-recall",
            "Test your memory! Try to recall each phrase WITHOUT looking.\n"
            "Say it out loud from memory, then check your answer."
        )

        score = 0
        for i, phrase in enumerate(phrases, 1):
            self.display.progress_indicator(i - 1, len(phrases))
            print(f"\n{self.display.color(f'Question {i}:', 'bold')}")
            print(f"  How do you say: {self.display.color(phrase.translation, 'cyan')}?")

            self.display.prompt("Say it out loud, then press Enter to see the answer...")

            # Reveal answer
            print(f"\n  {self.display.color('Answer:', 'green')}")
            print(f"  {phrase.native}")
            print(f"  {phrase.transliteration}")

            # Self-assessment
            got_it = self.display.prompt("Did you get it right? (y/n)").lower().strip()
            if got_it == 'y':
                score += 1
                self.display.success("Great!")
            else:
                self.display.info("Keep practicing this one!")

        # Show final score
        percentage = (score / len(phrases)) * 100
        print(f"\n{self.display.color('━' * 40, 'cyan')}")
        print(f"Final Score: {self.display.color(f'{score}/{len(phrases)}', 'bold')} ({percentage:.0f}%)")

        if percentage >= 80:
            self.display.success("Excellent! You're ready to move forward!")
        elif percentage >= 60:
            self.display.info("Good progress! Review the ones you missed.")
        else:
            self.display.warning("Keep practicing! Repetition is key.")

        self.display.success("Memory recall exercise complete!")

    def run_exercise(self, exercise_type: str, phrases: List[Phrase]) -> None:
        """
        Run the appropriate exercise based on type

        Args:
            exercise_type: Type of exercise to run
            phrases: List of phrases to practice
        """
        exercise_map = {
            'listen-repeat': self.listen_repeat,
            'shadowing': self.shadowing,
            'pattern-drill': self.pattern_drill,
            'progressive-speed': self.progressive_speed,
            'memory-recall': self.memory_recall,
        }

        exercise_func = exercise_map.get(exercise_type)
        if exercise_func:
            exercise_func(phrases)
        else:
            self.display.error(f"Unknown exercise type: {exercise_type}")
