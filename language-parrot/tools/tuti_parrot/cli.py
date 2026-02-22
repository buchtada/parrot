"""
Main CLI Application - Tuti Parrot Language Learning
"""

import sys
from pathlib import Path
from typing import Optional
from .lesson_loader import LessonLoader, Lesson
from .exercises import ExerciseRunner
from .progress import ProgressTracker
from .display import Display


class TutiParrotCLI:
    """Main application controller"""

    def __init__(self):
        self.loader = LessonLoader()
        self.exercise_runner = ExerciseRunner()
        self.progress = ProgressTracker()
        self.display = Display()
        self.current_lesson: Optional[Lesson] = None

    def show_welcome(self) -> None:
        """Display welcome message"""
        self.display.clear_screen()
        print(f"""
{self.display.color('╔═══════════════════════════════════════════════════════════╗', 'cyan')}
{self.display.color('║                                                           ║', 'cyan')}
{self.display.color('║', 'cyan')}          🦜 {self.display.color('طوطی خل', 'green')} - {self.display.color('Tuti Parrot', 'yellow')} 🦜                 {self.display.color('║', 'cyan')}
{self.display.color('║                                                           ║', 'cyan')}
{self.display.color('║', 'cyan')}           {self.display.color('Learn Languages Through Parroting', 'bold')}            {self.display.color('║', 'cyan')}
{self.display.color('║                                                           ║', 'cyan')}
{self.display.color('╚═══════════════════════════════════════════════════════════╝', 'cyan')}
""")

    def select_language(self) -> Optional[str]:
        """Select a language to learn"""
        languages = self.loader.list_languages()

        if not languages:
            self.display.error("No languages found in curriculum!")
            return None

        self.display.menu("Select a Language:", languages)

        choice = self.display.prompt("Enter language number (or 'q' to quit):").strip()

        if choice.lower() == 'q':
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(languages):
                return languages[idx]
        except ValueError:
            pass

        self.display.error("Invalid choice")
        return None

    def select_lesson(self, language: str) -> Optional[str]:
        """Select a lesson from the language"""
        lessons = self.loader.list_lessons(language)

        if not lessons:
            self.display.error(f"No lessons found for {language}!")
            return None

        # Show lessons with progress
        self.display.subheader(f"Lessons for {language.upper()}")
        for i, lesson_info in enumerate(lessons, 1):
            prog = self.progress.get_lesson_progress(lesson_info['lesson_id'])
            progress_str = ""
            if prog:
                prog_pct = f'{prog.completion_percentage:.0f}%'
                progress_str = f" [{self.display.color(prog_pct, 'green')}]"

            num_colored = self.display.color(f'{i}.', 'cyan')
            level_str = f"[{lesson_info['level']}]"
            level_colored = self.display.color(level_str, 'dim')

            print(f"  {num_colored} {lesson_info['title']} {level_colored}{progress_str}")

        choice = self.display.prompt("\nEnter lesson number (or 'b' for back):").strip()

        if choice.lower() == 'b':
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(lessons):
                return lessons[idx]['filename']
        except ValueError:
            pass

        self.display.error("Invalid choice")
        return None

    def show_lesson_menu(self, lesson: Lesson) -> str:
        """Show lesson menu and get user choice"""
        self.display.lesson_overview(lesson)

        options = [
            "Study all phrases",
            "Practice exercises",
            "Review specific phrases",
            "View progress",
            "Back to lesson selection"
        ]

        self.display.menu("What would you like to do?", options)

        choice = self.display.prompt("Enter your choice:").strip()
        return choice

    def study_all_phrases(self, lesson: Lesson) -> None:
        """Study all phrases in the lesson"""
        self.display.header("📖 Studying All Phrases")

        for i, phrase in enumerate(lesson.phrases, 1):
            self.display.progress_indicator(i - 1, len(lesson.phrases))
            self.display.phrase_full(phrase, show_number=i)

            choice = self.display.prompt(
                "Enter 'm' to mark as mastered, Enter to continue, 'q' to quit:"
            ).strip().lower()

            if choice == 'm':
                self.progress.mark_phrase_mastered(
                    lesson.lesson_id,
                    lesson.language,
                    phrase.id
                )
                self.display.success("Phrase marked as mastered!")
            elif choice == 'q':
                break

        self.progress.update_completion(
            lesson.lesson_id,
            lesson.language,
            len(lesson.parrot_exercises),
            len(lesson.phrases)
        )

    def practice_exercises(self, lesson: Lesson) -> None:
        """Practice the lesson exercises"""
        if not lesson.parrot_exercises:
            self.display.warning("No exercises available for this lesson")
            return

        self.display.menu(
            "Available Exercises:",
            [f"{ex.exercise_type} - {ex.instructions[:50]}..."
             for ex in lesson.parrot_exercises]
        )

        choice = self.display.prompt(
            "Enter exercise number (or 'a' for all):").strip()

        if choice.lower() == 'a':
            # Do all exercises
            for i, exercise in enumerate(lesson.parrot_exercises, 1):
                self.display.info(f"Exercise {i} of {len(lesson.parrot_exercises)}")
                phrases = lesson.get_phrases_by_ids(exercise.phrase_ids)
                self.exercise_runner.run_exercise(exercise.exercise_type, phrases)
                self.progress.mark_exercise_complete(
                    lesson.lesson_id,
                    lesson.language,
                    exercise.exercise_type
                )

                if i < len(lesson.parrot_exercises):
                    cont = self.display.prompt("Continue to next exercise? (y/n):").strip().lower()
                    if cont != 'y':
                        break
        else:
            # Do specific exercise
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(lesson.parrot_exercises):
                    exercise = lesson.parrot_exercises[idx]
                    phrases = lesson.get_phrases_by_ids(exercise.phrase_ids)
                    self.exercise_runner.run_exercise(exercise.exercise_type, phrases)
                    self.progress.mark_exercise_complete(
                        lesson.lesson_id,
                        lesson.language,
                        exercise.exercise_type
                    )
                else:
                    self.display.error("Invalid exercise number")
            except ValueError:
                self.display.error("Invalid input")

        self.progress.update_completion(
            lesson.lesson_id,
            lesson.language,
            len(lesson.parrot_exercises),
            len(lesson.phrases)
        )

    def review_phrases(self, lesson: Lesson) -> None:
        """Review specific phrases"""
        self.display.subheader("All Phrases in This Lesson")

        for i, phrase in enumerate(lesson.phrases, 1):
            self.display.phrase_compact(phrase, i)

        choice = self.display.prompt(
            "\nEnter phrase number to review (or 'b' for back):"
        ).strip()

        if choice.lower() == 'b':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(lesson.phrases):
                phrase = lesson.phrases[idx]
                self.display.phrase_full(phrase)

                mark = self.display.prompt(
                    "Mark as mastered? (y/n):"
                ).strip().lower()

                if mark == 'y':
                    self.progress.mark_phrase_mastered(
                        lesson.lesson_id,
                        lesson.language,
                        phrase.id
                    )
                    self.display.success("Phrase marked as mastered!")

                self.progress.update_completion(
                    lesson.lesson_id,
                    lesson.language,
                    len(lesson.parrot_exercises),
                    len(lesson.phrases)
                )
            else:
                self.display.error("Invalid phrase number")
        except ValueError:
            self.display.error("Invalid input")

    def view_progress(self, lesson: Lesson) -> None:
        """View progress for current lesson"""
        prog = self.progress.get_lesson_progress(lesson.lesson_id)

        self.display.header(f"Progress: {lesson.title}")

        if not prog:
            self.display.info("No progress recorded yet for this lesson")
            return

        print(f"{self.display.color('Overall Completion:', 'bold')} {prog.completion_percentage:.1f}%")
        print(f"{self.display.color('Last Practiced:', 'bold')} {prog.last_practiced}")

        print(f"\n{self.display.color('Completed Exercises:', 'green')}")
        if prog.completed_exercises:
            for ex in prog.completed_exercises:
                print(f"  ✓ {ex}")
        else:
            print(f"  {self.display.color('None yet', 'dim')}")

        print(f"\n{self.display.color('Mastered Phrases:', 'green')} {len(prog.phrases_mastered)}/{len(lesson.phrases)}")

        self.display.progress_indicator(
            len(prog.phrases_mastered),
            len(lesson.phrases)
        )

    def lesson_loop(self, lesson: Lesson) -> None:
        """Main loop for interacting with a lesson"""
        self.current_lesson = lesson

        while True:
            choice = self.show_lesson_menu(lesson)

            if choice == '1':
                self.study_all_phrases(lesson)
            elif choice == '2':
                self.practice_exercises(lesson)
            elif choice == '3':
                self.review_phrases(lesson)
            elif choice == '4':
                self.view_progress(lesson)
            elif choice == '5':
                break
            else:
                self.display.error("Invalid choice")

            self.display.prompt("\nPress Enter to continue...")

    def run(self) -> None:
        """Main application loop"""
        self.show_welcome()

        while True:
            # Select language
            language = self.select_language()
            if not language:
                self.display.info("Goodbye! Keep parroting! 🦜")
                break

            # Select lesson
            lesson_file = self.select_lesson(language)
            if not lesson_file:
                continue

            # Load lesson
            try:
                lesson = self.loader.load_lesson(language, lesson_file)
                self.lesson_loop(lesson)
            except FileNotFoundError as e:
                self.display.error(f"Lesson not found: {e}")
            except Exception as e:
                self.display.error(f"Error loading lesson: {e}")


def main():
    """Entry point for CLI"""
    try:
        app = TutiParrotCLI()
        app.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 🦜")
        sys.exit(0)


if __name__ == "__main__":
    main()
