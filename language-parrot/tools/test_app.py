"""
Quick test to verify the app works
"""

from tuti_parrot.lesson_loader import LessonLoader
from tuti_parrot.display import Display

# Test lesson loading
loader = LessonLoader()
display = Display()

# Check languages
languages = loader.list_languages()
display.success(f"Found {len(languages)} language(s): {', '.join(languages)}")

# Load first Farsi lesson
if 'farsi' in languages:
    lessons = loader.list_lessons('farsi')
    display.success(f"Found {len(lessons)} Farsi lesson(s)")

    if lessons:
        # Load first lesson
        lesson = loader.load_lesson('farsi', lessons[0]['filename'])
        display.header(f"Loaded: {lesson.title}")

        print(f"Level: {lesson.level}")
        print(f"Phrases: {len(lesson.phrases)}")
        print(f"Exercises: {len(lesson.parrot_exercises)}")

        # Show first phrase
        if lesson.phrases:
            display.subheader("First Phrase Example:")
            phrase = lesson.phrases[0]
            print(f"  {display.color(phrase.native, 'green')} ({phrase.transliteration})")
            print(f"  → {phrase.translation}")
            print(f"  {phrase.cultural_notes}")

        display.success("\n✅ All systems working! Run: python -m tuti_parrot.cli")
else:
    display.error("Farsi curriculum not found!")
