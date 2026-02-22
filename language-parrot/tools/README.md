# Tuti Parrot CLI Tool 🦜

Interactive command-line application for learning languages through parroting.

## Features

- 📚 **Load lessons** from structured JSON curriculum
- 🎯 **5 exercise types**: listen-repeat, shadowing, pattern-drill, progressive-speed, memory-recall
- 📊 **Progress tracking**: Automatically saves your learning progress
- 🎨 **Beautiful terminal UI**: Color-coded, easy to navigate
- 🦜 **Parroting methodology**: Learn through repetition and pattern recognition

## Installation

### Quick Start

```bash
# Navigate to the tools directory
cd language-parrot/tools

# Run directly with Python
python -m tuti_parrot.cli
```

### Install as Command

```bash
# Install in development mode
pip install -e .

# Now you can run from anywhere
tuti
```

## Usage

### Starting the App

```bash
python -m tuti_parrot.cli
# or if installed:
tuti
```

### Navigation

1. **Select Language** - Choose which language to learn (e.g., Farsi)
2. **Select Lesson** - Pick a lesson (shows progress percentages)
3. **Choose Activity**:
   - Study all phrases - Read through every phrase with full details
   - Practice exercises - Do structured parroting exercises
   - Review specific phrases - Look up individual phrases
   - View progress - See your completion stats

### Exercise Types

#### 1. Listen-Repeat
Listen to native pronunciation (or read), then repeat aloud multiple times.
- **Goal**: Train ear and mouth for new sounds

#### 2. Shadowing
Speak simultaneously with the audio/reading - don't wait, overlap!
- **Goal**: Develop natural rhythm and flow

#### 3. Pattern Drill
Notice grammatical patterns, practice variations systematically.
- **Goal**: Internalize language structure

#### 4. Progressive Speed
Start very slow, gradually increase to conversational speed over 5 reps.
- **Goal**: Build muscle memory

#### 5. Memory Recall
Recall phrases from memory without looking, self-assess accuracy.
- **Goal**: Test retention, move to unconscious competence

### Progress Tracking

Progress is automatically saved to `~/.tuti_parrot_progress.json`

Tracks:
- ✓ Completed exercises per lesson
- ✓ Phrases marked as mastered
- ✓ Overall completion percentage
- ✓ Last practice date

## Project Structure

```
tuti_parrot/
├── __init__.py           # Package initialization
├── cli.py                # Main CLI application
├── lesson_loader.py      # Loads lessons from JSON
├── exercises.py          # Exercise implementations
├── progress.py           # Progress tracking
└── display.py            # Terminal UI utilities
```

## Requirements

**Python 3.8+** - No external dependencies!

Uses only Python standard library:
- `json` - Lesson data parsing
- `pathlib` - File handling
- `dataclasses` - Data structures
- `datetime` - Progress timestamps
- `time` - Exercise timing

## Development

### Adding New Exercise Types

Edit `exercises.py` and add a new method to `ExerciseRunner`:

```python
def my_new_exercise(self, phrases: List[Phrase]) -> None:
    """My custom exercise"""
    self.display.exercise_instructions(
        "my-exercise-type",
        "Instructions for learners..."
    )
    # Implementation here
```

Then add to the `exercise_map` in `run_exercise()`.

### Custom Progress Location

```python
from tuti_parrot.progress import ProgressTracker

tracker = ProgressTracker(progress_file="/custom/path/progress.json")
```

### Using as a Library

```python
from tuti_parrot.lesson_loader import LessonLoader
from tuti_parrot.exercises import ExerciseRunner

# Load a lesson
loader = LessonLoader()
lesson = loader.load_lesson('farsi', '01-essential-greetings.json')

# Run an exercise
runner = ExerciseRunner()
runner.listen_repeat(lesson.phrases[:3])
```

## Examples

### Study Session

```bash
$ tuti

🦜 طوطی خل - Tuti Parrot 🦜
Learn Languages Through Parroting

Select a Language:
  1. farsi

Enter language number: 1

Lessons for FARSI:
  1. Essential Greetings and First Words [beginner]
  2. The Poetic Heart - Beautiful Everyday Expressions [beginner]

Enter lesson number: 1

📚 Essential Greetings and First Words

What would you like to do?
  1. Study all phrases
  2. Practice exercises
  3. Review specific phrases
  4. View progress
  5. Back to lesson selection

Enter your choice: 2

Available Exercises:
  1. listen-repeat - Listen to each phrase 3 times...
  2. shadowing - Play the audio and speak simultaneously...
  ...
```

## Keyboard Shortcuts

- **Enter** - Continue/Next
- **q** - Quit current activity
- **b** - Back to previous menu
- **m** - Mark phrase as mastered
- **y/n** - Yes/No responses

## Tips for Effective Parroting

1. **Don't be shy** - Say it out loud! Whisper-learning doesn't work
2. **Exaggerate sounds** - Especially unfamiliar phonemes
3. **Use emotion** - Persian is dramatic - lean into it!
4. **Repeat in chunks** - Word → Phrase → Sentence
5. **Practice daily** - Even 10 minutes > 1 hour once a week
6. **Review mastered** - Come back to "mastered" phrases periodically

## Future Enhancements

- [ ] Audio playback integration
- [ ] Spaced repetition scheduling
- [ ] Voice recording for comparison
- [ ] Rich terminal UI with `rich` library
- [ ] Export progress reports
- [ ] Custom lesson creation tools
- [ ] Multi-user support

## Troubleshooting

### Can't find lessons?

Make sure you're running from the `tools/` directory, or that the curriculum path is correct:

```python
loader = LessonLoader(curriculum_path="/path/to/curriculum")
```

### Progress not saving?

Check file permissions for `~/.tuti_parrot_progress.json`

### Unicode display issues?

Ensure your terminal supports UTF-8 encoding for Perso-Arabic script display.

## License

MIT

## Contributing

Found a bug? Want to add a feature? PRs welcome!

---

**Happy parroting!** 🦜 Keep practicing, keep repeating, keep learning!
