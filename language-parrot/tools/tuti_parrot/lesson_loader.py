"""
Lesson Loader - Handles loading and parsing lesson JSON files
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PhraseBreakdown:
    """Individual word breakdown in a phrase"""
    word: str
    transliteration: str
    meaning: str
    part_of_speech: str


@dataclass
class ParrotMethod:
    """Instructions for parroting a specific phrase"""
    repetitions_suggested: int = 10
    speed: str = "normal"
    focus_sounds: List[str] = field(default_factory=list)


@dataclass
class Phrase:
    """A single phrase in a lesson"""
    id: str
    native: str
    transliteration: str
    translation: str
    literal_translation: Optional[str] = None
    audio_file: Optional[str] = None
    frequency_rank: Optional[int] = None
    formality: str = "neutral"
    usage_context: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    cultural_notes: Optional[str] = None
    breakdown: List[PhraseBreakdown] = field(default_factory=list)
    parrot_method: Optional[ParrotMethod] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'Phrase':
        """Create Phrase from dictionary"""
        # Parse breakdown if present
        breakdown = []
        if 'breakdown' in data:
            breakdown = [
                PhraseBreakdown(**item) for item in data['breakdown']
            ]

        # Parse parrot_method if present
        parrot_method = None
        if 'parrot_method' in data:
            parrot_method = ParrotMethod(**data['parrot_method'])

        return cls(
            id=data['id'],
            native=data['native'],
            transliteration=data['transliteration'],
            translation=data['translation'],
            literal_translation=data.get('literal_translation'),
            audio_file=data.get('audio_file'),
            frequency_rank=data.get('frequency_rank'),
            formality=data.get('formality', 'neutral'),
            usage_context=data.get('usage_context', []),
            patterns=data.get('patterns', []),
            cultural_notes=data.get('cultural_notes'),
            breakdown=breakdown,
            parrot_method=parrot_method
        )


@dataclass
class ParrotExercise:
    """A parroting exercise in a lesson"""
    exercise_type: str
    instructions: str
    phrase_ids: List[str]


@dataclass
class Lesson:
    """A complete lesson with phrases and exercises"""
    lesson_id: str
    language: str
    level: str
    title: str
    description: str
    learning_objectives: List[str]
    phrases: List[Phrase]
    parrot_exercises: List[ParrotExercise]

    @classmethod
    def from_dict(cls, data: Dict) -> 'Lesson':
        """Create Lesson from dictionary"""
        phrases = [Phrase.from_dict(p) for p in data['phrases']]

        exercises = []
        if 'parrot_exercises' in data:
            exercises = [
                ParrotExercise(**ex) for ex in data['parrot_exercises']
            ]

        return cls(
            lesson_id=data['lesson_id'],
            language=data['language'],
            level=data['level'],
            title=data['title'],
            description=data['description'],
            learning_objectives=data['learning_objectives'],
            phrases=phrases,
            parrot_exercises=exercises
        )

    def get_phrase_by_id(self, phrase_id: str) -> Optional[Phrase]:
        """Get a specific phrase by ID"""
        for phrase in self.phrases:
            if phrase.id == phrase_id:
                return phrase
        return None

    def get_phrases_by_ids(self, phrase_ids: List[str]) -> List[Phrase]:
        """Get multiple phrases by their IDs"""
        return [p for p in self.phrases if p.id in phrase_ids]


class LessonLoader:
    """Loads and manages lessons from JSON files"""

    def __init__(self, curriculum_path: Optional[Path] = None):
        """
        Initialize the lesson loader

        Args:
            curriculum_path: Path to curriculum directory (defaults to ../curriculum)
        """
        if curriculum_path is None:
            # Default to curriculum directory relative to this file
            # __file__ is in tools/tuti_parrot/, so we go up 2 levels to language-parrot/
            self.curriculum_path = Path(__file__).parent.parent.parent / "curriculum"
        else:
            self.curriculum_path = Path(curriculum_path)

    def load_lesson(self, language: str, lesson_filename: str) -> Lesson:
        """
        Load a specific lesson

        Args:
            language: Language code (e.g., 'farsi')
            lesson_filename: Lesson filename (e.g., '01-essential-greetings.json')

        Returns:
            Lesson object
        """
        lesson_path = self.curriculum_path / language / "lessons" / lesson_filename

        if not lesson_path.exists():
            raise FileNotFoundError(f"Lesson not found: {lesson_path}")

        with open(lesson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return Lesson.from_dict(data)

    def list_lessons(self, language: str) -> List[Dict[str, str]]:
        """
        List all available lessons for a language

        Args:
            language: Language code (e.g., 'farsi')

        Returns:
            List of dicts with 'filename' and 'title' keys
        """
        lessons_dir = self.curriculum_path / language / "lessons"

        if not lessons_dir.exists():
            return []

        lessons = []
        for lesson_file in sorted(lessons_dir.glob("*.json")):
            try:
                with open(lesson_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                lessons.append({
                    'filename': lesson_file.name,
                    'title': data.get('title', lesson_file.stem),
                    'level': data.get('level', 'unknown'),
                    'lesson_id': data.get('lesson_id', lesson_file.stem)
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return lessons

    def list_languages(self) -> List[str]:
        """
        List all available languages

        Returns:
            List of language codes
        """
        if not self.curriculum_path.exists():
            return []

        languages = []
        for item in self.curriculum_path.iterdir():
            if item.is_dir() and (item / "lessons").exists():
                languages.append(item.name)

        return sorted(languages)
