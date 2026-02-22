"""
Progress Tracking - Save and load learner progress
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class LessonProgress:
    """Progress for a single lesson"""
    lesson_id: str
    language: str
    completed_exercises: List[str]
    phrases_mastered: List[str]
    last_practiced: str
    completion_percentage: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'LessonProgress':
        """Create from dictionary"""
        return cls(**data)


class ProgressTracker:
    """Tracks and persists learner progress"""

    def __init__(self, progress_file: Optional[Path] = None):
        """
        Initialize progress tracker

        Args:
            progress_file: Path to progress JSON file (defaults to ~/.tuti_parrot_progress.json)
        """
        if progress_file is None:
            self.progress_file = Path.home() / ".tuti_parrot_progress.json"
        else:
            self.progress_file = Path(progress_file)

        self.progress: Dict[str, LessonProgress] = {}
        self._load()

    def _load(self) -> None:
        """Load progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.progress = {
                        k: LessonProgress.from_dict(v)
                        for k, v in data.items()
                    }
            except (json.JSONDecodeError, KeyError):
                # If file is corrupted, start fresh
                self.progress = {}

    def _save(self) -> None:
        """Save progress to file"""
        data = {k: v.to_dict() for k, v in self.progress.items()}
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_lesson_progress(self, lesson_id: str) -> Optional[LessonProgress]:
        """Get progress for a specific lesson"""
        return self.progress.get(lesson_id)

    def mark_exercise_complete(self, lesson_id: str, language: str, exercise_type: str) -> None:
        """Mark an exercise as completed"""
        if lesson_id not in self.progress:
            self.progress[lesson_id] = LessonProgress(
                lesson_id=lesson_id,
                language=language,
                completed_exercises=[],
                phrases_mastered=[],
                last_practiced=datetime.now().isoformat(),
                completion_percentage=0.0
            )

        lesson_prog = self.progress[lesson_id]

        if exercise_type not in lesson_prog.completed_exercises:
            lesson_prog.completed_exercises.append(exercise_type)

        lesson_prog.last_practiced = datetime.now().isoformat()
        self._save()

    def mark_phrase_mastered(self, lesson_id: str, language: str, phrase_id: str) -> None:
        """Mark a phrase as mastered"""
        if lesson_id not in self.progress:
            self.progress[lesson_id] = LessonProgress(
                lesson_id=lesson_id,
                language=language,
                completed_exercises=[],
                phrases_mastered=[],
                last_practiced=datetime.now().isoformat(),
                completion_percentage=0.0
            )

        lesson_prog = self.progress[lesson_id]

        if phrase_id not in lesson_prog.phrases_mastered:
            lesson_prog.phrases_mastered.append(phrase_id)

        lesson_prog.last_practiced = datetime.now().isoformat()
        self._save()

    def update_completion(self, lesson_id: str, language: str, total_exercises: int, total_phrases: int) -> None:
        """Update completion percentage for a lesson"""
        if lesson_id not in self.progress:
            return

        lesson_prog = self.progress[lesson_id]

        completed_ex = len(lesson_prog.completed_exercises)
        mastered_ph = len(lesson_prog.phrases_mastered)

        # Weight: 60% exercises, 40% phrases
        ex_weight = 0.6
        ph_weight = 0.4

        ex_percent = (completed_ex / total_exercises * 100) if total_exercises > 0 else 0
        ph_percent = (mastered_ph / total_phrases * 100) if total_phrases > 0 else 0

        lesson_prog.completion_percentage = (ex_percent * ex_weight) + (ph_percent * ph_weight)
        self._save()

    def get_all_progress(self, language: Optional[str] = None) -> List[LessonProgress]:
        """
        Get all lesson progress, optionally filtered by language

        Args:
            language: Filter by language (e.g., 'farsi')

        Returns:
            List of LessonProgress objects
        """
        if language:
            return [p for p in self.progress.values() if p.language == language]
        return list(self.progress.values())

    def reset_lesson(self, lesson_id: str) -> None:
        """Reset progress for a specific lesson"""
        if lesson_id in self.progress:
            del self.progress[lesson_id]
            self._save()

    def reset_all(self) -> None:
        """Reset all progress"""
        self.progress = {}
        self._save()
