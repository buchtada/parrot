# Language Parrot (طوطی خل - Tuti Khol)

> Learn languages through the power of parroting - repetition, mimicry, and pattern recognition

## Philosophy

Language acquisition mirrors how parrots learn: through repeated listening, mimicking sounds, and gradually internalizing patterns. This curriculum embraces this natural learning method, focusing on:

1. **Sound First**: Hear native pronunciation before reading
2. **Pattern Recognition**: Notice repeating structures in language
3. **Emotional Context**: Learn language as it's actually spoken - with feeling, drama, and cultural context
4. **Frequency-Based**: Start with the most useful, commonly-used phrases
5. **Poetic Soul**: Preserve the beauty and cultural essence of each language

## Why "Tuti Khol" (Silly Parrot)?

**طوطی خل** (tuti khol) means "silly parrot" in Farsi - a playful acknowledgment that language learning starts with imitation. Just as parrots repeat what they hear, language learners build fluency through repetition and mimicry. We embrace the "silliness" of not understanding everything at first - just repeat, parrot, and gradually the patterns emerge!

## Current Languages

### Farsi (فارسی)

Iranian Farsi curriculum focusing on:
- Survival phrases for immediate communication
- The poetic and dramatic nature of Persian speech
- Cultural context and emotional expression
- High-frequency patterns in everyday conversation

**Available Lessons:**
1. `01-essential-greetings.json` - Core greetings, thank yous, basic responses
2. `02-poetic-heart.json` - Beautiful everyday expressions that reveal Farsi's dramatic soul

## Curriculum Structure

### Data Format

All lessons are structured as JSON files following the schema in `/schemas/lesson-schema.json`

Each lesson includes:
- **Phrases**: Native script, transliteration, translation, literal translation
- **Audio References**: Paths to native speaker pronunciations
- **Frequency Ranking**: How common the phrase is in everyday use
- **Cultural Notes**: Context about when/how/why phrases are used
- **Pattern Breakdown**: Linguistic structure and grammar patterns
- **Parrot Method**: Specific instructions for practicing each phrase
- **Exercises**: Structured parroting activities

### Parroting Methodology

The curriculum uses five core parroting exercises:

#### 1. Listen-Repeat
Listen to native audio multiple times, then repeat aloud. Focus on exact sound reproduction.

**Goal**: Train your mouth and ears to recognize and produce unfamiliar sounds

#### 2. Shadowing
Play audio and speak simultaneously with the native speaker. Match their pace, rhythm, and emotion.

**Goal**: Develop natural flow and prosody

#### 3. Pattern Drill
Identify repeating grammatical patterns and practice variations systematically.

**Goal**: Internalize language structure through repetition

#### 4. Progressive Speed
Start very slowly, gradually increase speed over multiple repetitions until reaching conversational pace.

**Goal**: Build muscle memory for complex sound combinations

#### 5. Memory Recall
Recall and produce phrases from memory without looking at text.

**Goal**: Move from conscious to unconscious competence

## Directory Structure

```
language-parrot/
├── README.md                          # This file
├── schemas/
│   └── lesson-schema.json             # JSON schema for lessons
├── curriculum/
│   └── farsi/
│       ├── lessons/
│       │   ├── 01-essential-greetings.json
│       │   ├── 02-poetic-heart.json
│       │   └── ... (more lessons)
│       └── audio/
│           └── ... (audio files referenced in lessons)
├── web/                               # Web interface
│   ├── index.html                     # Main page
│   ├── css/style.css                  # Persian-inspired styling
│   ├── js/app.js                      # Interactive functionality
│   ├── serve.py                       # Development server
│   └── README.md                      # Web app documentation
└── tools/                             # CLI application
    ├── tuti_parrot/
    │   ├── cli.py                     # Main application
    │   ├── lesson_loader.py           # Lesson loading
    │   ├── exercises.py               # Exercise implementations
    │   ├── progress.py                # Progress tracking
    │   └── display.py                 # Terminal UI
    ├── run.sh                         # Quick launcher
    ├── setup.py                       # Installation script
    └── README.md                      # Tool documentation
```

## Using the Curriculum

### 🌐 Web Interface (Recommended!)

**Clean, beautiful web interface inspired by Persian art:**

```bash
cd web
python serve.py
# Then open http://localhost:8080
```

Features:
- ✨ Clean, minimal Persian-inspired design
- 📚 Browse and study all lessons
- 🎯 Mark phrases as mastered
- 📊 Track your progress
- 🦜 Parroting exercise guides
- 📱 Responsive (works on phone, tablet, desktop)

See [web/README.md](web/README.md) for details.

### 🖥️ CLI App (Terminal)

**Interactive command-line application:**

```bash
cd tools
python -m tuti_parrot.cli
```

Features:
- 🎯 Guided exercise sessions
- 📊 Automatic progress tracking
- 🎨 Colorful terminal UI
- 🦜 All 5 parroting exercise types

See [tools/README.md](tools/README.md) for full documentation.

### For Self-Study (Manual)

1. **Start with Lesson 01**: Begin with essential greetings
2. **Read the lesson description** and learning objectives
3. **For each phrase**:
   - Read the cultural notes and context first
   - Listen to the audio (when available)
   - Follow the parrot_method instructions
   - Practice the word breakdown
4. **Complete the parrot_exercises** at the end of each lesson
5. **Move to next lesson** when you can recall 80%+ of phrases from memory

### For Developers

The JSON structure is designed to be machine-readable and can be used to:
- Build web or mobile apps
- Generate flashcards
- Create spaced repetition systems
- Integrate with audio players
- Track learner progress

Example: Reading a lesson programmatically

```python
import json

with open('curriculum/farsi/lessons/01-essential-greetings.json', 'r', encoding='utf-8') as f:
    lesson = json.load(f)

# Access phrases
for phrase in lesson['phrases']:
    print(f"{phrase['native']} ({phrase['transliteration']})")
    print(f"  → {phrase['translation']}")
    print(f"  Cultural note: {phrase['cultural_notes']}\n")
```

### For Contributors

To add a new lesson:

1. Follow the schema in `/schemas/lesson-schema.json`
2. Include native script, transliteration, and translation
3. Add cultural context - WHY is this phrase used?
4. Include pattern breakdown for grammar learning
5. Add parrot method instructions specific to challenging sounds
6. Create varied exercises that reinforce different skills

## Farsi Learning Notes

### The Poetic Nature of Persian

Persian (Farsi) is inherently poetic and dramatic. Everyday speech is filled with:
- **Hyperbolic expressions**: "I sacrifice myself for you" (قربان شما)
- **Heart-centered emotion**: "My heart wants" instead of "I want" (دلم می‌خواد)
- **Metaphorical language**: "My heart has become narrow" = "I miss you" (دلم تنگ شده)
- **Blessing-based gratitude**: "May your hand not hurt" = "Thank you" (دستت درد نکنه)

This isn't flowery language - it's how Iranians communicate daily. The curriculum preserves this essence rather than reducing Farsi to dry vocabulary lists.

### Pronunciation Notes

Key Farsi sounds that don't exist in English:
- **خ (kh)**: Guttural sound from back of throat (like German 'ch')
- **غ (gh)**: Similar to 'kh' but voiced
- **ق (gh)**: Deep guttural 'g' sound
- **ع (')**: Glottal stop (like Arabic ain)

Don't worry about perfection - parrot what you hear, and these sounds will develop naturally with practice.

### Cultural Context: Ta'arof

Many phrases in the curriculum reflect **تعارف** (ta'arof) - the Persian cultural practice of ritual politeness and humility. Iranians will:
- Downplay their efforts (قابلی نداره)
- Offer extravagant praise (قربان شما)
- Refuse offers multiple times before accepting (part of polite dance)

Understanding ta'arof is essential to understanding Persian communication.

## Future Development

### ✅ Completed
- [x] Structured JSON curriculum (2 Farsi lessons, 20 phrases)
- [x] Interactive CLI application with progress tracking
- [x] Beautiful web interface with Persian-inspired design
- [x] 5 parroting exercise methodologies
- [x] Cultural context and pattern breakdowns

### 🚧 Planned
- [ ] Audio files for all phrases (native speaker recordings)
- [ ] Additional Farsi lessons (questions, numbers, food, family, travel, etc.)
- [ ] Other languages using the same methodology
- [ ] Interactive exercise modes in web app (practice in-browser)
- [ ] Spaced repetition algorithm
- [ ] Voice recording and comparison
- [ ] Mobile app (React Native or Flutter)
- [ ] Community contributions

## Quick Reference: Fun Farsi Phrases

- **طوطی** (tuti) = parrot
- **طوطی خل** (tuti khol) = silly parrot 🦜
- **غاز خل** (ghaz-e khol) = silly goose
- **خل** (khol) = silly/crazy (affectionate)

## License

MIT

## Contributing

We welcome contributions! Whether it's:
- Adding new lessons
- Recording audio
- Fixing transliterations
- Sharing cultural insights
- Building tools that use this curriculum

Please feel free to open issues or submit pull requests.

---

**Remember**: Learning a language isn't about being perfect - it's about being a طوطی خل (silly parrot) who isn't afraid to repeat, practice, and gradually find their voice. 🦜
