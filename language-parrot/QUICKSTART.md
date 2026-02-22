# 🦜 Tuti Parrot - Quick Start Guide

Welcome to Tuti Parrot (طوطی خل) - Learn Farsi through parroting!

## Two Ways to Learn

### 🌐 Option 1: Web Interface (Recommended)

**Clean, beautiful interface inspired by Persian art**

```bash
cd web
./start.sh
```

Or manually:
```bash
cd web
python serve.py
```

Then open **http://localhost:8080** in your browser.

**Features:**
- ✨ Clean minimal design with Persian aesthetics
- 📚 Browse lessons with progress tracking
- 🎯 Mark phrases as mastered
- 📱 Works on phone, tablet, desktop
- 🎨 Subtle geometric patterns inspired by Persian tiles
- 🦜 All 20 Farsi phrases with cultural context
- 🔍 **Pattern visualization** - See language patterns highlighted like AI attention!

---

### 🖥️ Option 2: Command Line

**Interactive terminal application**

```bash
cd tools
python -m tuti_parrot.cli
```

Or use the launcher:
```bash
cd tools
./run.sh
```

**Features:**
- 🎯 Guided parroting exercises
- 📊 Progress tracking
- 🎨 Colorful terminal UI
- 🦜 5 exercise modes (listen-repeat, shadowing, pattern-drill, progressive-speed, memory-recall)

---

## What You'll Learn

### 📚 Lesson 1: Essential Greetings (10 phrases)
- سلام (salaam) - Hello
- ممنون (mamnun) - Thank you
- خداحافظ (khodahafez) - Goodbye
- Plus 7 more essential phrases

### ❤️ Lesson 2: The Poetic Heart (10 phrases)
- قربان شما (ghorbane shoma) - I sacrifice myself for you (endearment)
- دلم برات تنگ شده (delam barat tang shode) - I miss you (lit: my heart has become narrow)
- جانم (jaanam) - My soul (term of endearment)
- Plus 7 more poetic everyday expressions

**Every phrase includes:**
- Native Perso-Arabic script
- Transliteration for pronunciation
- Literal translation (to see the poetry!)
- Cultural context and usage notes
- Word-by-word breakdown
- When and how to use it
- Parroting instructions

---

## The Parroting Method

Learn like a parrot! 🦜 Through repetition and mimicry:

1. **Listen-Repeat** 👂 - Train your ears and mouth
2. **Shadowing** 🗣️ - Speak along with native audio
3. **Pattern Drill** 🔍 - Internalize grammar structures
4. **Progressive Speed** ⚡ - Build muscle memory
5. **Memory Recall** 🧠 - Test your retention

---

## System Requirements

- **Python 3.8+** (already have it!)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)
- **No installation required** - just run and learn!

---

## File Structure

```
language-parrot/
├── curriculum/farsi/lessons/    # 2 JSON lessons, 20 phrases
├── web/                          # Web interface (HTML/CSS/JS)
└── tools/tuti_parrot/           # CLI app (Python)
```

---

## Quick Tips

### Web Interface
- Click lessons to explore phrases
- Click "Mark as Mastered" to track progress
- View your progress in the Progress tab
- All data saved in your browser

### CLI App
- Use numbers to navigate menus
- Press 'm' to mark phrases as mastered
- Press 'q' to go back
- Progress saved to `~/.tuti_parrot_progress.json`

---

## Need Help?

- **Web docs**: `web/README.md`
- **CLI docs**: `tools/README.md`
- **Curriculum info**: `README.md`

---

**طوطی خل - Be a silly parrot! Don't be afraid to repeat, make mistakes, and practice. That's how we learn!** 🦜
