# Pattern Attention Visualization 🎯

## What Is This?

The Pattern Attention feature visualizes Farsi language patterns similar to how Large Language Models (LLMs) use attention mechanisms. It highlights which parts of phrases you should focus on for faster learning.

## How It Works

### Attention Weights

Just like transformers assign "attention weights" to important tokens, we assign visual intensity to important parts of Farsi phrases:

- **Low intensity** (light gold) = Less important for pattern recognition
- **Medium intensity** (darker gold) = Helpful to notice
- **High intensity** (brightest gold) = **KEY PATTERN** - Master this!

### Visual Learning

Instead of just memorizing whole phrases, you'll see:
- Which suffixes repeat across phrases (possessives: م، ت، ش)
- Which root words are used frequently (خوب، دل، چشم)
- How patterns combine to create meaning
- Connections between related phrases

## Available Patterns

### 1. Possessive Suffixes
**What:** Adding م (my), ت (your), ش (his/her) to words
**Examples:**
- دل**م** (my heart)
- دست**ت** (your hand)
- جان**م** (my soul)

### 2. خوب (Good) Pattern
**What:** خوب (good) combines with different endings
**Examples:**
- خوب (good)
- خوب**م** (I am good)
- خوب **هستید**? (are you good?)

### 3. دل (Heart) Expressions
**What:** Emotions expressed through "heart" not "head"
**Examples:**
- **دل**م می‌خواد (my heart wants = I want)
- **دل**م برات تنگ شده (my heart is narrow = I miss you)

### 4. Present Continuous (می)
**What:** می before verbs = ongoing action (like -ing)
**Examples:**
- **می**‌خواد (is wanting)
- **می**‌کنم (am doing)

### 5. Formal/Polite Forms
**What:** Formal vs informal "you"
**Examples:**
- شما (you - formal)
- هست**ید** (are - you formal)
- ببخش**ید** (excuse me - formal)

### 6. Body Part Metaphors
**What:** Poetic use of body parts
**Examples:**
- **دست**ت درد نکنه (may your hand not hurt = thanks)
- **دم**ت گرم (your breath warm = good job!)
- **چشم** (eye = yes, with pleasure)

### 7. Dramatic Expressions
**What:** Hyperbolic everyday phrases
**Examples:**
- **قربان** شما (sacrifice of you = term of endearment)
- **تنگ** شده (has become narrow = miss you)
- **عشق** است (is love = it's wonderful)

## How to Use

1. **Click "Patterns" in the navigation**
2. **Select a pattern** from the buttons at top
3. **Observe the highlighting**:
   - Darker gold = more important
   - Lighter gold = supportive context
   - Hover over words to see meanings
4. **Study the connections** - see all phrases using this pattern
5. **Compare before/after** - see how pattern recognition helps

## Learning Strategy

### Without Patterns
❌ "دلم برات تنگ شده looks like random sounds to memorize"

### With Patterns
✅ **دل**م **برات** **تنگ** شده
- **دل** (heart) - already know this from دلم می‌خواد!
- **م** (my) - possessive suffix I've seen before!
- **تنگ** (narrow) - new metaphorical word
- **شده** (has become) - verb ending

Suddenly it's not random - it's building blocks you recognize!

## Benefits

1. **Faster Recognition** - See patterns across phrases instantly
2. **Better Retention** - Visual connections create stronger memories
3. **Natural Grammar** - Learn structure through pattern, not rules
4. **Focus Your Energy** - Know exactly what to practice
5. **Build Intuition** - Start predicting new phrases

## Technical Details

### Attention Calculation

Attention weights (0-5) are assigned based on:
- **Frequency** - How often this element appears
- **Structural importance** - Key to grammar/meaning
- **Learning value** - High ROI for pattern recognition
- **Difficulty** - Needs more focus

### Color Coding

```
attention-0: transparent (grammatical glue)
attention-1: rgba(212, 165, 116, 0.1) - minimal focus
attention-2: rgba(212, 165, 116, 0.3) - some focus
attention-3: rgba(212, 165, 116, 0.5) - moderate focus
attention-4: rgba(212, 165, 116, 0.7) - high focus
attention-5: var(--persian-gold) - CRITICAL PATTERN
```

### Inspired by Transformers

Like how BERT or GPT attend to important tokens:
```
Q (Query): What pattern am I learning?
K (Key): Which parts of this phrase matter?
V (Value): What should I remember?
```

Our visualization shows the "attention weights" visually!

## Adding New Patterns

Want to add your own patterns? Edit `js/attention.js`:

```javascript
new_pattern: {
    name: "Your Pattern Name",
    description: "What this pattern teaches",
    examples: [
        {
            word: "فارسی",
            parts: ["فار", "سی"],
            meanings: ["persia", "language"],
            attention: [3, 5]  // 0-5 intensity
        }
    ],
    focusPattern: /regex/
}
```

## Future Enhancements

- [ ] Interactive highlighting (click to explore)
- [ ] Attention heatmaps across entire lessons
- [ ] Self-attention (connections within a phrase)
- [ ] Cross-attention (connections between phrases)
- [ ] Custom pattern creation by users
- [ ] Attention-based quizzing

---

**Remember:** Patterns are your shortcut to fluency. Master 10 patterns, recognize 100 phrases! 🚀
