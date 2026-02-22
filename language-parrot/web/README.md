# Tuti Parrot Web Interface 🦜

A clean, beautiful web interface for learning Farsi through parroting, inspired by Persian art and design.

## Design Philosophy

**Clean & Minimal**
- Generous white space
- Simple, elegant typography
- Focus on content over decoration

**Persian-Inspired**
- Subtle geometric patterns (Persian tile motifs)
- Color palette inspired by Persian art (deep blues, teals, gold, cream)
- Respect for Persian script and calligraphy

**User-Focused**
- Intuitive navigation
- Clear visual hierarchy
- Smooth transitions and interactions

## Features

✨ **Beautiful Interface**
- Clean, minimal design
- Persian-inspired color palette
- Responsive layout (works on mobile, tablet, desktop)

📚 **Lesson Browser**
- View all available Farsi lessons
- See phrase count and exercise count
- Track mastery progress per lesson

🎯 **Phrase Study**
- Native Perso-Arabic script
- Transliteration for pronunciation
- Literal translations to show structure
- Rich cultural context
- Word-by-word breakdowns
- Parroting instructions

📊 **Progress Tracking**
- Mark phrases as mastered
- View overall progress percentage
- Track progress by lesson
- Data saved in browser (localStorage)

🦜 **Parroting Exercises**
- 5 exercise types displayed with instructions
- Clear methodology explanation

## Running the App

### Option 1: Python Server (Recommended)

```bash
cd web
python serve.py
```

Then open http://localhost:8080 in your browser.

### Option 2: Python Built-in Server

```bash
cd web
python -m http.server 8080
```

Then open http://localhost:8080 in your browser.

### Option 3: Any Web Server

Point any web server to the `web/` directory. The app is pure HTML/CSS/JS with no build step required. Use port 8080 or any available port.

## Project Structure

```
web/
├── index.html          # Main HTML structure
├── css/
│   └── style.css       # Persian-inspired styling
├── js/
│   └── app.js          # Lesson loading and interaction
├── serve.py            # Simple Python server
└── README.md           # This file
```

## Technical Details

### Pure Frontend
- No frameworks or libraries required
- Vanilla JavaScript (ES6+)
- Modern CSS (Grid, Flexbox, CSS Variables)
- Loads lessons directly from JSON files

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Requires ES6+ support
- Uses `fetch()` API for loading lessons

### Data Storage
- Progress saved in `localStorage`
- No backend required
- Data persists between sessions

### Lesson Loading
The app loads lessons from:
```
../curriculum/farsi/lessons/
├── 01-essential-greetings.json
└── 02-poetic-heart.json
```

## Design Details

### Color Palette

```css
--persian-blue: #1a4d7a      /* Primary blue */
--persian-teal: #2d8b8f      /* Accent teal */
--persian-gold: #d4a574      /* Highlight gold */
--persian-cream: #f9f7f4     /* Background cream */
--persian-dark: #1a1a1a      /* Text dark */
--persian-gray: #6b6b6b      /* Secondary text */
```

### Typography

- **Body Text**: Inter (clean, readable sans-serif)
- **Farsi Text**: Scheherazade New (beautiful Persian font)
- **Line Height**: 1.7 for comfortable reading
- **Generous spacing** throughout

### Persian Design Elements

1. **Geometric Patterns**
   - Subtle diagonal grid in background
   - Inspired by Persian tile work
   - Minimal opacity (3%) for subtlety

2. **Card Hover Effects**
   - Left border accent on hover (like Persian miniature frames)
   - Gentle elevation with shadows
   - Smooth transitions

3. **Color Accents**
   - Gold highlights for important elements
   - Blue/teal for interactive elements
   - Cream backgrounds for emphasis areas

## Usage Guide

### 1. Home Page
- See total phrases, lessons, and mastered count
- Learn about the 5 parroting exercises
- Click "Start Learning" to begin

### 2. Lessons View
- Browse all available lessons
- See progress at a glance (mastered/total)
- Click any lesson card to open

### 3. Lesson Detail
- Study each phrase in depth
- Read cultural context and usage notes
- See word-by-word breakdowns
- Mark phrases as mastered
- Review parroting exercises

### 4. Progress View
- See overall completion percentage
- Track progress by individual lesson
- Click lessons to review

## Customization

### Adding New Lessons

Simply add new JSON files to `../curriculum/farsi/lessons/` and update the lesson loading in `js/app.js`:

```javascript
const lesson3 = await fetch('../curriculum/farsi/lessons/03-your-lesson.json').then(r => r.json());
state.lessons = [lesson1, lesson2, lesson3];
```

### Changing Colors

Edit CSS variables in `css/style.css`:

```css
:root {
    --persian-blue: #your-color;
    --persian-teal: #your-color;
    /* etc. */
}
```

### Modifying Layout

The app uses CSS Grid and Flexbox for layout. Main layout sections:
- `.container` - Max width and centering
- `.lessons-grid` - Responsive lesson cards
- `.method-grid` - Responsive method cards
- `.phrase-card` - Individual phrase layout

## Performance

- **Lightweight**: ~50KB total (HTML + CSS + JS)
- **No dependencies**: Pure vanilla code
- **Fast loading**: Minimal resources
- **Responsive**: Smooth on all devices

## Future Enhancements

Potential additions:
- [ ] Audio playback integration
- [ ] Interactive exercise modes (practice in-browser)
- [ ] Spaced repetition scheduling
- [ ] Voice recording and comparison
- [ ] Export progress as PDF
- [ ] Dark mode option
- [ ] More Persian design flourishes (optional animations)

## Browser Developer Tools

To reset progress (clear localStorage):
```javascript
localStorage.removeItem('tutiParrotProgress');
location.reload();
```

## License

MIT

---

**Remember**: This interface is designed to be clean, minimal, and respectful of both Persian culture and the learner's focus. Every element serves the purpose of learning. 🦜
