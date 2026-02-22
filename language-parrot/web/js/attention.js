/**
 * Attention Mechanism Visualization
 * Highlights patterns in Farsi phrases like LLM attention weights
 */

// Pattern definitions - what learners should focus on
const FARSI_PATTERNS = {
    possessives: {
        name: "Possessive Suffixes",
        description: "Adding م (my), ت (your), ش (his/her) to words shows ownership",
        examples: [
            { word: "دلم", parts: ["دل", "م"], meanings: ["heart", "my"], attention: [3, 5] },
            { word: "دستت", parts: ["دست", "ت"], meanings: ["hand", "your"], attention: [3, 5] },
            { word: "جانم", parts: ["جان", "م"], meanings: ["soul", "my"], attention: [3, 5] },
            { word: "دمت", parts: ["دم", "ت"], meanings: ["breath", "your"], attention: [3, 5] },
            { word: "عزیزم", parts: ["عزیز", "م"], meanings: ["dear", "my"], attention: [3, 5] }
        ],
        focusPattern: /م$|ت$|ش$/
    },

    khub_pattern: {
        name: "خوب (Good) Pattern",
        description: "خوب (good) combines with different endings to form phrases",
        examples: [
            { word: "خوب", parts: ["خوب"], meanings: ["good"], attention: [5] },
            { word: "خوبم", parts: ["خوب", "م"], meanings: ["good", "I-am"], attention: [5, 4] },
            { word: "خوب هستید", parts: ["خوب", " ", "هستید"], meanings: ["good", " ", "are-you"], attention: [5, 0, 4] }
        ],
        focusPattern: /خوب/
    },

    del_heart: {
        name: "دل (Heart) Expressions",
        description: "Persians express emotions through the heart (دل), not the head",
        examples: [
            { word: "دلم", parts: ["دل", "م"], meanings: ["heart", "my"], attention: [5, 4] },
            { word: "دلم می‌خواد", parts: ["دلم", " ", "می‌خواد"], meanings: ["my-heart", " ", "wants"], attention: [5, 0, 4] },
            { word: "دلم برات تنگ شده", parts: ["دلم", " ", "برات", " ", "تنگ", " ", "شده"],
              meanings: ["my-heart", " ", "for-you", " ", "narrow", " ", "has-become"],
              attention: [5, 0, 4, 0, 5, 0, 3] }
        ],
        focusPattern: /دل/
    },

    verb_present: {
        name: "Present Continuous (می)",
        description: "می before verbs indicates ongoing action (like -ing in English)",
        examples: [
            { word: "می‌خواد", parts: ["می", "‌", "خواد"], meanings: ["is/am", " ", "wanting"], attention: [5, 0, 3] },
            { word: "می‌کنم", parts: ["می", "‌", "کنم"], meanings: ["is/am", " ", "doing"], attention: [5, 0, 3] }
        ],
        focusPattern: /می‌/
    },

    polite_forms: {
        name: "Formal/Polite Forms",
        description: "Persian has formal (شما/هستید) and informal (تو/هستی) you",
        examples: [
            { word: "شما", parts: ["شما"], meanings: ["you-formal"], attention: [5] },
            { word: "هستید", parts: ["هست", "ید"], meanings: ["are", "you-formal"], attention: [3, 5] },
            { word: "ببخشید", parts: ["ببخش", "ید"], meanings: ["forgive", "you-formal"], attention: [4, 5] }
        ],
        focusPattern: /شما|ید$/
    },

    body_metaphors: {
        name: "Body Part Metaphors",
        description: "Persians use body parts poetically: hand (دست), breath (دم), eye (چشم)",
        examples: [
            { word: "دستت درد نکنه", parts: ["دستت", " ", "درد", " ", "نکنه"],
              meanings: ["your-hand", " ", "pain", " ", "not-do"],
              attention: [5, 0, 4, 0, 3] },
            { word: "دمت گرم", parts: ["دمت", " ", "گرم"],
              meanings: ["your-breath", " ", "warm"],
              attention: [5, 0, 5] },
            { word: "چشم", parts: ["چشم"], meanings: ["eye (yes!)"], attention: [5] }
        ],
        focusPattern: /دست|دم|چشم/
    },

    dramatic_expressions: {
        name: "Dramatic Expressions",
        description: "Hyperbolic expressions are normal in Persian: sacrifice, narrow heart, etc.",
        examples: [
            { word: "قربان شما", parts: ["قربان", " ", "شما"],
              meanings: ["sacrifice-of", " ", "you"],
              attention: [5, 0, 4] },
            { word: "تنگ شده", parts: ["تنگ", " ", "شده"],
              meanings: ["narrow", " ", "has-become"],
              attention: [5, 0, 4] },
            { word: "عشق است", parts: ["عشق", " ", "است"],
              meanings: ["love", " ", "is"],
              attention: [5, 0, 3] }
        ],
        focusPattern: /قربان|تنگ|عشق/
    }
};

// Get all phrases from lessons for pattern analysis
function getAllPhrasesForPattern(pattern) {
    return pattern.examples;
}

// Generate attention visualization HTML
function generateAttentionHTML(pattern) {
    const patternData = FARSI_PATTERNS[pattern];
    if (!patternData) return '';

    let html = `
        <div class="attention-view pattern-reveal">
            <div class="attention-header">
                <h3 class="attention-title">${patternData.name}</h3>
                <p class="attention-subtitle">${patternData.description}</p>
            </div>

            <div class="attention-legend">
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(212, 165, 116, 0.1);"></div>
                    <span>Low Focus</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(212, 165, 116, 0.5);"></div>
                    <span>Medium Focus</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: var(--persian-gold);"></div>
                    <span>High Focus - KEY PATTERN!</span>
                </div>
            </div>

            <div class="attention-phrases">
    `;

    // Generate phrase displays with attention highlighting
    patternData.examples.forEach((example, idx) => {
        html += `
            <div class="attention-phrase">
                <div class="attention-phrase-native">
        `;

        // Build the word with attention highlighting
        example.parts.forEach((part, partIdx) => {
            const attentionLevel = example.attention[partIdx] || 0;
            const meaning = example.meanings[partIdx] || '';

            html += `<span class="attention-word attention-${attentionLevel}" data-tooltip="${meaning}">${part}</span>`;
        });

        html += `
                </div>
                <div class="attention-phrase-trans">
                    ${example.meanings.join(' → ')}
                </div>
            </div>
        `;
    });

    html += `
            </div>

            <div class="pattern-explanation">
                <h3>🎯 What to Parrot</h3>
                <p>Focus on the <strong>highlighted parts</strong> (darker gold = more important).
                These are the building blocks that appear in many phrases.
                Master these patterns, and you'll recognize them everywhere!</p>
            </div>

            <div class="pattern-connections">
                <div class="connection-title">All Examples with This Pattern</div>
                <div class="connection-grid">
                    ${patternData.examples.map(ex =>
                        `<div class="connection-item">${ex.word}</div>`
                    ).join('')}
                </div>
            </div>
        </div>
    `;

    return html;
}

// Show pattern comparison - before/after understanding the pattern
function showPatternComparison(pattern) {
    const patternData = FARSI_PATTERNS[pattern];
    if (!patternData) return;

    const container = document.getElementById('pattern-comparison');
    if (!container) return;

    let html = `
        <div class="comparison-view">
            <div class="comparison-column">
                <h4>❌ Without Pattern Recognition</h4>
                <div class="attention-phrase">
                    <div class="attention-phrase-native">
    `;

    // Show first example without highlighting (all same attention)
    const firstExample = patternData.examples[0];
    firstExample.parts.forEach(part => {
        html += `<span class="attention-word attention-2">${part}</span>`;
    });

    html += `
                    </div>
                    <p style="margin-top: 1rem; color: var(--persian-gray); font-size: 0.9rem;">
                        Looks like random sounds to memorize 😰
                    </p>
                </div>
            </div>

            <div class="comparison-column">
                <h4>✅ With Pattern Recognition</h4>
                <div class="attention-phrase">
                    <div class="attention-phrase-native">
    `;

    // Show with proper attention highlighting
    firstExample.parts.forEach((part, idx) => {
        const attentionLevel = firstExample.attention[idx] || 0;
        const meaning = firstExample.meanings[idx] || '';
        html += `<span class="attention-word attention-${attentionLevel}" data-tooltip="${meaning}">${part}</span>`;
    });

    html += `
                    </div>
                    <p style="margin-top: 1rem; color: var(--persian-gray); font-size: 0.9rem;">
                        Clear structure: recognize the pattern! 🎯
                    </p>
                </div>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Initialize attention visualization
function initAttentionView() {
    const container = document.getElementById('attention-content');
    if (!container) return;

    // Create pattern selector buttons
    let html = `
        <div class="attention-header">
            <h2 class="page-title">Pattern Attention Visualization</h2>
            <p class="attention-subtitle" style="text-align: center; font-size: 1.1rem; margin-bottom: 2rem;">
                Like how AI attention highlights important parts, see which patterns to focus on for faster learning
            </p>
        </div>

        <div class="pattern-selector">
    `;

    Object.keys(FARSI_PATTERNS).forEach((key, idx) => {
        const pattern = FARSI_PATTERNS[key];
        html += `
            <button class="pattern-btn ${idx === 0 ? 'active' : ''}"
                    onclick="selectPattern('${key}')">
                ${pattern.name}
            </button>
        `;
    });

    html += `
        </div>
        <div id="pattern-display"></div>
        <div id="pattern-comparison"></div>
    `;

    container.innerHTML = html;

    // Show first pattern by default
    selectPattern(Object.keys(FARSI_PATTERNS)[0]);
}

// Select and display a pattern
function selectPattern(patternKey) {
    // Update button states
    document.querySelectorAll('.pattern-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event?.target?.classList?.add('active') ||
        document.querySelector(`[onclick="selectPattern('${patternKey}')"]`)?.classList.add('active');

    // Display pattern
    const display = document.getElementById('pattern-display');
    if (display) {
        display.innerHTML = generateAttentionHTML(patternKey);
    }

    // Show comparison
    showPatternComparison(patternKey);
}

// Export functions
window.initAttentionView = initAttentionView;
window.selectPattern = selectPattern;
