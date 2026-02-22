// Tuti Parrot Web App
// Clean, simple JavaScript for lesson loading and interaction

// ===================================
// State Management
// ===================================

const state = {
    lessons: [],
    currentLesson: null,
    progress: loadProgress()
};

// ===================================
// Progress Management
// ===================================

function loadProgress() {
    const saved = localStorage.getItem('tutiParrotProgress');
    return saved ? JSON.parse(saved) : {
        masteredPhrases: [],
        completedLessons: []
    };
}

function saveProgress() {
    localStorage.setItem('tutiParrotProgress', JSON.stringify(state.progress));
}

function markPhraseMastered(phraseId) {
    if (!state.progress.masteredPhrases.includes(phraseId)) {
        state.progress.masteredPhrases.push(phraseId);
        saveProgress();
        updateStats();
    }
}

// ===================================
// Navigation
// ===================================

function navigateToView(viewId) {
    // Hide all views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });

    // Show selected view
    document.getElementById(`${viewId}-view`).classList.add('active');

    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.view === viewId) {
            btn.classList.add('active');
        }
    });

    // Load view-specific content
    if (viewId === 'lessons') {
        loadLessonsView();
    } else if (viewId === 'patterns') {
        loadPatternsView();
    } else if (viewId === 'scene') {
        loadSceneBuilderView();
    } else if (viewId === 'progress') {
        loadProgressView();
    }
}

function navigateToLessons() {
    navigateToView('lessons');
}

function navigateToLessonDetail(lessonId) {
    const lesson = state.lessons.find(l => l.lesson_id === lessonId);
    if (lesson) {
        state.currentLesson = lesson;
        renderLessonDetail(lesson);
        navigateToView('lesson-detail');
    }
}

// Setup nav buttons
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        navigateToView(btn.dataset.view);
    });
});

// ===================================
// Lesson Loading
// ===================================

async function loadLessons() {
    try {
        // Load both Farsi lessons
        const lesson1 = await fetch('../curriculum/farsi/lessons/01-essential-greetings.json').then(r => r.json());
        const lesson2 = await fetch('../curriculum/farsi/lessons/02-poetic-heart.json').then(r => r.json());

        state.lessons = [lesson1, lesson2];
        updateStats();
    } catch (error) {
        console.error('Error loading lessons:', error);
    }
}

function updateStats() {
    const totalPhrases = state.lessons.reduce((sum, lesson) => sum + lesson.phrases.length, 0);
    const totalLessons = state.lessons.length;
    const masteredCount = state.progress.masteredPhrases.length;

    document.getElementById('total-phrases').textContent = totalPhrases;
    document.getElementById('total-lessons').textContent = totalLessons;
    document.getElementById('mastered-count').textContent = masteredCount;
}

// ===================================
// Lessons View
// ===================================

function loadLessonsView() {
    const container = document.getElementById('lessons-list');
    container.innerHTML = '';

    state.lessons.forEach(lesson => {
        const card = createLessonCard(lesson);
        container.appendChild(card);
    });
}

function createLessonCard(lesson) {
    const card = document.createElement('div');
    card.className = 'lesson-card';
    card.onclick = () => navigateToLessonDetail(lesson.lesson_id);

    const masteredCount = lesson.phrases.filter(p =>
        state.progress.masteredPhrases.includes(p.id)
    ).length;

    card.innerHTML = `
        <div class="lesson-header">
            <div class="lesson-level">${lesson.level}</div>
            <h3 class="lesson-title">${lesson.title}</h3>
            <p class="lesson-description">${lesson.description}</p>
        </div>
        <div class="lesson-meta">
            <div class="lesson-stat">
                <strong>${lesson.phrases.length}</strong> phrases
            </div>
            <div class="lesson-stat">
                <strong>${lesson.parrot_exercises.length}</strong> exercises
            </div>
            <div class="lesson-stat">
                <strong>${masteredCount}/${lesson.phrases.length}</strong> mastered
            </div>
        </div>
    `;

    return card;
}

// ===================================
// Lesson Detail View
// ===================================

function renderLessonDetail(lesson) {
    const container = document.getElementById('lesson-content');

    container.innerHTML = `
        <div class="lesson-detail-header">
            <div class="lesson-level">${lesson.level}</div>
            <h2 class="lesson-title">${lesson.title}</h2>
            <p class="lesson-description">${lesson.description}</p>

            ${lesson.learning_objectives ? `
                <div class="lesson-objectives">
                    <h3>Learning Objectives</h3>
                    <ul>
                        ${lesson.learning_objectives.map(obj => `<li>${obj}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        </div>

        <div class="phrases-section">
            <h3 class="section-title">Phrases</h3>
            ${lesson.phrases.map(phrase => createPhraseCard(phrase)).join('')}
        </div>

        ${lesson.parrot_exercises && lesson.parrot_exercises.length > 0 ? `
            <div class="exercises-section mt-lg">
                <h3 class="section-title">Parroting Exercises</h3>
                <p class="text-center mb-md">Practice these exercises to internalize the phrases</p>
                ${lesson.parrot_exercises.map(ex => createExerciseCard(ex, lesson)).join('')}
            </div>
        ` : ''}
    `;
}

function createPhraseCard(phrase) {
    const isMastered = state.progress.masteredPhrases.includes(phrase.id);

    return `
        <div class="phrase-card" data-phrase-id="${phrase.id}">
            <div class="phrase-main">
                <div class="phrase-native">${phrase.native}</div>
                <div class="phrase-transliteration">${phrase.transliteration}</div>
                <div class="phrase-translation">→ ${phrase.translation}</div>
                ${phrase.literal_translation ? `
                    <div class="phrase-literal">Literally: "${phrase.literal_translation}"</div>
                ` : ''}
            </div>

            ${phrase.cultural_notes ? `
                <div class="phrase-cultural">
                    <strong>🎭 Cultural Context:</strong> ${phrase.cultural_notes}
                </div>
            ` : ''}

            <div class="phrase-details">
                ${phrase.usage_context && phrase.usage_context.length > 0 ? `
                    <div class="phrase-section">
                        <h4>💬 When to use</h4>
                        <ul>
                            ${phrase.usage_context.map(ctx => `<li>${ctx}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}

                ${phrase.patterns && phrase.patterns.length > 0 ? `
                    <div class="phrase-section">
                        <h4>🔍 Language Patterns</h4>
                        <ul>
                            ${phrase.patterns.map(pattern => `<li>${pattern}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}

                ${phrase.breakdown && phrase.breakdown.length > 0 ? `
                    <div class="phrase-section">
                        <h4>📝 Word Breakdown</h4>
                        <div class="phrase-breakdown">
                            ${phrase.breakdown.map(item => `
                                <div class="breakdown-item">
                                    <span class="breakdown-word">${item.word}</span>
                                    (${item.transliteration}) = ${item.meaning}
                                    <em>[${item.part_of_speech}]</em>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                ${phrase.parrot_method ? `
                    <div class="phrase-section">
                        <h4>🦜 Parroting Instructions</h4>
                        <ul>
                            <li>Repetitions: ${phrase.parrot_method.repetitions_suggested}</li>
                            <li>Speed: ${phrase.parrot_method.speed}</li>
                            ${phrase.parrot_method.focus_sounds && phrase.parrot_method.focus_sounds.length > 0 ? `
                                <li>Focus sounds:
                                    <ul>
                                        ${phrase.parrot_method.focus_sounds.map(sound => `<li>${sound}</li>`).join('')}
                                    </ul>
                                </li>
                            ` : ''}
                        </ul>
                    </div>
                ` : ''}
            </div>

            <div style="text-align: center; margin-top: 1.5rem;">
                <button class="btn-primary" onclick="togglePhraseMastery('${phrase.id}')" id="master-btn-${phrase.id}">
                    ${isMastered ? '✓ Mastered' : 'Mark as Mastered'}
                </button>
            </div>
        </div>
    `;
}

function createExerciseCard(exercise, lesson) {
    const exerciseIcons = {
        'listen-repeat': '👂',
        'shadowing': '🗣️',
        'pattern-drill': '🔍',
        'progressive-speed': '⚡',
        'memory-recall': '🧠'
    };

    const icon = exerciseIcons[exercise.exercise_type] || '🦜';

    return `
        <div class="method-card" style="text-align: left;">
            <div class="method-icon">${icon}</div>
            <h4>${exercise.exercise_type.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</h4>
            <p style="margin: 1rem 0;">${exercise.instructions}</p>
            <p style="font-size: 0.9rem; color: var(--persian-gray);">
                <strong>Phrases:</strong> ${exercise.phrase_ids.length} included
            </p>
        </div>
    `;
}

function togglePhraseMastery(phraseId) {
    if (state.progress.masteredPhrases.includes(phraseId)) {
        // Remove from mastered
        state.progress.masteredPhrases = state.progress.masteredPhrases.filter(id => id !== phraseId);
    } else {
        // Add to mastered
        state.progress.masteredPhrases.push(phraseId);
    }

    saveProgress();
    updateStats();

    // Update button
    const btn = document.getElementById(`master-btn-${phraseId}`);
    if (btn) {
        const isMastered = state.progress.masteredPhrases.includes(phraseId);
        btn.textContent = isMastered ? '✓ Mastered' : 'Mark as Mastered';
    }

    // Reload lessons view if visible
    if (document.getElementById('lessons-view').classList.contains('active')) {
        loadLessonsView();
    }
}

// ===================================
// Patterns View (Attention Visualization)
// ===================================

function loadPatternsView() {
    // Initialize the attention visualization
    if (typeof initAttentionView === 'function') {
        initAttentionView();
    }
}

// ===================================
// Progress View
// ===================================

function loadProgressView() {
    const container = document.getElementById('progress-content');

    if (state.progress.masteredPhrases.length === 0) {
        container.innerHTML = `
            <div class="progress-empty">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🦜</div>
                <h3>No progress yet!</h3>
                <p>Start learning and mark phrases as mastered to track your progress.</p>
                <button class="btn-primary" onclick="navigateToLessons()" style="margin-top: 2rem;">
                    Start Learning
                </button>
            </div>
        `;
        return;
    }

    const totalPhrases = state.lessons.reduce((sum, lesson) => sum + lesson.phrases.length, 0);
    const percentage = Math.round((state.progress.masteredPhrases.length / totalPhrases) * 100);

    let progressHTML = `
        <div style="text-align: center; margin-bottom: 3rem;">
            <h3 style="font-size: 4rem; color: var(--persian-teal); margin: 0;">
                ${percentage}%
            </h3>
            <p style="font-size: 1.2rem; color: var(--persian-gray);">
                ${state.progress.masteredPhrases.length} of ${totalPhrases} phrases mastered
            </p>
        </div>

        <h3 class="section-title">Progress by Lesson</h3>
    `;

    state.lessons.forEach(lesson => {
        const masteredInLesson = lesson.phrases.filter(p =>
            state.progress.masteredPhrases.includes(p.id)
        ).length;
        const lessonPercentage = Math.round((masteredInLesson / lesson.phrases.length) * 100);

        progressHTML += `
            <div class="lesson-card" onclick="navigateToLessonDetail('${lesson.lesson_id}')" style="margin-bottom: 1rem;">
                <h4 style="margin-bottom: 1rem;">${lesson.title}</h4>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.5rem; color: var(--persian-teal);">
                            ${masteredInLesson}/${lesson.phrases.length}
                        </strong>
                        <span style="color: var(--persian-gray);"> phrases mastered</span>
                    </div>
                    <div style="font-size: 2rem; font-weight: bold; color: var(--persian-blue);">
                        ${lessonPercentage}%
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = progressHTML;
}

// ===================================
// Initialization
// ===================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadLessons();
    updateStats();
});
