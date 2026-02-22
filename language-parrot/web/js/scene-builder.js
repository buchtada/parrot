/**
 * Scene-Based Lesson Builder
 * Upload image → Detect objects → Overlay Farsi → Generate lesson
 */

let currentImage = null;
let detectedObjects = [];
let sceneCanvas = null;
let sceneCtx = null;

// Initialize Scene Builder
function initSceneBuilder() {
    const uploadZone = document.getElementById('upload-zone');
    const imageInput = document.getElementById('image-upload');
    sceneCanvas = document.getElementById('scene-canvas');
    sceneCtx = sceneCanvas.getContext('2d');

    // Click to upload
    uploadZone.addEventListener('click', () => imageInput.click());

    // File selection
    imageInput.addEventListener('change', handleImageUpload);

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragging');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragging');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragging');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            loadImage(file);
        }
    });
}

// Handle image upload
function handleImageUpload(e) {
    const file = e.target.files[0];
    if (file) {
        loadImage(file);
    }
}

// Load and display image
function loadImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            displayImage(img);
            document.getElementById('canvas-container').classList.add('active');
            document.getElementById('detect-btn').disabled = false;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Display image on canvas
function displayImage(img) {
    // Calculate canvas size (max 800px width)
    const maxWidth = 800;
    const scale = Math.min(1, maxWidth / img.width);
    sceneCanvas.width = img.width * scale;
    sceneCanvas.height = img.height * scale;

    // Draw image
    sceneCtx.clearRect(0, 0, sceneCanvas.width, sceneCanvas.height);
    sceneCtx.drawImage(img, 0, 0, sceneCanvas.width, sceneCanvas.height);
}

// Detect objects in image
async function detectObjects() {
    if (!currentImage) return;

    const progressEl = document.getElementById('detection-progress');
    progressEl.classList.add('active');

    try {
        // Load TensorFlow.js and COCO-SSD model
        await loadObjectDetectionModel();

        // Perform detection
        const predictions = await model.detect(sceneCanvas);

        // Process detections
        detectedObjects = predictions.map(pred => ({
            label: pred.class,
            confidence: pred.score,
            box: pred.bbox,
            farsi: getFarsiVocab(pred.class)
        }));

        // Display results
        displayDetectedObjects();
        overlayFarsiLabels();

        progressEl.classList.remove('active');
        document.getElementById('detected-objects').classList.add('active');
        document.getElementById('overlay-btn').disabled = false;
        document.getElementById('generate-btn').disabled = false;

    } catch (error) {
        console.error('Detection error:', error);
        alert('Object detection failed. Using demo mode with common objects.');
        useDemoDetection();
        progressEl.classList.remove('active');
    }
}

// Load object detection model (TensorFlow.js + COCO-SSD)
let model = null;
async function loadObjectDetectionModel() {
    if (model) return;

    // Check if TensorFlow.js and COCO-SSD are loaded
    if (typeof cocoSsd === 'undefined') {
        throw new Error('COCO-SSD not loaded. Using fallback detection.');
    }

    model = await cocoSsd.load();
}

// Fallback: Demo detection for testing without TensorFlow
function useDemoDetection() {
    // Simulate detection with common objects
    detectedObjects = [
        {
            label: 'person',
            confidence: 0.95,
            box: [100, 50, 200, 400],
            farsi: getFarsiVocab('person')
        },
        {
            label: 'chair',
            confidence: 0.88,
            box: [400, 200, 150, 200],
            farsi: getFarsiVocab('chair')
        },
        {
            label: 'book',
            confidence: 0.82,
            box: [250, 350, 80, 100],
            farsi: getFarsiVocab('book')
        }
    ];

    displayDetectedObjects();
    document.getElementById('detected-objects').classList.add('active');
    document.getElementById('overlay-btn').disabled = false;
    document.getElementById('generate-btn').disabled = false;
}

// Display detected objects list
function displayDetectedObjects() {
    const container = document.getElementById('objects-list');
    container.innerHTML = '';

    detectedObjects.forEach((obj, idx) => {
        const item = document.createElement('div');
        item.className = 'object-item';
        item.innerHTML = `
            <div class="object-english">${obj.label}</div>
            <div class="object-farsi">${obj.farsi.farsi}</div>
            <div class="object-transliteration">${obj.farsi.transliteration}</div>
        `;
        item.addEventListener('click', () => {
            item.classList.toggle('selected');
        });
        container.appendChild(item);
    });
}

// Overlay Farsi labels on image
function overlayFarsiLabels() {
    // Redraw original image
    displayImage(currentImage);

    // Draw labels with boxes
    sceneCtx.font = 'bold 24px Scheherazade New, serif';
    sceneCtx.textAlign = 'right';
    sceneCtx.textBaseline = 'top';

    detectedObjects.forEach(obj => {
        const [x, y, width, height] = obj.box;

        // Draw bounding box
        sceneCtx.strokeStyle = '#d4a574';
        sceneCtx.lineWidth = 3;
        sceneCtx.strokeRect(x, y, width, height);

        // Draw label background
        const padding = 10;
        const labelWidth = sceneCtx.measureText(obj.farsi.farsi).width + padding * 2;
        const labelHeight = 40;

        sceneCtx.fillStyle = 'rgba(26, 77, 122, 0.9)';
        sceneCtx.fillRect(x + width - labelWidth, y - labelHeight, labelWidth, labelHeight);

        // Draw Farsi text
        sceneCtx.fillStyle = '#ffffff';
        sceneCtx.fillText(obj.farsi.farsi, x + width - padding, y - labelHeight + 8);

        // Draw transliteration
        sceneCtx.font = '14px Inter, sans-serif';
        sceneCtx.fillStyle = '#d4a574';
        sceneCtx.textAlign = 'left';
        sceneCtx.fillText(obj.farsi.transliteration, x + padding, y + height + 5);
        sceneCtx.font = 'bold 24px Scheherazade New, serif';
        sceneCtx.textAlign = 'right';
    });
}

// Generate lesson from detected objects
function generateLesson() {
    if (detectedObjects.length === 0) {
        alert('No objects detected. Please detect objects first.');
        return;
    }

    // Create vocabulary list
    const vocabulary = detectedObjects.map(obj => ({
        english: obj.label,
        farsi: obj.farsi.farsi,
        transliteration: obj.farsi.transliteration,
        category: obj.farsi.category
    }));

    // Analyze patterns
    const patterns = analyzeVocabularyPatterns(detectedObjects.map(o => o.label));

    // Display lesson preview
    displayLessonPreview(vocabulary, patterns);

    document.getElementById('lesson-preview').classList.add('active');
}

// Display generated lesson
function displayLessonPreview(vocabulary, patterns) {
    const container = document.getElementById('vocabulary-grid');
    container.innerHTML = '';

    vocabulary.forEach(word => {
        const card = document.createElement('div');
        card.className = 'vocab-card';
        card.innerHTML = `
            <div class="vocab-farsi">${word.farsi}</div>
            <div class="vocab-translit">${word.transliteration}</div>
            <div class="vocab-english">${word.english}</div>
        `;
        container.appendChild(card);
    });

    // Display pattern analysis
    const patternsContainer = document.getElementById('pattern-list');
    patternsContainer.innerHTML = '';

    if (patterns.loanwords.length > 0) {
        const li = document.createElement('li');
        li.textContent = `${patterns.loanwords.length} loanwords (sounds like English): ${patterns.loanwords.map(w => w.farsi).join('، ')}`;
        patternsContainer.appendChild(li);
    }

    if (patterns.compounds.length > 0) {
        const li = document.createElement('li');
        li.textContent = `${patterns.compounds.length} compound words: ${patterns.compounds.map(w => w.farsi).join('، ')}`;
        patternsContainer.appendChild(li);
    }

    // Show similar sounds
    Object.entries(patterns.similar_sounds).forEach(([sound, words]) => {
        if (words.length > 1) {
            const li = document.createElement('li');
            li.textContent = `Words starting with '${sound}': ${words.map(w => w.farsi).join('، ')}`;
            patternsContainer.appendChild(li);
        }
    });
}

// Save lesson as JSON
function saveLesson() {
    const vocabulary = detectedObjects.map(obj => ({
        english: obj.label,
        farsi: obj.farsi.farsi,
        transliteration: obj.farsi.transliteration,
        category: obj.farsi.category
    }));

    const lesson = {
        lesson_id: `scene-${Date.now()}`,
        language: "farsi",
        level: "beginner",
        title: "Scene Vocabulary",
        description: "Vocabulary from uploaded image",
        learning_objectives: [
            "Learn vocabulary from real-world scenes",
            "Practice object recognition in context",
            "Build visual memory associations"
        ],
        phrases: vocabulary.map((word, idx) => ({
            id: `scene-${Date.now()}-${idx}`,
            native: word.farsi,
            transliteration: word.transliteration,
            translation: word.english,
            frequency_rank: 100,
            formality: "neutral",
            usage_context: [`Referring to ${word.english} in everyday situations`],
            patterns: [],
            cultural_notes: `Common vocabulary: ${word.category}`,
            parrot_method: {
                repetitions_suggested: 10,
                speed: "normal",
                focus_sounds: []
            }
        }))
    };

    // Download as JSON
    const dataStr = JSON.stringify(lesson, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `scene-lesson-${Date.now()}.json`;
    link.click();
}

// Initialize when view loads
function loadSceneBuilderView() {
    initSceneBuilder();
}

// Export functions
window.loadSceneBuilderView = loadSceneBuilderView;
window.detectObjects = detectObjects;
window.overlayFarsiLabels = overlayFarsiLabels;
window.generateLesson = generateLesson;
window.saveLesson = saveLesson;
