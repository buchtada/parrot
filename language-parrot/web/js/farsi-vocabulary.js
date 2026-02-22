/**
 * Farsi Vocabulary Database
 * Maps English objects to Farsi translations
 */

const FARSI_VOCABULARY = {
    // People
    "person": { farsi: "آدم", transliteration: "adam", category: "people" },
    "man": { farsi: "مرد", transliteration: "mard", category: "people" },
    "woman": { farsi: "زن", transliteration: "zan", category: "people" },
    "child": { farsi: "بچه", transliteration: "bacheh", category: "people" },
    "baby": { farsi: "نوزاد", transliteration: "nowzad", category: "people" },

    // Animals
    "cat": { farsi: "گربه", transliteration: "gorbeh", category: "animals" },
    "dog": { farsi: "سگ", transliteration: "sag", category: "animals" },
    "bird": { farsi: "پرنده", transliteration: "parandeh", category: "animals" },
    "horse": { farsi: "اسب", transliteration: "asb", category: "animals" },
    "cow": { farsi: "گاو", transliteration: "gav", category: "animals" },

    // Food & Kitchen
    "apple": { farsi: "سیب", transliteration: "sib", category: "food" },
    "banana": { farsi: "موز", transliteration: "mowz", category: "food" },
    "orange": { farsi: "پرتقال", transliteration: "porteghaal", category: "food" },
    "cake": { farsi: "کیک", transliteration: "keyk", category: "food" },
    "pizza": { farsi: "پیتزا", transliteration: "pizza", category: "food" },
    "sandwich": { farsi: "ساندویچ", transliteration: "sandvich", category: "food" },
    "bottle": { farsi: "بطری", transliteration: "botri", category: "kitchen" },
    "cup": { farsi: "فنجان", transliteration: "fenjaan", category: "kitchen" },
    "bowl": { farsi: "کاسه", transliteration: "kaseh", category: "kitchen" },
    "fork": { farsi: "چنگال", transliteration: "changaal", category: "kitchen" },
    "knife": { farsi: "چاقو", transliteration: "chaghu", category: "kitchen" },
    "spoon": { farsi: "قاشق", transliteration: "ghashog", category: "kitchen" },

    // Furniture & Home
    "chair": { farsi: "صندلی", transliteration: "sandali", category: "furniture" },
    "table": { farsi: "میز", transliteration: "miz", category: "furniture" },
    "bed": { farsi: "تخت", transliteration: "takht", category: "furniture" },
    "couch": { farsi: "مبل", transliteration: "mobl", category: "furniture" },
    "sofa": { farsi: "کاناپه", transliteration: "kanapeh", category: "furniture" },
    "door": { farsi: "در", transliteration: "dar", category: "home" },
    "window": { farsi: "پنجره", transliteration: "panjereh", category: "home" },
    "clock": { farsi: "ساعت", transliteration: "sa'at", category: "home" },
    "lamp": { farsi: "چراغ", transliteration: "cheragh", category: "home" },
    "book": { farsi: "کتاب", transliteration: "ketaab", category: "home" },
    "vase": { farsi: "گلدان", transliteration: "goldaan", category: "home" },

    // Electronics
    "phone": { farsi: "تلفن", transliteration: "telefon", category: "electronics" },
    "cell phone": { farsi: "موبایل", transliteration: "mobayl", category: "electronics" },
    "laptop": { farsi: "لپ‌تاپ", transliteration: "laptop", category: "electronics" },
    "computer": { farsi: "کامپیوتر", transliteration: "kompyuter", category: "electronics" },
    "tv": { farsi: "تلویزیون", transliteration: "televizyon", category: "electronics" },
    "keyboard": { farsi: "صفحه‌کلید", transliteration: "safhe-kelid", category: "electronics" },
    "mouse": { farsi: "موس", transliteration: "moos", category: "electronics" },

    // Transportation
    "car": { farsi: "ماشین", transliteration: "mashin", category: "transportation" },
    "bicycle": { farsi: "دوچرخه", transliteration: "docharkheh", category: "transportation" },
    "motorcycle": { farsi: "موتورسیکلت", transliteration: "motorsiklet", category: "transportation" },
    "bus": { farsi: "اتوبوس", transliteration: "otobus", category: "transportation" },
    "truck": { farsi: "کامیون", transliteration: "kamiyon", category: "transportation" },
    "airplane": { farsi: "هواپیما", transliteration: "havaapeymaa", category: "transportation" },
    "train": { farsi: "قطار", transliteration: "ghataar", category: "transportation" },
    "boat": { farsi: "قایق", transliteration: "ghayegh", category: "transportation" },

    // Nature
    "tree": { farsi: "درخت", transliteration: "derakht", category: "nature" },
    "flower": { farsi: "گل", transliteration: "gol", category: "nature" },
    "grass": { farsi: "چمن", transliteration: "chaman", category: "nature" },
    "mountain": { farsi: "کوه", transliteration: "kuh", category: "nature" },
    "river": { farsi: "رودخانه", transliteration: "rudkhaneh", category: "nature" },
    "sun": { farsi: "خورشید", transliteration: "khorshid", category: "nature" },
    "moon": { farsi: "ماه", transliteration: "mah", category: "nature" },
    "star": { farsi: "ستاره", transliteration: "setareh", category: "nature" },
    "cloud": { farsi: "ابر", transliteration: "abr", category: "nature" },

    // Clothing
    "shirt": { farsi: "پیراهن", transliteration: "pirahan", category: "clothing" },
    "pants": { farsi: "شلوار", transliteration: "shalvaar", category: "clothing" },
    "shoes": { farsi: "کفش", transliteration: "kafsh", category: "clothing" },
    "hat": { farsi: "کلاه", transliteration: "kolah", category: "clothing" },
    "tie": { farsi: "کراوات", transliteration: "keravat", category: "clothing" },
    "dress": { farsi: "لباس", transliteration: "lebaas", category: "clothing" },

    // Sports & Activities
    "ball": { farsi: "توپ", transliteration: "tup", category: "sports" },
    "tennis racket": { farsi: "راکت تنیس", transliteration: "raket tenis", category: "sports" },
    "skateboard": { farsi: "اسکیت‌برد", transliteration: "eskeyt-bord", category: "sports" },
    "surfboard": { farsi: "تخته‌موج‌سواری", transliteration: "takhte-mowj-savari", category: "sports" },
    "kite": { farsi: "بادبادک", transliteration: "badbadak", category: "sports" },

    // Common Objects
    "umbrella": { farsi: "چتر", transliteration: "chatr", category: "objects" },
    "bag": { farsi: "کیف", transliteration: "kif", category: "objects" },
    "backpack": { farsi: "کوله‌پشتی", transliteration: "kuleh-poshti", category: "objects" },
    "suitcase": { farsi: "چمدان", transliteration: "chamedaan", category: "objects" },
    "pen": { farsi: "خودکار", transliteration: "khodkaar", category: "objects" },
    "pencil": { farsi: "مداد", transliteration: "medaad", category: "objects" },
    "scissors": { farsi: "قیچی", transliteration: "gheichi", category: "objects" },

    // Fallback for unknown
    "object": { farsi: "شیء", transliteration: "shey", category: "general" },
    "thing": { farsi: "چیز", transliteration: "chiz", category: "general" }
};

// Analyze patterns in vocabulary
function analyzeVocabularyPatterns(words) {
    const patterns = {
        plurals: [],
        compounds: [],
        loanwords: [],
        similar_sounds: {}
    };

    words.forEach(word => {
        const vocab = FARSI_VOCABULARY[word.toLowerCase()];
        if (!vocab) return;

        // Check for loanwords (words that sound like English)
        if (vocab.transliteration.toLowerCase().includes(word.toLowerCase().substring(0, 3))) {
            patterns.loanwords.push({ english: word, ...vocab });
        }

        // Check for compound words (with -)
        if (vocab.farsi.includes('‌')) {
            patterns.compounds.push({ english: word, ...vocab });
        }

        // Group by similar sounds
        const firstSound = vocab.transliteration[0];
        if (!patterns.similar_sounds[firstSound]) {
            patterns.similar_sounds[firstSound] = [];
        }
        patterns.similar_sounds[firstSound].push({ english: word, ...vocab });
    });

    return patterns;
}

// Get vocabulary for a word
function getFarsiVocab(englishWord) {
    const word = englishWord.toLowerCase();
    return FARSI_VOCABULARY[word] || FARSI_VOCABULARY["thing"];
}

// Get all words in a category
function getVocabByCategory(category) {
    return Object.entries(FARSI_VOCABULARY)
        .filter(([_, vocab]) => vocab.category === category)
        .map(([english, vocab]) => ({ english, ...vocab }));
}

window.FARSI_VOCABULARY = FARSI_VOCABULARY;
window.getFarsiVocab = getFarsiVocab;
window.analyzeVocabularyPatterns = analyzeVocabularyPatterns;
window.getVocabByCategory = getVocabByCategory;
