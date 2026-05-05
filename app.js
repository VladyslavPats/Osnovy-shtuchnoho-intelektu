// Глобальні змінні стану
let globalTasks = [];
let favoriteIds = JSON.parse(localStorage.getItem('userFavorites')) || [];
let totalAnswers = parseInt(localStorage.getItem('totalAnswers')) || 0;
let correctAnswers = parseInt(localStorage.getItem('correctAnswers')) || 0;
let examTimer;

// Тексти інтерфейсу для локалізації
const interfaceTexts = {
    uk: {
        appTitle: "Вивчення англійської", registerButton: "Реєстрація", loginButton: "Вхід", guestButton: "Гість",
        logoutButton: "Вийти", progressTitle: "Прогрес", topicLabel: "Тема:", modeLabel: "Режим:", generateButton: "Старт",
        checkButton: "Перевірити", topicFood: "Їжа", topicTravel: "Подорожі", topicTechnology: "IT", topicSport: "Спорт", topicNature: "Природа",
        processingText: "Обробка...", dictionaryTitle: "Словник", filterAll: "Всі", filterFavorites: "Улюблені",
        modePractice: "Практика", modeExam: "Іспит", modeAI: "AI Генерація", resultTitle: "Результат: [SCORE]/[TOTAL]",
        regUsernamePlaceholder: "Ім'я", regEmailPlaceholder: "Email", regPasswordPlaceholder: "Пароль",
        loginEmailPlaceholder: "Email", loginPasswordPlaceholder: "Пароль", settingsTitle: "Налаштування"
    },
    en: {
        appTitle: "Learn English", registerButton: "Sign Up", loginButton: "Login", guestButton: "Guest",
        logoutButton: "Logout", progressTitle: "Progress", topicLabel: "Topic:", modeLabel: "Mode:", generateButton: "Start",
        checkButton: "Check", topicFood: "Food", topicTravel: "Travel", topicTechnology: "IT", topicSport: "Sport", topicNature: "Nature",
        processingText: "Processing...", dictionaryTitle: "Vocabulary", filterAll: "All", filterFavorites: "Favorites",
        modePractice: "Practice", modeExam: "Exam", modeAI: "AI Generation", resultTitle: "Result: [SCORE]/[TOTAL]",
        regUsernamePlaceholder: "Name", regEmailPlaceholder: "Email", regPasswordPlaceholder: "Password",
        loginEmailPlaceholder: "Email", loginPasswordPlaceholder: "Password", settingsTitle: "Settings"
    }
};

let currentLang = localStorage.getItem('userLang') || 'uk';

// Функція оновлення прогрес-бару
function updateProgressBar() {
    const fill = document.getElementById('progress-bar-fill');
    const text = document.getElementById('progress-text');
    let percent = totalAnswers === 0 ? 0 : Math.round((correctAnswers / totalAnswers) * 100);
    fill.style.width = `${percent}%`;
    text.textContent = `${percent}%`;
}

// Функція навігації між екранами
function navigateTo(id) {
    document.querySelectorAll('.screen').forEach(s => s.style.display = 'none');
    const target = document.getElementById(id);
    if(target) target.style.display = 'block';
    if (id === 'app-screen') {
        loadData();
        updateProgressBar();
    }
}

// Функція перекладу інтерфейсу
function updateUI() {
    const texts = interfaceTexts[currentLang];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if(texts[key]) el.textContent = texts[key];
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if(texts[key]) el.placeholder = texts[key];
    });
}

// Завантаження даних словника
async function loadData() {
    try {
        const res = await fetch('data.json');
        globalTasks = await res.json();
        renderCards(globalTasks);
    } catch (e) {
        console.error("Помилка завантаження JSON:", e);
    }
}

// Відтворення карток словника
function renderCards(data) {
    const container = document.getElementById('dictionary-container');
    container.innerHTML = '';
    data.forEach(item => {
        const isFav = favoriteIds.includes(item.id);
        container.insertAdjacentHTML('beforeend', `
            <div class="word-card" data-category="${item.topic}">
                <div class="word-info">
                    <strong>${item.q} <i class="fas fa-volume-up speak-btn" data-word="${item.q}" title="Listen"></i></strong><br>
                    <small>${item.a}</small>
                </div>
                <button class="favorite-btn ${isFav ? 'is-active' : ''}" onclick="toggleFav(${item.id}, this)">
                    <i class="${isFav ? 'fa-solid' : 'fa-regular'} fa-heart"></i>
                </button>
            </div>
        `);
    });
    
    // Додаємо подію озвучки для іконок динаміка
    document.querySelectorAll('.speak-btn').forEach(btn => {
        btn.onclick = (e) => {
            e.stopPropagation();
            const ut = new SpeechSynthesisUtterance(btn.dataset.word);
            ut.lang = 'en-US';
            window.speechSynthesis.speak(ut);
        };
    });
}

// Додавання/видалення з улюблених
function toggleFav(id, btn) {
    if (favoriteIds.includes(id)) {
        favoriteIds = favoriteIds.filter(f => f !== id);
        btn.querySelector('i').className = 'fa-regular fa-heart';
    } else {
        favoriteIds.push(id);
        btn.querySelector('i').className = 'fa-solid fa-heart';
    }
    btn.classList.toggle('is-active');
    localStorage.setItem('userFavorites', JSON.stringify(favoriteIds));
}

// ОСНОВНА ЛОГІКА ГЕНЕРАЦІЇ ЗАВДАНЬ
async function generateTasks() {
    const topic = document.getElementById('topic').value;
    const mode = document.getElementById('mode').value;
    const container = document.getElementById('task-container');
    const timerDisplay = document.getElementById('exam-timer');
    
    clearInterval(examTimer);
    container.innerHTML = '';
    timerDisplay.style.display = 'none';

    if (mode === 'ai') {
        const availableWords = globalTasks.filter(t => t.topic === topic);
        const selectedForAI = availableWords.sort(() => 0.5 - Math.random()).slice(0, 3).map(t => t.q);

        container.innerHTML = `<div class="loader"><i class="fas fa-spinner fa-spin"></i> ${interfaceTexts[currentLang].processingText}</div>`;
        try {
            const res = await fetch('http://127.0.0.1:8000/api/generate-task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic, words: selectedForAI })
            });
            const data = await res.json();
            container.innerHTML = ''; // Очищаємо лоадер

            const div = document.createElement('div');
            div.className = 'task screen';
            div.innerHTML = `<p style="line-height: 1.8;">${data.text.replace(/___/g, '<span style="display:inline-block; width:60px; border-bottom:2px solid var(--primary-color);"></span>')}</p>`;

            const btn = document.createElement('button');
            btn.className = 'auth-btn primary-btn';
            btn.textContent = interfaceTexts[currentLang].checkButton;

            data.answers.forEach((ans, i) => {
                const input = document.createElement('input');
                input.type = 'text';
                input.id = `ai-ans-${i}`;
                input.placeholder = `...`;
                input.style.marginTop = '10px';
                
                input.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        btn.click();
                    }
                });

                div.appendChild(input); 
            });

            container.appendChild(div);

            btn.onclick = async () => {
                btn.disabled = true;
                await checkAI(data.answers);
            };
            container.appendChild(btn);

        } catch (e) {
            container.innerHTML = `<div class="error-text">API Error. Перевірте, чи запущений сервер (uvicorn).</div>`;
        }
    } else {
        const tasks = globalTasks.filter(t => t.topic === topic).sort(() => 0.5 - Math.random()).slice(0, 3);
        
        if (mode === 'exam') {
            timerDisplay.style.display = 'block';
            let time = 60;
            examTimer = setInterval(() => {
                time--; timerDisplay.textContent = time;
                if(time <= 0) { clearInterval(examTimer); checkSimple(tasks); }
            }, 1000);
        }

        tasks.forEach((t, i) => {
            container.insertAdjacentHTML('beforeend', `<div class="task screen"><p>${t.q}</p><input type="text" id="ans-${i}" placeholder="..."></div>`);
        });

        const btn = document.createElement('button');
        btn.className = 'auth-btn primary-btn';
        btn.textContent = interfaceTexts[currentLang].checkButton;
        
        container.querySelectorAll('input').forEach(inp => {
            inp.onkeypress = (e) => { if(e.key === 'Enter') btn.click(); };
        });

        btn.onclick = () => checkSimple(tasks);
        container.appendChild(btn);
    }
}

// Розумна AI перевірка
async function checkAI(correctList) {
    let score = 0;
    for (let i = 0; i < correctList.length; i++) {
        const inp = document.getElementById(`ai-ans-${i}`);
        const userWord = inp.value.trim();
        if(!userWord) continue;

        const res = await fetch('http://127.0.0.1:8000/api/check-answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_word: userWord, correct_word: correctList[i] })
        });
        const data = await res.json();
        
        if (data.is_correct) {
            inp.style.borderColor = 'var(--success-color)';
            score++; correctAnswers++;
        } else {
            inp.style.borderColor = 'var(--fail-color)';
        }
        totalAnswers++;
    }
    saveAndAlert(score, correctList.length, correctList);
}

// Звичайна перевірка (Практика/Іспит)
function checkSimple(tasks) {
    clearInterval(examTimer);
    let score = 0;
    tasks.forEach((t, i) => {
        const inp = document.getElementById(`ans-${i}`);
        if(inp.value.toLowerCase().trim() === t.a.toLowerCase()) {
            inp.style.borderColor = 'var(--success-color)';
            score++; correctAnswers++;
        } else {
            inp.style.borderColor = 'var(--fail-color)';
        }
        totalAnswers++;
    });
    saveAndAlert(score, tasks.length);
}

function saveAndAlert(score, total, answers = null) {
    localStorage.setItem('correctAnswers', correctAnswers);
    localStorage.setItem('totalAnswers', totalAnswers);
    updateProgressBar();
    const msg = `${interfaceTexts[currentLang].resultTitle.replace('[SCORE]', score).replace('[TOTAL]', total)}${answers ? '\nCorrect: ' + answers.join(', ') : ''}`;
    alert(msg);
}

// ОБРОБНИКИ ПОДІЙ АВТОРИЗАЦІЇ
document.getElementById('show-registration-btn').onclick = () => navigateTo('registration-screen');
document.getElementById('show-login-btn').onclick = () => navigateTo('login-screen');
document.getElementById('back-to-auth-reg-btn').onclick = () => navigateTo('auth-screen');
document.getElementById('back-to-auth-login-btn').onclick = () => navigateTo('auth-screen');

document.getElementById('register-btn').onclick = () => {
    const name = document.getElementById('reg-username-input').value.trim();
    if(name.length < 2) return;
    localStorage.setItem('mockUser', JSON.stringify({name, email: 'user@example.com', pass: '123456'}));
    localStorage.setItem('currentUser', name);
    document.getElementById('user-display-name').textContent = name;
    navigateTo('app-screen');
};

document.getElementById('login-btn').onclick = () => {
    const user = JSON.parse(localStorage.getItem('mockUser')) || {name: 'Admin'};
    localStorage.setItem('currentUser', user.name);
    document.getElementById('user-display-name').textContent = user.name;
    navigateTo('app-screen');
};

document.getElementById('guest-btn').onclick = () => {
    localStorage.setItem('currentUser', 'Guest');
    document.getElementById('user-display-name').textContent = 'Guest';
    navigateTo('app-screen');
};

document.getElementById('logout-btn').onclick = () => {
    localStorage.removeItem('currentUser');
    location.reload();
};

// --- ЛОГІКА НАЛАШТУВАНЬ ТА БУРГЕР-МЕНЮ ---
const mobileMenu = document.getElementById('mobile-menu');
const burgerBtn = document.getElementById('burger-btn');
const closeMenuBtn = document.getElementById('close-menu-btn');

if (burgerBtn) burgerBtn.onclick = () => mobileMenu.classList.add('is-open');
if (closeMenuBtn) closeMenuBtn.onclick = () => mobileMenu.classList.remove('is-open');

document.getElementById('settings-btn').onclick = () => {
    document.getElementById('settings-modal').style.display = 'flex';
    if (mobileMenu) mobileMenu.classList.remove('is-open'); // Закриваємо бургер при відкритті налаштувань
};
document.getElementById('close-settings-btn').onclick = () => document.getElementById('settings-modal').style.display = 'none';

document.getElementById('theme-select').onchange = (e) => {
    document.documentElement.setAttribute('data-theme', e.target.value);
    localStorage.setItem('userTheme', e.target.value);
};

document.getElementById('lang-select').onchange = (e) => {
    currentLang = e.target.value;
    localStorage.setItem('userLang', currentLang);
    updateUI();
};

document.getElementById('generate').onclick = generateTasks;

// ЛОГІКА ФІЛЬТРІВ СЛОВНИКА
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const filter = btn.dataset.filter;
        document.querySelectorAll('.word-card').forEach(card => {
            const isFav = card.querySelector('.favorite-btn').classList.contains('is-active');
            if(filter === 'all' || card.dataset.category === filter || (filter === 'favorites' && isFav)) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    };
});

// ІНІЦІАЛІЗАЦІЯ ПРИ ЗАВАНТАЖЕННІ
const savedUser = localStorage.getItem('currentUser');
if(savedUser) {
    document.getElementById('user-display-name').textContent = savedUser;
    navigateTo('app-screen');
} else {
    navigateTo('auth-screen');
}

const savedTheme = localStorage.getItem('userTheme');
if(savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);

updateUI();