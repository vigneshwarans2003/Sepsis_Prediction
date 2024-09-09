document.addEventListener('DOMContentLoaded', function() {
    const faqItems = document.querySelectorAll('.faq-item');

    faqItems.forEach(item => {
        item.querySelector('.faq-question').addEventListener('click', () => {
            item.classList.toggle('active');
        });
    });
});

document.getElementById('theme-toggle').addEventListener('click', function() {
    const body = document.body;
    const currentMode = body.classList.contains('light-mode') ? 'light' : 'dark';

    if (currentMode === 'light') {
        body.classList.remove('light-mode');
        body.classList.add('dark-mode');
        this.textContent = 'Switch to Light Mode';
    } else {
        body.classList.remove('dark-mode');
        body.classList.add('light-mode');
        this.textContent = 'Switch to Dark Mode';
    }

    // Save the user's preference in localStorage
    localStorage.setItem('theme', currentMode === 'light' ? 'dark' : 'light');
});

// Apply the saved theme on page load
window.addEventListener('load', function() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.classList.add(savedTheme + '-mode');
    document.getElementById('theme-toggle').textContent = savedTheme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode';
});