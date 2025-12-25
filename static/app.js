const queryInput = document.getElementById('query');
const searchBtn = document.getElementById('search-btn');
const resultsDiv = document.getElementById('results');
const statsDiv = document.getElementById('stats');
const topKSelect = document.getElementById('top-k');

// Load stats on page load
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.error) {
            statsDiv.textContent = 'No index loaded. Run the indexing pipeline first.';
        } else {
            statsDiv.className = 'stats loaded';
            statsDiv.textContent = `Index: ${data.total_items} items (${data.text_items} text, ${data.image_items} images) • ${data.documents} document(s) • ${data.embedder} embedder`;
        }
    } catch (err) {
        statsDiv.textContent = 'Could not connect to server.';
    }
}

// Get selected mode
function getMode() {
    const selected = document.querySelector('input[name="mode"]:checked');
    return selected ? selected.value : 'all';
}

// Perform search
async function search() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    const topK = parseInt(topKSelect.value);
    const mode = getMode();
    
    resultsDiv.innerHTML = '<div class="loading">Searching</div>';
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK, mode })
        });
        
        const data = await response.json();
        
        if (data.error) {
            resultsDiv.innerHTML = `<div class="empty">${data.error}</div>`;
            return;
        }
        
        if (!data.results || data.results.length === 0) {
            resultsDiv.innerHTML = '<div class="empty">No results found. Try a different query.</div>';
            return;
        }
        
        renderResults(data.results);
        
    } catch (err) {
        resultsDiv.innerHTML = '<div class="empty">Search failed. Is the server running?</div>';
    }
}

// Render results
function renderResults(results) {
    resultsDiv.innerHTML = results.map(result => {
        const isImage = result.modality === 'image';
        
        let content = '';
        if (isImage && result.image) {
            content = `<img class="result-image" src="data:image/png;base64,${result.image}" alt="Search result">`;
        } else if (result.content) {
            content = `<p class="result-content">${escapeHtml(result.content)}</p>`;
        }
        
        return `
            <div class="result-card">
                <div class="result-header">
                    <div class="result-meta">
                        <span class="badge ${result.modality}">${result.modality}</span>
                        <span class="result-source">${result.document} • Page ${result.page}</span>
                    </div>
                    <span class="score">${(result.score * 100).toFixed(1)}%</span>
                </div>
                ${content}
            </div>
        `;
    }).join('');
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

searchBtn.addEventListener('click', search);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') search();
});

loadStats();
