// ── Routing button state ──
const btn = document.getElementById('go-btn');
const form = document.getElementById('route-form');
form.addEventListener('submit', () => {
    btn.classList.add('loading');
    btn.querySelector('span').textContent = 'Routing…';
});

// ── Nominatim autocomplete ──
let BBOX = null
let FULL_PLACE_NAME = null

const get_bbox = (async () => {
    const res = await fetch("/city-metadata");
    const data = await res.json();
    BBOX = data["bbox-bounds"];
    FULL_PLACE_NAME = data["name"]
})()


function setupAutocomplete(inputId, suggestionsId) {
    const input = document.getElementById(inputId);
    const list = document.getElementById(suggestionsId);
    let timer = null;
    let selected = false;  // whether user clicked a suggestion

    input.addEventListener('input', () => {
        clearTimeout(timer);
        const q = input.value.trim();
        if (q.length < 3) { list.innerHTML = ''; list.classList.remove('open'); return; }
        timer = setTimeout(() => fetchSuggestions(q, list, input), 300);
    });

    input.addEventListener('blur', () => {
        // small delay so click on suggestion fires first
        setTimeout(() => { list.innerHTML = ''; list.classList.remove('open'); }, 150);
    });
}

async function fetchSuggestions(query, list, input) {
    const url = new URL('https://nominatim.openstreetmap.org/search');
    url.searchParams.set('q', `${query}, ${FULL_PLACE_NAME}`);
    url.searchParams.set('format', 'json');
    url.searchParams.set('limit', '6');
    url.searchParams.set('viewbox', `${BBOX.west},${BBOX.north},${BBOX.east},${BBOX.south}`);
    url.searchParams.set('bounded', '1');
    url.searchParams.set('addressdetails', '1');

    try {
        const res = await fetch(url, { headers: { 'Accept-Language': 'en' } });
        const data = await res.json();

        list.innerHTML = '';
        if (!data.length) { list.classList.remove('open'); return; }

        data.forEach(place => {
            // Build a short readable label
            const addr = place.address || {};
            const parts = [
                place.name || addr.amenity || addr.road,
                addr.suburb || addr.neighbourhood || addr.village,
                addr.city || addr.town,
            ].filter(Boolean);
            const label = [...new Set(parts)].slice(0, 3).join(', ');

            const li = document.createElement('li');
            li.textContent = label || place.display_name;
            li.addEventListener('mousedown', () => {
                input.value = label || place.display_name;
                list.innerHTML = '';
                list.classList.remove('open');
            });
            list.appendChild(li);
        });

        list.classList.add('open');
    } catch (e) {
        console.warn('Nominatim error', e);
    }
}

setupAutocomplete('start', 'start-suggestions');
setupAutocomplete('end', 'end-suggestions');