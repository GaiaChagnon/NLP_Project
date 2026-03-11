"""Netflix-style book recommender GUI powered by Gradio.

Loads the enriched book catalog, renders a dark-themed browsing
interface with genre rows, search, filtering, and a personal list.
Delegates NLP-powered recommendations to the FastAPI backend.
"""

import html as html_lib
import json as _json

import gradio as gr
import httpx
import pandas as pd

from recommender import load_config

# ═══════════════════════════════════════════════════════════════
# Configuration & Data
# ═══════════════════════════════════════════════════════════════

cfg = load_config()
API_URL = f"http://localhost:{cfg['api']['port']}"
GUI_PORT = cfg.get("gui", {}).get("port", 7860)

df = pd.read_csv(cfg["data"]["enriched"]).fillna("")
for col in ("published_year", "num_pages", "ratings_count"):
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
df["average_rating"] = pd.to_numeric(
    df["average_rating"], errors="coerce"
).fillna(0.0)

df = df[df["thumbnail"].str.strip().astype(bool)].copy()

_REMOVE_TITLES = {
    "The Hobbit, Or, There and Back Again",
    "Harry Potter and the Chamber of Secrets (Book 2)",
}
df = df[~df["title"].isin(_REMOVE_TITLES)].copy()

# Open Library Covers API: high-res (300-500px), 98%+ coverage, proper 404 on miss.
# ?default=false → returns 404 (triggers onerror) instead of a 1px placeholder.
df["thumbnail_ol"] = (
    "https://covers.openlibrary.org/b/isbn/"
    + df["isbn13"].astype(int).astype(str)
    + "-L.jpg?default=false"
)

BOOKS: dict[int, dict] = {
    int(row["isbn13"]): row.to_dict() for _, row in df.iterrows()
}

# Genre index: genre -> [isbn13 ...] sorted by rating descending
_genre_acc: dict[str, list[tuple[int, float]]] = {}
for _isbn, _bk in BOOKS.items():
    for _g in str(_bk.get("genres", "")).split(";"):
        _g = _g.strip()
        if _g:
            _genre_acc.setdefault(_g, []).append(
                (_isbn, float(_bk.get("average_rating", 0)))
            )

GENRES: dict[str, list[int]] = {}
for _g in sorted(_genre_acc, key=lambda k: len(_genre_acc[k]), reverse=True):
    _items = sorted(_genre_acc[_g], key=lambda x: x[1], reverse=True)
    GENRES[_g] = [isbn for isbn, _ in _items]

TOP_GENRES = list(GENRES.keys())[:12]

PLACEHOLDER = (
    "data:image/svg+xml;charset=utf-8,"
    "%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%27160%27 height=%27240%27%3E"
    "%3Crect width=%27160%27 height=%27240%27 fill=%27%23333%27/%3E"
    "%3Ctext x=%2780%27 y=%27120%27 text-anchor=%27middle%27 fill=%27%23777%27 "
    "font-family=%27sans-serif%27 font-size=%2713%27%3ENo Cover%3C/text%3E%3C/svg%3E"
)


def _esc(text) -> str:
    return html_lib.escape(str(text))


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════

CSS = """
/* ── Dark base ─────────────────────────────────────────────── */
.gradio-container, .gradio-container * {
    --background-fill-primary: #141414 !important;
    --background-fill-secondary: #1a1a1a !important;
}
.gradio-container {
    background: #141414 !important;
    max-width: 100% !important;
    padding: 0 !important;
    color: #e5e5e5 !important;
}
body, .dark { background: #141414 !important; }
footer, header { display: none !important; }
.contain, .block, .wrap, .panel { background: transparent !important; }

/* ── Gradio component overrides ────────────────────────────── */
#nav-row, #nav-row .block, #nav-row .wrap {
    background: #1a1a1a !important;
    border-bottom: 1px solid #333 !important;
    padding: 10px 3% !important;
    gap: 12px !important;
    position: sticky; top: 0; z-index: 1000;
}
#nav-row .gr-button { min-width: 110px !important; }

#home-btn {
    background: transparent !important;
    border: none !important;
    color: #E50914 !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    padding: 4px 8px !important;
}
#home-btn:hover { opacity: 0.85; }

#search-btn {
    background: #E50914 !important;
    border: none !important;
    color: #fff !important;
}
#search-btn:hover { background: #b20710 !important; }

#list-btn {
    background: transparent !important;
    border: 1px solid #666 !important;
    color: #e5e5e5 !important;
}
#list-btn:hover {
    border-color: #E50914 !important;
    color: #E50914 !important;
}

.gr-accordion, .accordion, [class*="accordion"] {
    background: #1a1a1a !important;
    border-color: #333 !important;
}
.gr-accordion .label-wrap, .accordion .label-wrap,
[class*="accordion"] .label-wrap { color: #aaa !important; }
.gr-accordion .block, .accordion .block { background: #1a1a1a !important; }

input, textarea, select {
    background: #2a2a2a !important;
    color: #e5e5e5 !important;
    border-color: #444 !important;
}
label span { color: #aaa !important; }

/* ── Hero banner (featured book at top) ────────────────────── */
.bf-hero {
    position: relative;
    width: 100%; height: 420px;
    overflow: hidden; cursor: pointer;
    margin-bottom: 30px;
}
.bf-hero-bg {
    position: absolute; inset: 0;
    background-size: cover; background-position: center;
    filter: blur(20px) brightness(0.3);
    transform: scale(1.1);
}
.bf-hero-inner {
    position: relative; z-index: 1;
    display: flex; align-items: center; gap: 36px;
    height: 100%; padding: 0 5%;
}
.bf-hero-cover { flex: 0 0 200px; height: 300px; border-radius: 8px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.6); }
.bf-hero-cover img { width: 100%; height: 100%; object-fit: cover; }
.bf-hero-info { flex: 1; color: #fff; }
.bf-hero-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; color: #E50914; font-weight: 700; margin-bottom: 8px; }
.bf-hero-title { font-size: 2.2rem; font-weight: 800; line-height: 1.15; margin: 0 0 10px; }
.bf-hero-meta { font-size: 0.95rem; color: #ccc; margin-bottom: 12px; display: flex; gap: 16px; flex-wrap: wrap; }
.bf-hero-desc { font-size: 0.9rem; line-height: 1.6; color: #bbb; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; max-width: 700px; }
.bf-hero-fade { position: absolute; bottom: 0; left: 0; right: 0; height: 80px; background: linear-gradient(transparent, #141414); z-index: 2; pointer-events: none; }

/* ── Section row (scrollable carousel) ─────────────────────── */
.bf-row { margin-bottom: 36px; position: relative; }

.bf-row-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e5e5e5;
    padding: 0 4% 12px;
    margin: 0;
}

.bf-scroll-wrap { position: relative; }

.bf-scroll {
    display: flex;
    overflow-x: auto;
    gap: 14px;
    padding: 4px 4% 12px;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}
.bf-scroll::-webkit-scrollbar { display: none; }

.bf-scroll .bf-card {
    flex: 0 0 calc((100% - 5 * 14px) / 6);
    min-width: 160px;
}

/* ── Carousel arrows ──────────────────────────────────────── */
.bf-arrow {
    position: absolute; top: 50%; transform: translateY(-50%);
    width: 44px; height: 80px;
    background: rgba(20,20,20,0.85);
    border: none; color: #fff;
    font-size: 1.6rem; cursor: pointer;
    z-index: 20; display: flex;
    align-items: center; justify-content: center;
    opacity: 0; transition: opacity 0.25s;
    border-radius: 4px;
}
.bf-scroll-wrap:hover .bf-arrow { opacity: 1; }
.bf-arrow:hover { background: rgba(40,40,40,0.95); }
.bf-arrow-l { left: 8px; }
.bf-arrow-r { right: 8px; }

/* ── Book card ─────────────────────────────────────────────── */
.bf-card {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    background: #222;
    aspect-ratio: 2/3;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.bf-card:hover {
    transform: scale(1.06);
    box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    z-index: 10;
}
.bf-card img {
    width: 100%; height: 100%;
    object-fit: cover;
    transition: filter 0.3s ease;
    image-rendering: auto;
    -webkit-backface-visibility: hidden;
}
.bf-card:hover img {
    filter: blur(4px) brightness(0.25);
}

/* ── Hover overlay ─────────────────────────────────────────── */
.bf-overlay {
    position: absolute; inset: 0;
    padding: 14px;
    display: flex; flex-direction: column; justify-content: flex-end;
    opacity: 0;
    transition: opacity 0.3s ease;
    color: #fff;
    pointer-events: none;
}
.bf-card:hover .bf-overlay { opacity: 1; }

.bf-overlay .bf-otitle {
    font-weight: 700; font-size: 0.9rem;
    line-height: 1.2; margin-bottom: 4px;
    overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.bf-overlay .bf-oauthor {
    font-size: 0.78rem; color: #ccc; margin-bottom: 3px;
}
.bf-overlay .bf-orating {
    font-size: 0.78rem; color: #ffd700; margin-bottom: 5px;
}
.bf-overlay .bf-odesc {
    font-size: 0.7rem; line-height: 1.35; color: #bbb;
    overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 4; -webkit-box-orient: vertical;
}

/* ── Rank badge ────────────────────────────────────────────── */
.bf-rank {
    position: absolute; top: 8px; left: 8px;
    background: #E50914; color: #fff;
    font-weight: 800; font-size: 0.8rem;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%; z-index: 5;
}

/* ── 6-column grid (search results / your list) ───────────── */
.bf-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 16px;
    padding: 0 4%;
}
.bf-grid .bf-card {
    width: 100%; height: auto;
    aspect-ratio: 2/3;
}

/* ── Section title ─────────────────────────────────────────── */
.bf-section-title {
    font-size: 1.4rem; font-weight: 700;
    color: #e5e5e5; padding: 0 4% 16px; margin: 0;
}

/* ── Book detail page ──────────────────────────────────────── */
.bf-detail {
    display: flex; gap: 40px;
    padding: 30px 4%; max-width: 1200px;
    margin: 0 auto;
}
.bf-detail-cover {
    flex: 0 0 280px; height: 420px;
    border-radius: 8px; overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}
.bf-detail-cover img {
    width: 100%; height: 100%; object-fit: cover;
    image-rendering: auto;
}
.bf-detail-info { flex: 1; color: #e5e5e5; }

.bf-detail-title {
    font-size: 2rem; font-weight: 800;
    margin: 0 0 6px; line-height: 1.15;
}
.bf-detail-subtitle {
    font-size: 1.05rem; color: #999; margin-bottom: 16px;
}
.bf-detail-meta {
    display: flex; gap: 18px; flex-wrap: wrap;
    margin-bottom: 20px; font-size: 0.95rem; color: #ccc;
}
.bf-detail-meta span {
    display: flex; align-items: center; gap: 5px;
}
.bf-detail-desc {
    font-size: 0.95rem; line-height: 1.65;
    color: #bbb; margin-bottom: 22px;
}
.bf-detail-tags {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px;
}
.bf-tag {
    background: #2a2a2a; color: #ddd;
    padding: 4px 12px; border-radius: 14px; font-size: 0.8rem;
}
.bf-tag.genre  { border-left: 3px solid #E50914; }
.bf-tag.kw     { border-left: 3px solid #ffd700; }
.bf-tag.cat    { border-left: 3px solid #4a90d9; }

.bf-add-btn, .bf-remove-btn {
    padding: 12px 28px; border-radius: 4px;
    font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
    border: none;
}
.bf-add-btn {
    background: #E50914; color: #fff;
}
.bf-add-btn:hover { background: #b20710; }
.bf-remove-btn {
    background: transparent; color: #e5e5e5;
    border: 2px solid #555;
}
.bf-remove-btn:hover { border-color: #E50914; color: #E50914; }

.bf-back-btn {
    background: transparent; color: #999; border: none;
    font-size: 0.95rem; cursor: pointer; padding: 0;
    margin-bottom: 18px;
}
.bf-back-btn:hover { color: #fff; }

/* ── Empty state ───────────────────────────────────────────── */
.bf-empty {
    text-align: center; padding: 80px 20px;
    color: #666; font-size: 1.05rem;
}

/* ── Hidden triggers (must be in DOM for JS bridge) ────────── */
#hidden-triggers {
    position: fixed !important; left: -9999px !important;
    width: 1px !important; height: 1px !important;
    overflow: hidden !important; opacity: 0 !important;
    pointer-events: none !important;
}

/* ── Responsive ────────────────────────────────────────────── */
@media (max-width: 1200px) {
    .bf-grid { grid-template-columns: repeat(5, 1fr); }
    .bf-scroll .bf-card { flex: 0 0 calc((100% - 4 * 14px) / 5); }
}
@media (max-width: 900px) {
    .bf-grid { grid-template-columns: repeat(4, 1fr); }
    .bf-scroll .bf-card { flex: 0 0 calc((100% - 3 * 14px) / 4); }
    .bf-detail { flex-direction: column; align-items: center; }
    .bf-detail-cover { flex: 0 0 auto; width: 220px; height: 330px; }
    .bf-hero { height: 340px; }
    .bf-hero-title { font-size: 1.5rem; }
    .bf-hero-cover { flex: 0 0 140px; height: 210px; }
}
@media (max-width: 600px) {
    .bf-grid { grid-template-columns: repeat(3, 1fr); }
    .bf-scroll .bf-card { flex: 0 0 calc((100% - 2 * 14px) / 3); }
    .bf-hero-inner { flex-direction: column; text-align: center; padding-top: 30px; }
}
"""

# ═══════════════════════════════════════════════════════════════
# JavaScript Bridge (Gradio ↔ custom HTML click events)
# ═══════════════════════════════════════════════════════════════

_LOAD_JS = """() => {
    if (window._bfReady) return;
    window._bfReady = true;

    var _bridgePort = 7861;

    async function _post(path, body) {
        try {
            var r = await fetch('http://localhost:' + _bridgePort + '/bookflix' + path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!r.ok) return null;
            return await r.json();
        } catch(e) { console.error('BookFlix error:', e); }
        return null;
    }

    function _updateHTML(html) {
        var containers = document.querySelectorAll('[id*="html"]');
        containers.forEach(function(el) {
            var inner = el.querySelector('.prose') || el.querySelector('.html-container') || el;
            if (inner && inner.innerHTML && inner.innerHTML.includes('bf-')) {
                inner.innerHTML = html;
            }
        });
    }

    var _userList = [];

    function _updateListBtn() {
        var b = document.getElementById('list-btn');
        if (b) b.textContent = 'Your List (' + _userList.length + ')';
    }

    window.selectBook = async function(isbn) {
        var r = await _post('/select', { isbn: String(isbn), user_list: _userList });
        if (r && r.html) { _updateHTML(r.html); window.scrollTo(0, 0); }
    };
    window.addToList = async function(isbn) {
        var r = await _post('/add', { isbn: String(isbn), user_list: _userList });
        if (r) {
            _userList = r.user_list || _userList;
            if (r.html) _updateHTML(r.html);
            _updateListBtn();
        }
    };
    window.removeFromList = async function(isbn) {
        var r = await _post('/remove', { isbn: String(isbn), user_list: _userList });
        if (r) {
            _userList = r.user_list || _userList;
            if (r.html) _updateHTML(r.html);
            _updateListBtn();
        }
    };
    window.goHome = async function() {
        var r = await _post('/home', { user_list: _userList });
        if (r && r.html) { _updateHTML(r.html); window.scrollTo(0, 0); }
    };
    window.showList = async function() {
        var r = await _post('/your_list', { user_list: _userList });
        if (r && r.html) { _updateHTML(r.html); window.scrollTo(0, 0); }
    };

    /* Intercept Gradio button clicks so they use the JS bridge
       (which carries the real _userList) instead of Gradio State. */
    var _homeBtn = document.getElementById('home-btn');
    if (_homeBtn) _homeBtn.addEventListener('click', function(e) {
        e.stopImmediatePropagation(); window.goHome();
    }, true);
    var _listBtn = document.getElementById('list-btn');
    if (_listBtn) _listBtn.addEventListener('click', function(e) {
        e.stopImmediatePropagation(); window.showList();
    }, true);
}"""

# ═══════════════════════════════════════════════════════════════
# HTML Rendering Helpers
# ═══════════════════════════════════════════════════════════════


def _img_attrs(thumb_ol: str, thumb_gb: str) -> tuple[str, str]:
    """Build onerror and onload attributes for the OL→GB fallback chain.

    onerror: OL 404 → try Google Books → hide card.
    onload:  if OL loads but the image is clearly wrong (landscape /
             tiny / not a book cover), swap to Google Books.
    """
    onerror = (
        f"if(!this.dataset.retry){{"
        f"this.dataset.retry='1';this.src='{thumb_gb}'"
        f"}}else{{this.closest('.bf-card')&&"
        f"(this.closest('.bf-card').style.display='none')}}"
    )
    onload = (
        f"if(!this.dataset.retry&&this.naturalWidth>0){{"
        f"var r=this.naturalHeight/this.naturalWidth;"
        f"if(r<1.1||this.naturalWidth<30){{"
        f"this.dataset.retry='1';this.src='{thumb_gb}'"
        f"}}}}"
    )
    return onerror, onload


def _card_html(book: dict, rank: int | None = None) -> str:
    """Single book card with hover overlay.

    Image chain: Open Library -L (high-res) → Google Books zoom=1 → hide card.
    onload crosscheck swaps to Google Books if OL returns a non-book image.
    """
    isbn = int(book.get("isbn13", 0))
    title = _esc(book.get("title", "Unknown"))
    author = _esc(book.get("authors", "Unknown"))
    rating = float(book.get("average_rating", 0))
    desc = _esc(str(book.get("description", ""))[:150])
    thumb_ol = book.get("thumbnail_ol", "")
    thumb_gb = book.get("thumbnail", "")

    if not thumb_ol and not thumb_gb:
        return ""

    primary = thumb_ol or thumb_gb
    rank_html = f'<div class="bf-rank">{rank}</div>' if rank else ""
    onerror, onload = _img_attrs(thumb_ol, thumb_gb)

    return (
        f'<div class="bf-card" onclick="selectBook(\'{isbn}\')">'
        f'<img src="{primary}" loading="lazy" alt="{title}" '
        f'onerror="{onerror}" onload="{onload}">'
        f"{rank_html}"
        f'<div class="bf-overlay">'
        f'<div class="bf-otitle">{title}</div>'
        f'<div class="bf-oauthor">{author}</div>'
        f'<div class="bf-orating">★ {rating:.1f}</div>'
        f'<div class="bf-odesc">{desc}</div>'
        f"</div></div>"
    )


def _card_isbn(isbn: int) -> str:
    return _card_html(BOOKS.get(isbn, {"isbn13": isbn}))


_row_counter = 0


def _row_html(title: str, cards: str) -> str:
    """Horizontally scrollable carousel row with navigation arrows."""
    global _row_counter
    _row_counter += 1
    rid = f"bfr{_row_counter}"
    return (
        f'<div class="bf-row">'
        f'<h3 class="bf-row-title">{_esc(title)}</h3>'
        f'<div class="bf-scroll-wrap">'
        f'<button class="bf-arrow bf-arrow-l" '
        f"onclick=\"document.getElementById('{rid}').scrollBy({{left:-600,behavior:'smooth'}})\""
        f'>&lsaquo;</button>'
        f'<div class="bf-scroll" id="{rid}">{cards}</div>'
        f'<button class="bf-arrow bf-arrow-r" '
        f"onclick=\"document.getElementById('{rid}').scrollBy({{left:600,behavior:'smooth'}})\""
        f'>&rsaquo;</button>'
        f"</div></div>"
    )


ITEMS_PER_ROW = 20


def _render_hero(book: dict, label: str = "Featured") -> str:
    """Large Netflix-style hero banner for a single book."""
    isbn = int(book.get("isbn13", 0))
    title = _esc(book.get("title", ""))
    author = _esc(book.get("authors", ""))
    desc = _esc(str(book.get("description", ""))[:300])
    thumb_ol = book.get("thumbnail_ol", "")
    thumb_gb = book.get("thumbnail", "")
    primary = thumb_ol or thumb_gb
    rating = float(book.get("average_rating", 0))
    count = int(book.get("ratings_count", 0))
    year = int(book.get("published_year", 0))
    year_html = f"<span>{year}</span>" if year > 0 else ""

    onerror = (
        f"if(!this.dataset.retry){{"
        f"this.dataset.retry='1';this.src='{thumb_gb}'"
        f"}}"
    )
    onload = (
        f"if(!this.dataset.retry&&this.naturalWidth>0){{"
        f"var r=this.naturalHeight/this.naturalWidth;"
        f"if(r<1.1||this.naturalWidth<30){{"
        f"this.dataset.retry='1';this.src='{thumb_gb}'"
        f"}}}}"
    )

    return (
        f'<div class="bf-hero" onclick="selectBook(\'{isbn}\')">'
        f'<div class="bf-hero-bg" style="background-image:url(\'{primary}\')"></div>'
        f'<div class="bf-hero-inner">'
        f'<div class="bf-hero-cover"><img src="{primary}" '
        f'onerror="{onerror}" onload="{onload}" alt="{title}"></div>'
        f'<div class="bf-hero-info">'
        f'<div class="bf-hero-label">{_esc(label)}</div>'
        f'<h1 class="bf-hero-title">{title}</h1>'
        f'<div class="bf-hero-meta">'
        f"<span>★ {rating:.1f} ({count:,} ratings)</span>"
        f"<span>By {author}</span>"
        f"{year_html}"
        f"</div>"
        f'<div class="bf-hero-desc">{desc}</div>'
        f"</div></div>"
        f'<div class="bf-hero-fade"></div>'
        f"</div>"
    )


def _unique_isbns(candidates: list[int], seen: set[int], limit: int) -> list[int]:
    """Return up to *limit* ISBNs from *candidates*, skipping any in *seen*.
    Adds returned ISBNs to *seen* in-place so later rows won't repeat them.
    """
    out: list[int] = []
    for isbn in candidates:
        if isbn in seen:
            continue
        out.append(isbn)
        seen.add(isbn)
        if len(out) >= limit:
            break
    return out


def _round_robin_genres(
    genres: list[str],
    genre_index: dict[str, list[int]],
    seen: set[int],
    per_row: int,
) -> dict[str, list[int]]:
    """Fill genre rows round-robin: pick slot 1 for every genre, then slot 2,
    etc. This distributes quality evenly instead of front-loading popular
    books into the first few rows.
    """
    result: dict[str, list[int]] = {g: [] for g in genres}
    cursors: dict[str, int] = {g: 0 for g in genres}

    for _slot in range(per_row):
        for g in genres:
            if len(result[g]) >= per_row:
                continue
            pool = genre_index.get(g, [])
            while cursors[g] < len(pool):
                isbn = pool[cursors[g]]
                cursors[g] += 1
                if isbn not in seen:
                    result[g].append(isbn)
                    seen.add(isbn)
                    break
    return result


def _render_home(user_list: list[int]) -> str:
    """Full home view: hero + Recommended for You + Popular + genre rows.

    A global *seen* set ensures no book appears in more than one section.
    Genre rows are filled round-robin so every category gets evenly
    distributed quality instead of front-loading popular books.
    """
    global _row_counter
    _row_counter = 0
    parts: list[str] = []
    rec_books: list[dict] = []
    seen: set[int] = set()

    if user_list:
        rec_books = _get_suggestions(user_list)

    # Hero banner
    if rec_books:
        hero_isbn = int(rec_books[0].get("isbn13", 0))
        seen.add(hero_isbn)
        parts.append(_render_hero(rec_books[0], label="Recommended for You"))
    else:
        hero_df = df[df["ratings_count"] >= 100].nlargest(1, "average_rating")
        if not hero_df.empty:
            hero_isbn = int(hero_df.iloc[0]["isbn13"])
            seen.add(hero_isbn)
            parts.append(_render_hero(
                BOOKS.get(hero_isbn, hero_df.iloc[0].to_dict()),
                label="Top Rated",
            ))

    # Recommended for You row (skip the hero book)
    if rec_books:
        rec_isbns: list[int] = []
        rec_cards_parts: list[str] = []
        for s in rec_books[1:]:
            isbn = int(s.get("isbn13", 0))
            if isbn in seen:
                continue
            seen.add(isbn)
            rec_cards_parts.append(_card_html(s, rank=len(rec_isbns) + 1))
            rec_isbns.append(isbn)
            if len(rec_isbns) >= ITEMS_PER_ROW:
                break
        if rec_cards_parts:
            parts.append(_row_html("Recommended for You", "".join(rec_cards_parts)))

    # Top Rated = best combination of high rating AND meaningful review count.
    # Sort by average_rating among books with 500+ reviews so obscure 5-star
    # books with 3 reviews don't dominate.
    tr_candidates = (
        df[df["ratings_count"] >= 500]
        .nlargest(ITEMS_PER_ROW * 3, "average_rating")["isbn13"]
        .astype(int)
        .tolist()
    )
    tr_isbns = _unique_isbns(tr_candidates, seen, ITEMS_PER_ROW)
    if tr_isbns:
        parts.append(_row_html(
            "Top Rated",
            "".join(_card_isbn(i) for i in tr_isbns),
        ))

    # Genre rows — round-robin fill so every genre gets evenly good books
    genre_rows = _round_robin_genres(TOP_GENRES, GENRES, seen, ITEMS_PER_ROW)
    for genre in TOP_GENRES:
        isbns = genre_rows[genre]
        if isbns:
            parts.append(_row_html(
                genre,
                "".join(_card_isbn(i) for i in isbns),
            ))

    return '<div style="padding:20px 0">' + "".join(parts) + "</div>"


def _render_search(results: list[dict], query: str) -> str:
    """Search results in a ranked 6-column grid."""
    if not results:
        return (
            f'<div style="padding:20px 0">'
            f'<h2 class="bf-section-title">'
            f'No results for "{_esc(query)}"</h2>'
            f'<div class="bf-empty">Try different keywords or '
            f"adjust your filters.</div></div>"
        )

    cards = "".join(
        _card_html(r, rank=i + 1) for i, r in enumerate(results)
    )
    return (
        f'<div style="padding:20px 0">'
        f'<h2 class="bf-section-title">'
        f'Results for "{_esc(query)}" ({len(results)} found)</h2>'
        f'<div class="bf-grid">{cards}</div></div>'
    )


def _render_detail(isbn: int, user_list: list[int]) -> str:
    """Full book detail page with metadata and list action."""
    book = BOOKS.get(isbn)
    if not book:
        return '<div class="bf-empty">Book not found.</div>'

    title = _esc(book.get("title", ""))
    subtitle = _esc(book.get("subtitle", ""))
    author = _esc(book.get("authors", ""))
    desc = _esc(book.get("description", ""))
    thumb_ol = book.get("thumbnail_ol", "")
    thumb_gb = book.get("thumbnail", "")
    primary = thumb_ol or thumb_gb
    rating = float(book.get("average_rating", 0))
    count = int(book.get("ratings_count", 0))
    year = int(book.get("published_year", 0))
    pages = int(book.get("num_pages", 0))

    tags_html = ""
    for g in str(book.get("genres", "")).split(";"):
        g = g.strip()
        if g:
            tags_html += f'<span class="bf-tag genre">{_esc(g)}</span>'
    for k in str(book.get("keywords", "")).split(";"):
        k = k.strip()
        if k:
            tags_html += f'<span class="bf-tag kw">{_esc(k)}</span>'
    for c in str(book.get("categories", "")).split(";"):
        c = c.strip()
        if c:
            tags_html += f'<span class="bf-tag cat">{_esc(c)}</span>'

    sub_html = (
        f'<div class="bf-detail-subtitle">{subtitle}</div>' if subtitle else ""
    )
    year_html = f"<span>Published: {year}</span>" if year > 0 else ""
    pages_html = f"<span>{pages} pages</span>" if pages > 0 else ""

    if isbn in user_list:
        btn = (
            f'<button class="bf-remove-btn" '
            f"onclick=\"removeFromList('{isbn}')\">"
            f"In Your List (Remove)</button>"
        )
    else:
        btn = (
            f'<button class="bf-add-btn" '
            f"onclick=\"addToList('{isbn}')\">"
            f"+ Add to Your List</button>"
        )

    detail_onerror = (
        f"if(!this.dataset.retry){{"
        f"this.dataset.retry='1';this.src='{thumb_gb}'"
        f"}}"
    )
    detail_onload = (
        f"if(!this.dataset.retry&&this.naturalWidth>0){{"
        f"var r=this.naturalHeight/this.naturalWidth;"
        f"if(r<1.1||this.naturalWidth<30){{"
        f"this.dataset.retry='1';this.src='{thumb_gb}'"
        f"}}}}"
    )

    return (
        f'<div class="bf-detail">'
        f'<div class="bf-detail-cover">'
        f'<img src="{primary}" onerror="{detail_onerror}" '
        f'onload="{detail_onload}" alt="{title}">'
        f"</div>"
        f'<div class="bf-detail-info">'
        f'<button class="bf-back-btn" onclick="goHome()">Back to Browse</button>'
        f'<h1 class="bf-detail-title">{title}</h1>'
        f"{sub_html}"
        f'<div class="bf-detail-meta">'
        f"<span>★ {rating:.2f} ({count:,} ratings)</span>"
        f"<span>By {author}</span>"
        f"{year_html}{pages_html}"
        f"</div>"
        f'<p class="bf-detail-desc">{desc}</p>'
        f'<div class="bf-detail-tags">{tags_html}</div>'
        f"{btn}"
        f"</div></div>"
    )


def _render_list(user_list: list[int]) -> str:
    """Your List view — 6-column grid of saved books."""
    if not user_list:
        return (
            '<div style="padding:20px 0">'
            '<h2 class="bf-section-title">Your List</h2>'
            '<div class="bf-empty">'
            "Your list is empty.<br>"
            'Browse books and click "Add to Your List" to save them here.'
            "</div></div>"
        )

    cards = "".join(
        _card_isbn(isbn) for isbn in user_list if isbn in BOOKS
    )
    return (
        f'<div style="padding:20px 0">'
        f'<h2 class="bf-section-title">'
        f"Your List ({len(user_list)} books)</h2>"
        f'<div class="bf-grid">{cards}</div></div>'
    )


# ═══════════════════════════════════════════════════════════════
# API Communication
# ═══════════════════════════════════════════════════════════════


def _enrich_results(results: list[dict]) -> list[dict]:
    """Merge API recommendation dicts with local BOOKS data (adds thumbnail_ol)."""
    enriched = []
    for r in results:
        isbn = int(r.get("isbn13", 0))
        local = BOOKS.get(isbn)
        if local:
            merged = {**local, **r}
        else:
            merged = dict(r)
            if "thumbnail_ol" not in merged:
                merged["thumbnail_ol"] = (
                    f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false"
                )
        enriched.append(merged)
    return enriched


def _api_recommend(
    query: str,
    book_ids: list[int] | None = None,
    n: int = 20,
) -> list[dict]:
    """Call the FastAPI /recommend endpoint. Returns [] on failure."""
    try:
        payload: dict = {"query": query, "n": n}
        if book_ids is not None:
            payload["book_ids"] = book_ids
        resp = httpx.post(
            f"{API_URL}/recommend", json=payload, timeout=30.0,
        )
        resp.raise_for_status()
        return _enrich_results(resp.json().get("recommendations", []))
    except Exception:
        return []


def _get_suggestions(user_list: list[int]) -> list[dict]:
    """Build a composite query from liked books' genres + keywords,
    then call the recommender for similar titles."""
    liked = [BOOKS[isbn] for isbn in user_list if isbn in BOOKS]
    if not liked:
        return []

    parts: list[str] = []
    for b in liked:
        for col in ("genres", "keywords"):
            for item in str(b.get(col, "")).split(";"):
                item = item.strip()
                if item:
                    parts.append(item)

    query = "; ".join(parts[:60])
    if not query:
        return []

    exclude = set(user_list)
    recs = _api_recommend(query, n=30)
    return [r for r in recs if r.get("isbn13") not in exclude][:20]


# ═══════════════════════════════════════════════════════════════
# Gradio Callbacks
# ═══════════════════════════════════════════════════════════════


def _parse_trigger(value: str) -> int | None:
    """Extract ISBN from JS trigger value ('isbn|timestamp')."""
    try:
        return int(str(value).split("|")[0])
    except (ValueError, IndexError):
        return None


def on_search(query, min_pg, max_pg, min_rate, sort_by, n_res, user_list):
    """Filter catalog, call recommender API, apply sort, return HTML."""
    query = str(query).strip()
    if not query:
        return _render_home(user_list)

    filt = df.copy()
    if min_pg > 0:
        filt = filt[filt["num_pages"] >= min_pg]
    if max_pg < 2000:
        filt = filt[filt["num_pages"] <= max_pg]
    if min_rate > 0:
        filt = filt[filt["average_rating"] >= min_rate]

    candidate_ids = filt["isbn13"].astype(int).tolist()
    if not candidate_ids:
        return _render_search([], query)

    results = _api_recommend(query, book_ids=candidate_ids, n=int(n_res))

    # Fallback: local text search when API is unavailable
    if not results:
        q_lower = query.lower()
        matches = []
        for isbn in candidate_ids:
            b = BOOKS.get(isbn, {})
            haystack = " ".join(
                str(b.get(c, "")) for c in
                ("title", "authors", "genres", "keywords", "categories", "description")
            ).lower()
            if q_lower in haystack:
                matches.append(b)
        results = matches[:int(n_res)]

    if sort_by == "Alphabetical":
        results.sort(key=lambda r: str(r.get("title", "")).lower())
    elif sort_by == "Rating":
        results.sort(
            key=lambda r: float(r.get("average_rating", 0)), reverse=True,
        )
    elif sort_by == "Newest":
        results.sort(
            key=lambda r: int(
                BOOKS.get(r.get("isbn13", 0), {}).get("published_year", 0)
            ),
            reverse=True,
        )

    return _render_search(results, query)


def on_book_select(trigger_val, user_list):
    isbn = _parse_trigger(trigger_val)
    if isbn is None or isbn not in BOOKS:
        return _render_home(user_list)
    return _render_detail(isbn, user_list)


def on_list_add(trigger_val, user_list):
    isbn = _parse_trigger(trigger_val)
    if isbn is not None and isbn in BOOKS and isbn not in user_list:
        user_list = user_list + [isbn]
    html = _render_detail(isbn, user_list) if isbn and isbn in BOOKS else _render_home(user_list)
    return user_list, html, gr.update(value=f"Your List ({len(user_list)})")


def on_list_remove(trigger_val, user_list):
    isbn = _parse_trigger(trigger_val)
    if isbn is not None:
        user_list = [x for x in user_list if x != isbn]
    html = _render_detail(isbn, user_list) if isbn and isbn in BOOKS else _render_home(user_list)
    return user_list, html, gr.update(value=f"Your List ({len(user_list)})")


def on_go_home(_trigger, user_list):
    return _render_home(user_list)




# ═══════════════════════════════════════════════════════════════
# Gradio App Layout
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(title="BookFlix") as app:
    user_list_state = gr.State([])

    # Hidden JS→Python triggers (visible to DOM, hidden by CSS)
    with gr.Row(elem_id="hidden-triggers"):
        book_select_t = gr.Textbox(elem_id="book_select", show_label=False)
        list_add_t = gr.Textbox(elem_id="list_add", show_label=False)
        list_remove_t = gr.Textbox(elem_id="list_remove", show_label=False)
        go_home_t = gr.Textbox(elem_id="go_home", show_label=False)

    # ── Navigation bar ────────────────────────────────────────
    with gr.Row(elem_id="nav-row"):
        home_btn = gr.Button("BookFlix", elem_id="home-btn", scale=0, min_width=120)
        search_box = gr.Textbox(
            placeholder="Search by title, author, genre, keyword...",
            show_label=False,
            scale=4,
        )
        search_btn = gr.Button("Search", elem_id="search-btn", scale=0, min_width=100)
        list_btn = gr.Button("Your List (0)", elem_id="list-btn", scale=0, min_width=140)

    # ── Filter / sort bar ─────────────────────────────────────
    with gr.Accordion("Filters & Sort", open=False):
        with gr.Row():
            min_pages = gr.Slider(0, 2000, 0, step=10, label="Min Pages")
            max_pages = gr.Slider(0, 2000, 2000, step=10, label="Max Pages")
            min_rating = gr.Slider(0, 5, 0, step=0.5, label="Min Rating")
            sort_by = gr.Dropdown(
                ["Relevance", "Alphabetical", "Rating", "Newest"],
                value="Relevance",
                label="Sort By",
            )
            n_results = gr.Slider(5, 50, 20, step=5, label="Max Results")

    # ── Main content area ─────────────────────────────────────
    content = gr.HTML(value=_render_home([]))

    # ── Event wiring ──────────────────────────────────────────
    _search_inputs = [
        search_box, min_pages, max_pages, min_rating,
        sort_by, n_results, user_list_state,
    ]
    search_btn.click(on_search, _search_inputs, [content])
    search_box.submit(on_search, _search_inputs, [content])

    book_select_t.input(
        on_book_select, [book_select_t, user_list_state], [content],
    )
    list_add_t.input(
        on_list_add, [list_add_t, user_list_state],
        [user_list_state, content, list_btn],
    )
    list_remove_t.input(
        on_list_remove, [list_remove_t, user_list_state],
        [user_list_state, content, list_btn],
    )
    go_home_t.input(
        on_go_home, [go_home_t, user_list_state], [content],
    )

    home_btn.click(
        fn=lambda ul: _render_home(ul),
        inputs=[user_list_state],
        outputs=[content],
    )
    list_btn.click(
        fn=lambda ul: _render_list(ul),
        inputs=[user_list_state],
        outputs=[content],
    )

    # Inject JS bridge on page load
    app.load(fn=None, js=_LOAD_JS)


# ═══════════════════════════════════════════════════════════════
# Custom FastAPI routes (mounted on the Gradio server, no SSE)
# ═══════════════════════════════════════════════════════════════


def _handle_bookflix(path: str, body: dict) -> dict | None:
    """Process /bookflix/* requests. Returns response dict or None."""
    user_list = [int(x) for x in body.get("user_list", []) if str(x).isdigit()]

    if path == "/select":
        isbn = _parse_trigger(body.get("isbn", ""))
        if isbn is None or isbn not in BOOKS:
            return {"html": _render_home(user_list)}
        return {"html": _render_detail(isbn, user_list)}

    if path == "/add":
        isbn = _parse_trigger(body.get("isbn", ""))
        ul = list(user_list)
        if isbn is not None and isbn in BOOKS and isbn not in ul:
            ul.append(isbn)
        html = _render_detail(isbn, ul) if isbn and isbn in BOOKS else _render_home(ul)
        return {"user_list": ul, "html": html}

    if path == "/remove":
        isbn = _parse_trigger(body.get("isbn", ""))
        ul = [x for x in user_list if x != isbn]
        html = _render_detail(isbn, ul) if isbn and isbn in BOOKS else _render_home(ul)
        return {"user_list": ul, "html": html}

    if path == "/home":
        return {"html": _render_home(user_list)}

    if path == "/your_list":
        return {"html": _render_list(user_list)}

    return None


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=GUI_PORT, css=CSS)
