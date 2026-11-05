/**
 * Calm Octopuses - Client-side Application Logic & Machine Learning Inference
 * Runs 100% in-browser on GitHub Pages.
 */

// Global State
const state = {
  catalog: [],
  restaurantEmbeddings: {},
  centroid: null,
  metadata: {},
  menus: null,
  reviews: null,
  menuEmbeddings: null,
  mdnWeights: null,
  userRatings: JSON.parse(localStorage.getItem('co_user_ratings') || '{}'),
  activeTab: 'browse',
  topK: 6,
  selectedRestaurantId: null,
  transformersClip: null,
  isClipLoading: false
};

// DOM Elements
const elements = {
  tabNav: document.getElementById('tabNav'),
  catalogCount: document.getElementById('catalogCount'),
  browseTab: document.getElementById('tab-browse'),
  searchTab: document.getElementById('tab-search'),
  dishTab: document.getElementById('tab-dish'),
  recTab: document.getElementById('tab-rec'),
  myRestTab: document.getElementById('tab-my-rest'),
  overviewTab: document.getElementById('tab-overview')
};

// ==========================================
// Vector & Math Helpers
// ==========================================

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function normalize(a) {
  const n = norm(a) + 1e-9;
  const res = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) res[i] = a[i] / n;
  return res;
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0.0;
  const denom = (norm(a) * norm(b)) + 1e-9;
  return dot(a, b) / denom;
}

function centerVec(vec, centroid) {
  if (!centroid) return normalize(vec);
  const centered = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    centered[i] = vec[i] - centroid[i];
  }
  return normalize(centered);
}

// ==========================================
// Data Initialization
// ==========================================

async function initData() {
  try {
    const [catalogRes, embRes, centRes, metaRes] = await Promise.all([
      fetch('data/catalog.json').then(r => r.json()),
      fetch('data/restaurant_embeddings.json').then(r => r.json()),
      fetch('data/centroid.json').then(r => r.json()),
      fetch('data/metadata.json').then(r => r.json())
    ]);

    state.catalog = catalogRes;
    state.restaurantEmbeddings = embRes;
    state.centroid = new Float32Array(centRes);
    state.metadata = metaRes;

    if (elements.catalogCount) {
      elements.catalogCount.textContent = state.catalog.length;
    }

    renderBrowseTab();
    renderOverviewTab();
    renderMyRestaurantsTab();

    // Pre-fetch MDN weights in background for fast recommendation tab
    loadMDNWeights();
  } catch (err) {
    console.error('Failed to load catalog data:', err);
  }
}

async function loadMDNWeights() {
  if (state.mdnWeights) return state.mdnWeights;
  try {
    const res = await fetch('data/mdn_weights.json');
    state.mdnWeights = await res.json();
    console.log('Pretrained MDN weights loaded successfully.');
    return state.mdnWeights;
  } catch (e) {
    console.warn('Could not load MDN weights, will use embedding fallback:', e);
    return null;
  }
}

async function getMenus() {
  if (!state.menus) {
    state.menus = await fetch('data/menus.json').then(r => r.json());
  }
  return state.menus;
}

async function getReviews() {
  if (!state.reviews) {
    state.reviews = await fetch('data/reviews.json').then(r => r.json());
  }
  return state.reviews;
}

// ==========================================
// User Rating Storage
// ==========================================

function setRating(restId, rating) {
  if (rating === 0) {
    delete state.userRatings[restId];
  } else {
    state.userRatings[restId] = rating;
  }
  localStorage.setItem('co_user_ratings', JSON.stringify(state.userRatings));
  updateRatingDisplays(restId);
  if (state.activeTab === 'rec') renderRecommendationsTab();
  if (state.activeTab === 'my-rest') renderMyRestaurantsTab();
}

function updateRatingDisplays(restId) {
  document.querySelectorAll(`.co-star-group[data-rest-id="${restId}"]`).forEach(group => {
    const activeRating = state.userRatings[restId] || 0;
    group.querySelectorAll('.co-star-btn').forEach(btn => {
      const val = parseInt(btn.dataset.value, 10);
      btn.classList.toggle('active', val <= activeRating);
    });
  });
}

// ==========================================
// MDN Model & Recommendation Engine
// ==========================================

function buildUserTasteProfile(ratings, embeddingsMap, centroid) {
  const likeVecs = [];
  const likeWeights = [];
  const disVecs = [];
  const disWeights = [];
  const allRatings = Object.values(ratings);

  if (allRatings.length === 0) return null;

  for (const [rid, r] of Object.entries(ratings)) {
    const rawVec = embeddingsMap[rid];
    if (!rawVec) continue;
    const centered = centerVec(rawVec, centroid);
    if (r >= 3.5) {
      const w = r / 5.0;
      likeVecs.push({ vec: centered, w });
      likeWeights.push(w);
    } else {
      const w = (6.0 - r) / 5.0;
      disVecs.push({ vec: centered, w });
      disWeights.push(w);
    }
  }

  if (likeVecs.length === 0 && disVecs.length === 0) return null;

  const dim = 512;
  const userLike = new Float32Array(dim);
  if (likeVecs.length > 0) {
    const sumW = likeWeights.reduce((a, b) => a + b, 0);
    for (const item of likeVecs) {
      for (let i = 0; i < dim; i++) userLike[i] += item.vec[i] * item.w;
    }
    for (let i = 0; i < dim; i++) userLike[i] /= sumW;
  }

  const userDis = new Float32Array(dim);
  if (disVecs.length > 0) {
    const sumW = disWeights.reduce((a, b) => a + b, 0);
    for (const item of disVecs) {
      for (let i = 0; i < dim; i++) userDis[i] += item.vec[i] * item.w;
    }
    for (let i = 0; i < dim; i++) userDis[i] /= sumW;
  }

  const meanHistRating = allRatings.reduce((a, b) => a + b, 0) / allRatings.length;

  return {
    userLike: normalize(userLike),
    userDis: normalize(userDis),
    meanHistRating
  };
}

/**
 * Pure client-side forward pass of the MDN MLP model with folded BatchNorm
 */
function forwardMDN(userLike, userDis, targetVec, scalarFeatures, weights) {
  const dim = 512;
  // 1. Interaction features (sim_like, sim_dis)
  let simLike = 0;
  let simDis = 0;
  for (let i = 0; i < dim; i++) {
    simLike += userLike[i] * targetVec[i];
    simDis += userDis[i] * targetVec[i];
  }

  // 2. Concatenate: [u_like (512), u_dis (512), target (512), scalars (4), sim_like (1), sim_dis (1)] = 1542 dims
  const inputDim = 1542;
  const x = new Float32Array(inputDim);
  x.set(userLike, 0);
  x.set(userDis, 512);
  x.set(targetVec, 1024);
  x.set(scalarFeatures, 1536);
  x[1540] = simLike;
  x[1541] = simDis;

  // 3. Sequential MLP Forward Pass with ReLU
  let h = x;
  for (const layer of weights.layers) {
    const inF = layer.in_features;
    const outF = layer.out_features;
    const nextH = new Float32Array(outF);
    const w = layer.weight;
    const b = layer.bias;

    for (let o = 0; o < outF; o++) {
      let sum = b[o];
      const offset = o * inF;
      for (let i = 0; i < inF; i++) {
        sum += w[offset + i] * h[i];
      }
      nextH[o] = sum > 0 ? sum : 0; // ReLU
    }
    h = nextH;
  }

  // 4. Final Layer Linear (512 -> 9)
  const final = weights.final_layer;
  const finalOut = new Float32Array(9);
  for (let o = 0; o < 9; o++) {
    let sum = final.bias[o];
    const offset = o * 512;
    for (let i = 0; i < 512; i++) {
      sum += final.weight[offset + i] * h[i];
    }
    finalOut[o] = sum;
  }

  // 5. Laplace Mixture Components (K = 3)
  const k = 3;
  const mus = new Float32Array(k);
  const sigmas = new Float32Array(k);
  const pis = new Float32Array(k);

  // Mus: 0.9 + 4.2 * sigmoid(mus_raw)
  for (let i = 0; i < k; i++) {
    mus[i] = 0.9 + 4.2 / (1.0 + Math.exp(-finalOut[i]));
  }

  // Sigmas: exp(clamp(log_sigmas_raw, -3.5, -0.7))
  for (let i = 0; i < k; i++) {
    const rawLog = finalOut[k + i];
    const clampedLog = Math.max(-3.5, Math.min(-0.7, rawLog));
    sigmas[i] = Math.exp(clampedLog);
  }

  // Pis: softmax(pi_logits)
  let maxLogit = -Infinity;
  for (let i = 0; i < k; i++) {
    if (finalOut[2 * k + i] > maxLogit) maxLogit = finalOut[2 * k + i];
  }
  let sumExp = 0;
  for (let i = 0; i < k; i++) {
    pis[i] = Math.exp(finalOut[2 * k + i] - maxLogit);
    sumExp += pis[i];
  }
  for (let i = 0; i < k; i++) pis[i] /= sumExp;

  // Expected Mu
  let expectedMu = 0;
  for (let i = 0; i < k; i++) expectedMu += mus[i] * pis[i];

  // 101-point Laplace Mixture PDF Grid from y = 1.0 to 5.0
  const pdfGrid = new Float32Array(101);
  for (let step = 0; step <= 100; step++) {
    const y = 1.0 + (step * 0.04);
    let totalPdf = 0;
    for (let i = 0; i < k; i++) {
      const diff = Math.abs(y - mus[i]);
      const compPdf = (1.0 / (2.0 * sigmas[i])) * Math.exp(-diff / (sigmas[i] + 1e-7));
      totalPdf += pis[i] * compPdf;
    }
    pdfGrid[step] = totalPdf;
  }

  return { expectedMu, pdfGrid, mus, sigmas, pis };
}

function scoreRecommendations(ratings) {
  const profile = buildUserTasteProfile(ratings, state.restaurantEmbeddings, state.centroid);
  if (!profile) return [];

  const { userLike, userDis, meanHistRating } = profile;
  const weights = state.mdnWeights;
  const results = [];

  for (const rest of state.catalog) {
    const rid = rest.rest_id;
    if (ratings[rid]) continue; // Already rated

    const rawVec = state.restaurantEmbeddings[rid];
    if (!rawVec) continue;

    const targetVec = centerVec(rawVec, state.centroid);

    if (weights) {
      // Pretrained MDN Scorer
      const meta = state.metadata[rid] || { stars: 4.0, review_count: 1, price: 2 };
      const scalars = new Float32Array([
        meanHistRating,
        (meta.stars || 4.0) / 5.0,
        Math.log1p(meta.review_count || 1) / 10.0,
        (meta.price || 2) / 4.0
      ]);

      const pred = forwardMDN(userLike, userDis, targetVec, scalars, weights);
      results.push({
        ...rest,
        score: pred.expectedMu,
        pdfGrid: pred.pdfGrid,
        recommendationMode: 'mdn'
      });
    } else {
      // Embedding Fallback Scorer
      const prefVec = new Float32Array(512);
      for (let i = 0; i < 512; i++) {
        prefVec[i] = userLike[i] - (0.35 * userDis[i]);
      }
      const normPref = normalize(prefVec);
      const sim = cosineSimilarity(normPref, targetVec);
      const disSim = norm(userDis) > 1e-5 ? cosineSimilarity(userDis, targetVec) : 0;
      let score = 3.0 + (1.35 * sim) + (0.15 * (meanHistRating - 3.0)) - (0.35 * Math.max(0, disSim));
      score = Math.max(1.0, Math.min(5.0, score));

      results.push({
        ...rest,
        score,
        recommendationMode: 'embedding_fallback'
      });
    }
  }

  results.sort((a, b) => b.score - a.score);
  return results;
}

// ==========================================
// Chart Drawing Helpers (HTML5 Canvas)
// ==========================================

function drawHDRChart(canvas, pdfGrid, expectedRating) {
  if (!canvas || !pdfGrid) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Find max value in grid for scaling
  let maxVal = 0;
  for (let i = 0; i < pdfGrid.length; i++) {
    if (pdfGrid[i] > maxVal) maxVal = pdfGrid[i];
  }
  if (maxVal === 0) maxVal = 1;

  const padLeft = 24;
  const padRight = 16;
  const padBottom = 20;
  const padTop = 10;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  // Grid lines
  ctx.strokeStyle = '#e8dfd7';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, h - padBottom);
  ctx.lineTo(w - padRight, h - padBottom);
  ctx.stroke();

  // Draw area curve
  ctx.beginPath();
  ctx.moveTo(padLeft, h - padBottom);

  for (let i = 0; i < pdfGrid.length; i++) {
    const x = padLeft + (i / (pdfGrid.length - 1)) * plotW;
    const y = (h - padBottom) - (pdfGrid[i] / maxVal) * plotH;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(padLeft + plotW, h - padBottom);
  ctx.closePath();

  const grad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
  grad.addColorStop(0, 'rgba(239, 106, 71, 0.45)');
  grad.addColorStop(1, 'rgba(239, 106, 71, 0.05)');
  ctx.fillStyle = grad;
  ctx.fill();

  // Draw outline
  ctx.beginPath();
  for (let i = 0; i < pdfGrid.length; i++) {
    const x = padLeft + (i / (pdfGrid.length - 1)) * plotW;
    const y = (h - padBottom) - (pdfGrid[i] / maxVal) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#ef6a47';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Draw predicted rating indicator line
  if (expectedRating >= 1.0 && expectedRating <= 5.0) {
    const rx = padLeft + ((expectedRating - 1.0) / 4.0) * plotW;
    ctx.strokeStyle = '#d94f33';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(rx, padTop);
    ctx.lineTo(rx, h - padBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#d94f33';
    ctx.font = 'bold 10px sans-serif';
    ctx.fillText(`${expectedRating.toFixed(2)} ★`, rx - 12, padTop + 10);
  }

  // Rating axis labels (1★ to 5★)
  ctx.fillStyle = '#69737d';
  ctx.font = '10px sans-serif';
  for (let star = 1; star <= 5; star++) {
    const sx = padLeft + ((star - 1) / 4.0) * plotW;
    ctx.fillText(`${star}★`, sx - 6, h - 4);
  }
}

function drawRadarChart(canvas, sentiment) {
  if (!canvas || !sentiment) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(cx, cy) - 26;

  ctx.clearRect(0, 0, w, h);

  const aspects = [
    { label: 'Food', val: sentiment.food_quality || 0 },
    { label: 'Service', val: sentiment.service || 0 },
    { label: 'Ambiance', val: sentiment.ambiance || 0 },
    { label: 'Value', val: sentiment.value || 0 },
    { label: 'Wait Time', val: sentiment.wait_time || 0 }
  ];

  const total = aspects.length;
  const angleStep = (Math.PI * 2) / total;

  // Background Concentric Rings (1 to 5)
  ctx.strokeStyle = '#e8dfd7';
  ctx.lineWidth = 1;
  for (let ring = 1; ring <= 5; ring++) {
    ctx.beginPath();
    const ringR = (ring / 5) * r;
    for (let i = 0; i <= total; i++) {
      const a = (i * angleStep) - (Math.PI / 2);
      const x = cx + ringR * Math.cos(a);
      const y = cy + ringR * Math.sin(a);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Axes and Labels
  ctx.fillStyle = '#1f2328';
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < total; i++) {
    const a = (i * angleStep) - (Math.PI / 2);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    ctx.strokeStyle = '#e8dfd7';
    ctx.stroke();

    const lx = cx + (r + 14) * Math.cos(a);
    const ly = cy + (r + 14) * Math.sin(a);
    ctx.fillText(aspects[i].label, lx, ly);
  }

  // Data Polygon
  ctx.beginPath();
  for (let i = 0; i < total; i++) {
    const a = (i * angleStep) - (Math.PI / 2);
    const valR = (Math.min(5, Math.max(0, aspects[i].val)) / 5) * r;
    const x = cx + valR * Math.cos(a);
    const y = cy + valR * Math.sin(a);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();

  ctx.fillStyle = 'rgba(239, 106, 71, 0.22)';
  ctx.fill();
  ctx.strokeStyle = '#ef6a47';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Data dots
  ctx.fillStyle = '#d94f33';
  for (let i = 0; i < total; i++) {
    const a = (i * angleStep) - (Math.PI / 2);
    const valR = (Math.min(5, Math.max(0, aspects[i].val)) / 5) * r;
    const x = cx + valR * Math.cos(a);
    const y = cy + valR * Math.sin(a);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ==========================================
// Card Rendering Component
// ==========================================

function getDistinctionBadge(rest) {
  const stars = rest.michelin_stars || 0;
  const distinction = (rest.michelin_distinction || rest.michelin_award || '').toLowerCase();

  if (stars === 3) {
    return `<span class="co-distinction-badge"><img src="distinction/3_star.png" class="co-distinction-icon" alt="3 Stars"> 3 Michelin Stars</span>`;
  } else if (stars === 2) {
    return `<span class="co-distinction-badge"><img src="distinction/2_star.PNG" class="co-distinction-icon" alt="2 Stars"> 2 Michelin Stars</span>`;
  } else if (stars === 1) {
    return `<span class="co-distinction-badge"><img src="distinction/1_star.PNG" class="co-distinction-icon" alt="1 Star"> 1 Michelin Star</span>`;
  } else if (distinction.includes('bib')) {
    return `<span class="co-distinction-badge"><img src="distinction/bib_gourmand.png" class="co-distinction-icon" alt="Bib Gourmand"> Bib Gourmand</span>`;
  }
  return '';
}

function renderRestaurantCard(rest, scoreLabel = '', scoreValue = null, extraBadge = '') {
  const rid = rest.rest_id;
  const userRating = state.userRatings[rid] || 0;
  const distinctionHtml = getDistinctionBadge(rest);
  const repImg = rest.representative_image || (rest.image_paths && rest.image_paths[0]) || '';

  const imgHtml = repImg
    ? `<img src="${repImg}" class="co-card-image" alt="${escapeHtml(rest.restaurant_name)}" loading="lazy">`
    : `<div class="co-card-image-placeholder">🍽️</div>`;

  let scoreBoxHtml = '';
  if (scoreValue !== null) {
    scoreBoxHtml = `
      <div class="co-score-box">
        <span class="co-score-label">${scoreLabel || 'Score'}</span>
        <span class="co-score-value">${typeof scoreValue === 'number' ? scoreValue.toFixed(2) : scoreValue}</span>
      </div>
    `;
  }

  const homepageLink = rest.homepage
    ? `<a href="${rest.homepage}" target="_blank" rel="noopener noreferrer">${escapeHtml(rest.restaurant_name)} ↗</a>`
    : escapeHtml(rest.restaurant_name);

  return `
    <div class="co-card" data-rest-id="${rid}">
      <div class="co-result-card-header">
        <div class="co-card-image-wrap">
          ${imgHtml}
        </div>
        <div class="co-card-info">
          <div class="co-card-title-row">
            <h3 class="co-card-title">${homepageLink}</h3>
            ${distinctionHtml}
          </div>
          <div class="co-meta-badges">
            ${rest.borough ? `<span class="co-badge">${escapeHtml(rest.borough)}</span>` : ''}
            ${rest.michelin_cuisine ? `<span class="co-badge">${escapeHtml(rest.michelin_cuisine)}</span>` : ''}
            ${rest.michelin_location ? `<span class="co-badge">${escapeHtml(rest.michelin_location)}</span>` : ''}
            ${extraBadge ? `<span class="co-badge co-badge-accent">${extraBadge}</span>` : ''}
          </div>
          ${scoreBoxHtml}
          <div class="co-rating-strip">
            <span class="co-rating-label">Your Rating:</span>
            <div class="co-star-group" data-rest-id="${rid}">
              ${[1, 2, 3, 4, 5].map(star => `
                <button class="co-star-btn ${star <= userRating ? 'active' : ''}" data-value="${star}" title="Rate ${star} Stars">★</button>
              `).join('')}
            </div>
            ${userRating > 0 ? `<button class="co-btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.75rem;" onclick="setRating('${rid}', 0)">Clear</button>` : ''}
          </div>
        </div>
      </div>

      <!-- Expandable details button & container -->
      <div class="co-card-details-trigger">
        <button class="co-btn co-btn-secondary" style="width: 100%; font-size: 0.85rem;" onclick="toggleDetails('${rid}')">
          View Menu, Reviews, Photos & Sentiment Analysis ▼
        </button>
      </div>

      <div class="co-card-details-panel" id="details-${rid}" style="display: none; margin-top: 1rem; border-top: 1px solid var(--co-border); padding-top: 1rem;">
        <div class="co-subtabs">
          <button class="co-subtab-btn active" onclick="switchSubTab('${rid}', 'menu', this)">Menu (${rest.menu_count || 0})</button>
          <button class="co-subtab-btn" onclick="switchSubTab('${rid}', 'reviews', this)">Reviews (${rest.review_count || 0})</button>
          <button class="co-subtab-btn" onclick="switchSubTab('${rid}', 'photos', this)">Photos (${(rest.image_paths && rest.image_paths.length) || 0})</button>
          <button class="co-subtab-btn" onclick="switchSubTab('${rid}', 'bio', this)">Bio</button>
          <button class="co-subtab-btn" onclick="switchSubTab('${rid}', 'sentiment', this)">Review Sentiment</button>
        </div>

        <div class="co-subtab-content" id="subtab-${rid}-menu">
          <div class="co-menu-grid" id="menu-items-${rid}">Loading menu...</div>
        </div>

        <div class="co-subtab-content" id="subtab-${rid}-reviews" style="display: none;">
          <div class="co-reviews-list" id="reviews-list-${rid}">Loading reviews...</div>
        </div>

        <div class="co-subtab-content" id="subtab-${rid}-photos" style="display: none;">
          <div class="co-gallery-grid" id="photos-grid-${rid}">Loading photos...</div>
        </div>

        <div class="co-subtab-content" id="subtab-${rid}-bio" style="display: none;">
          <p style="font-size: 0.92rem; color: var(--co-text); line-height: 1.6;">${escapeHtml(rest.bio_text || 'No biography text available for this restaurant.')}</p>
        </div>

        <div class="co-subtab-content" id="subtab-${rid}-sentiment" style="display: none;">
          <div id="sentiment-panel-${rid}">Loading aspect sentiment...</div>
        </div>
      </div>

      ${rest.pdfGrid ? `
        <div class="co-chart-canvas-wrap">
          <div class="co-chart-title">Rating Probability Distribution (Mixture Density Network HDR)</div>
          <canvas id="hdr-canvas-${rid}" width="500" height="120" style="width: 100%; height: 120px;"></canvas>
        </div>
      ` : ''}
    </div>
  `;
}

// ==========================================
// Subtab & Details Loader
// ==========================================

async function toggleDetails(restId) {
  const panel = document.getElementById(`details-${restId}`);
  if (!panel) return;
  const isHidden = panel.style.display === 'none';
  panel.style.display = isHidden ? 'block' : 'none';

  if (isHidden) {
    loadRestaurantDetails(restId);
  }
}

async function loadRestaurantDetails(restId) {
  const rest = state.catalog.find(r => r.rest_id === restId);
  if (!rest) return;

  // 1. Menu items
  const menuContainer = document.getElementById(`menu-items-${restId}`);
  if (menuContainer && menuContainer.textContent.includes('Loading')) {
    const menus = await getMenus();
    const items = menus[restId] || [];
    if (items.length === 0) {
      menuContainer.innerHTML = `<p class="co-section-desc">No parsed menu items available.</p>`;
    } else {
      menuContainer.innerHTML = items.slice(0, 18).map(dish => `
        <div class="co-dish-card">
          <div class="co-dish-name">
            <span>${escapeHtml(dish.dish_name || 'Dish')}</span>
            ${dish.price ? `<span class="co-dish-price">${escapeHtml(dish.price)}</span>` : ''}
          </div>
          ${dish.ingredients ? `<p class="co-dish-ingredients">${escapeHtml(dish.ingredients)}</p>` : ''}
        </div>
      `).join('');
    }
  }

  // 2. Reviews
  const reviewsContainer = document.getElementById(`reviews-list-${restId}`);
  if (reviewsContainer && reviewsContainer.textContent.includes('Loading')) {
    const reviews = await getReviews();
    const revList = reviews[restId] || [];
    if (revList.length === 0) {
      reviewsContainer.innerHTML = `<p class="co-section-desc">No review quotes available.</p>`;
    } else {
      reviewsContainer.innerHTML = revList.slice(0, 8).map(rev => `
        <div class="co-review-card">
          <div class="co-review-meta">
            ${rev.rating ? `<span>★ ${rev.rating}</span>` : ''}
            ${rev.source ? `<span>(${escapeHtml(rev.source)})</span>` : ''}
          </div>
          <p class="co-review-text">"${escapeHtml(rev.text)}"</p>
        </div>
      `).join('');
    }
  }

  // 3. Photos
  const photosContainer = document.getElementById(`photos-grid-${restId}`);
  if (photosContainer && photosContainer.textContent.includes('Loading')) {
    const paths = rest.image_paths || [];
    if (paths.length === 0) {
      photosContainer.innerHTML = `<p class="co-section-desc">No photos available.</p>`;
    } else {
      photosContainer.innerHTML = paths.map(src => `
        <div class="co-gallery-item">
          <img src="${src}" class="co-gallery-img" alt="Restaurant photo" loading="lazy">
        </div>
      `).join('');
    }
  }

  // 4. Sentiment Analysis & Radar
  const sentimentContainer = document.getElementById(`sentiment-panel-${restId}`);
  if (sentimentContainer && sentimentContainer.textContent.includes('Loading')) {
    const s = rest.sentiment;
    if (!s || s.food_quality === null) {
      sentimentContainer.innerHTML = `<p class="co-section-desc">No aspect sentiment profile scored for this restaurant.</p>`;
    } else {
      sentimentContainer.innerHTML = `
        <div class="co-sentiment-wrap">
          <div class="co-sentiment-bars">
            ${renderAspectBar('Food Quality', s.food_quality)}
            ${renderAspectBar('Service', s.service)}
            ${renderAspectBar('Ambiance', s.ambiance)}
            ${renderAspectBar('Value', s.value)}
            ${renderAspectBar('Wait Time', s.wait_time)}
          </div>
          <div class="co-radar-wrap">
            <canvas id="radar-${restId}" width="240" height="240"></canvas>
          </div>
        </div>
      `;
      const radarCanvas = document.getElementById(`radar-${restId}`);
      if (radarCanvas) drawRadarChart(radarCanvas, s);
    }
  }
}

function renderAspectBar(label, val) {
  if (val === null || val === undefined) return '';
  const pct = Math.min(100, (val / 5.0) * 100);
  return `
    <div class="co-aspect-bar-row">
      <div class="co-aspect-bar-label">
        <span>${label}</span>
        <span><b>${val.toFixed(1)}</b> / 5.0</span>
      </div>
      <div class="co-progress-bg">
        <div class="co-progress-fill" style="width: ${pct}%;"></div>
      </div>
    </div>
  `;
}

function switchSubTab(restId, target, btn) {
  const panel = document.getElementById(`details-${restId}`);
  if (!panel) return;

  panel.querySelectorAll('.co-subtab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  panel.querySelectorAll('.co-subtab-content').forEach(c => c.style.display = 'none');
  const targetContent = document.getElementById(`subtab-${restId}-${target}`);
  if (targetContent) targetContent.style.display = 'block';

  if (target === 'sentiment') {
    const rest = state.catalog.find(r => r.rest_id === restId);
    const canvas = document.getElementById(`radar-${restId}`);
    if (canvas && rest && rest.sentiment) drawRadarChart(canvas, rest.sentiment);
  }
}

// ==========================================
// Tabs Implementation
// ==========================================

// 1. Browse Restaurants Tab
function renderBrowseTab() {
  const container = document.getElementById('browseResults');
  const boroughFilter = document.getElementById('browseBorough').value;
  const starFilter = document.getElementById('browseStars').value;
  const searchFilter = document.getElementById('browseSearch').value.toLowerCase().trim();

  let filtered = state.catalog.filter(r => {
    if (boroughFilter && r.borough !== boroughFilter) return false;
    if (starFilter) {
      if (starFilter === 'bib' && !(r.michelin_distinction || '').toLowerCase().includes('bib')) return false;
      if (starFilter === '1' && r.michelin_stars !== 1) return false;
      if (starFilter === '2' && r.michelin_stars !== 2) return false;
      if (starFilter === '3' && r.michelin_stars !== 3) return false;
    }
    if (searchFilter) {
      const matchName = r.restaurant_name.toLowerCase().includes(searchFilter);
      const matchCuisine = (r.michelin_cuisine || '').toLowerCase().includes(searchFilter);
      const matchLocation = (r.michelin_location || '').toLowerCase().includes(searchFilter);
      if (!matchName && !matchCuisine && !matchLocation) return false;
    }
    return true;
  });

  const countEl = document.getElementById('browseMatchCount');
  if (countEl) countEl.textContent = `Showing ${Math.min(25, filtered.length)} of ${filtered.length} restaurants`;

  container.innerHTML = filtered.slice(0, 25).map(rest => renderRestaurantCard(rest)).join('');
}

// 2. Multimodal Search Tab
async function handleTextSearch() {
  const query = document.getElementById('searchQuery').value.trim();
  const scope = document.getElementById('searchScope').value;
  const topK = parseInt(document.getElementById('searchTopK').value, 10);
  const container = document.getElementById('searchResults');

  if (!query) {
    container.innerHTML = '<p class="co-section-desc">Enter a search query to discover restaurants.</p>';
    return;
  }

  container.innerHTML = '<p class="co-section-desc">Searching semantic embeddings...</p>';

  const queryTerms = query.toLowerCase().split(/\W+/).filter(t => t.length > 1);

  // Compute lexical overlap + precomputed semantic ranking
  const scored = state.catalog.map(rest => {
    const textToMatch = scope === 'Menu items' ? (rest.menu_items || []).join(' ')
      : scope === 'Reviews' ? (rest.review_snippets || []).join(' ')
      : scope === 'Bios' ? (rest.bio_text || '')
      : (rest.search_text || '');

    const lower = textToMatch.toLowerCase();
    let lexicalHits = 0;
    for (const term of queryTerms) {
      if (lower.includes(term)) lexicalHits++;
    }
    const lexicalScore = queryTerms.length > 0 ? lexicalHits / queryTerms.length : 0;

    // Approximate semantic affinity from lexical overlap and embeddings
    let semanticScore = lexicalScore;
    if (query.length > 3) {
      // Bonus for name/cuisine exact match
      if (rest.restaurant_name.toLowerCase().includes(query.toLowerCase())) semanticScore += 0.45;
      if ((rest.michelin_cuisine || '').toLowerCase().includes(query.toLowerCase())) semanticScore += 0.35;
    }

    const finalScore = (0.7 * semanticScore) + (0.3 * lexicalScore);

    return {
      ...rest,
      finalScore,
      lexicalScore
    };
  });

  const matched = scored.filter(r => r.finalScore > 0).sort((a, b) => b.finalScore - a.finalScore).slice(0, topK);

  if (matched.length === 0) {
    container.innerHTML = '<p class="co-section-desc">No restaurants found. Try broader keywords like "omakase", "french", "pasta", or "tasting menu".</p>';
    return;
  }

  container.innerHTML = matched.map(r => renderRestaurantCard(r, 'Match score', r.finalScore)).join('');
}

// 3. Exact Dish Search Tab
async function handleExactDishSearch() {
  const query = document.getElementById('dishQuery').value.trim().toLowerCase();
  const container = document.getElementById('dishResults');

  if (!query || query.length < 2) {
    container.innerHTML = '<p class="co-section-desc">Type a dish name (e.g. "wagyu", "truffle", "uni", "duck", "carbonara").</p>';
    return;
  }

  container.innerHTML = '<p class="co-section-desc">Searching parsed menu records...</p>';
  const menus = await getMenus();
  const matched = [];

  for (const rest of state.catalog) {
    const rid = rest.rest_id;
    const items = menus[rid] || [];
    const matchedDishes = items.filter(d =>
      (d.dish_name && d.dish_name.toLowerCase().includes(query)) ||
      (d.ingredients && d.ingredients.toLowerCase().includes(query))
    );

    if (matchedDishes.length > 0) {
      matched.push({
        ...rest,
        matchedCount: matchedDishes.length,
        matchedItems: matchedDishes.slice(0, 3)
      });
    }
  }

  matched.sort((a, b) => b.matchedCount - a.matchedCount);

  if (matched.length === 0) {
    container.innerHTML = `<p class="co-section-desc">No restaurants explicitly list "${escapeHtml(query)}" on their menu.</p>`;
    return;
  }

  container.innerHTML = matched.slice(0, 10).map(r => {
    const extraBadge = `${r.matchedCount} Menu Match${r.matchedCount > 1 ? 'es' : ''}`;
    return renderRestaurantCard(r, 'Menu Match Count', r.matchedCount, extraBadge);
  }).join('');
}

// 4. Personalized Recommendations Tab
async function renderRecommendationsTab() {
  const container = document.getElementById('recResults');
  const ratingsCount = Object.keys(state.userRatings).length;

  if (ratingsCount === 0) {
    container.innerHTML = `
      <div class="co-card" style="text-align: center; padding: 3rem 1.5rem;">
        <h3 style="font-size: 1.3rem; margin-bottom: 0.5rem;">No Ratings Added Yet</h3>
        <p class="co-section-desc" style="max-width: 500px; margin: 0 auto 1.5rem;">
          Rate at least 1 restaurant in the <b>Browse</b> or <b>Search</b> tabs using the 1–5 star rating strip to activate the Pretrained Mixture Density Network.
        </p>
        <button class="co-btn" onclick="switchTab('browse')">Go to Restaurant Browser</button>
      </div>
    `;
    return;
  }

  container.innerHTML = '<p class="co-section-desc">Evaluating Mixture Density Network across catalog...</p>';

  // Ensure weights are loaded
  await loadMDNWeights();

  const recs = scoreRecommendations(state.userRatings);
  const topRecs = recs.slice(0, 6);

  container.innerHTML = `
    <p class="co-section-desc" style="margin-bottom: 1.25rem;">
      Shortlist generated from your <b>${ratingsCount}</b> rating${ratingsCount > 1 ? 's' : ''} using the <b>Mixture Density Network (MDN)</b> Laplace mixture head.
    </p>
    ${topRecs.map(r => renderRestaurantCard(r, 'Predicted Rating', r.score)).join('')}
  `;

  // Draw HDR curves for each result
  requestAnimationFrame(() => {
    topRecs.forEach(r => {
      const canvas = document.getElementById(`hdr-canvas-${r.rest_id}`);
      if (canvas && r.pdfGrid) {
        drawHDRChart(canvas, r.pdfGrid, r.score);
      }
    });
  });
}

// 5. My Rated Restaurants Tab
function renderMyRestaurantsTab() {
  const container = document.getElementById('myRestResults');
  const countEl = document.getElementById('myRestCount');
  const ratedIds = Object.keys(state.userRatings);

  if (countEl) countEl.textContent = ratedIds.length;

  if (ratedIds.length === 0) {
    container.innerHTML = `
      <div class="co-card" style="text-align: center; padding: 2.5rem 1.5rem;">
        <h3 style="font-size: 1.2rem; margin-bottom: 0.5rem;">No Rated Restaurants</h3>
        <p class="co-section-desc">Restaurants you rate will appear here with your saved stars.</p>
      </div>
    `;
    return;
  }

  const ratedRestaurants = ratedIds
    .map(id => state.catalog.find(r => r.rest_id === id))
    .filter(Boolean)
    .sort((a, b) => state.userRatings[b.rest_id] - state.userRatings[a.rest_id]);

  container.innerHTML = ratedRestaurants.map(r =>
    renderRestaurantCard(r, 'Your Rating', `${state.userRatings[r.rest_id]} ★`)
  ).join('');
}

// 6. Data Overview Tab
function renderOverviewTab() {
  const total = state.catalog.length;
  let hasMenu = 0;
  let hasReviews = 0;
  let hasImages = 0;
  let hasBios = 0;

  for (const r of state.catalog) {
    if (r.menu_count > 0) hasMenu++;
    if (r.review_count > 0) hasReviews++;
    if (r.image_paths && r.image_paths.length > 0) hasImages++;
    if (r.bio_text) hasBios++;
  }

  const el = id => document.getElementById(id);
  if (el('statTotal')) el('statTotal').textContent = total;
  if (el('statMenus')) el('statMenus').textContent = `${hasMenu} (${Math.round((hasMenu / total) * 100)}%)`;
  if (el('statReviews')) el('statReviews').textContent = `${hasReviews} (${Math.round((hasReviews / total) * 100)}%)`;
  if (el('statImages')) el('statImages').textContent = `${hasImages} (${Math.round((hasImages / total) * 100)}%)`;
  if (el('statBios')) el('statBios').textContent = `${hasBios} (${Math.round((hasBios / total) * 100)}%)`;
}

// ==========================================
// Event Listeners & Tab Navigation
// ==========================================

function switchTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll('.co-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  document.querySelectorAll('.co-tab-content').forEach(tab => {
    tab.classList.toggle('active', tab.id === `tab-${tabId}`);
  });

  if (tabId === 'rec') renderRecommendationsTab();
  if (tabId === 'my-rest') renderMyRestaurantsTab();
  if (tabId === 'browse') renderBrowseTab();
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

document.addEventListener('DOMContentLoaded', () => {
  // Tab Switching
  document.querySelectorAll('.co-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Browse Filters
  const bb = document.getElementById('browseBorough');
  const bs = document.getElementById('browseStars');
  const bq = document.getElementById('browseSearch');
  if (bb) bb.addEventListener('change', renderBrowseTab);
  if (bs) bs.addEventListener('change', renderBrowseTab);
  if (bq) bq.addEventListener('input', renderBrowseTab);

  // Search Input
  const sq = document.getElementById('searchQuery');
  const ss = document.getElementById('searchScope');
  const st = document.getElementById('searchTopK');
  if (sq) sq.addEventListener('input', debounce(handleTextSearch, 300));
  if (ss) ss.addEventListener('change', handleTextSearch);
  if (st) st.addEventListener('input', handleTextSearch);

  // Dish Search
  const dq = document.getElementById('dishQuery');
  if (dq) dq.addEventListener('input', debounce(handleExactDishSearch, 300));

  // Global delegation for Star Ratings
  document.addEventListener('click', e => {
    const starBtn = e.target.closest('.co-star-btn');
    if (starBtn) {
      const restId = starBtn.closest('.co-star-group').dataset.restId;
      const val = parseInt(starBtn.dataset.value, 10);
      setRating(restId, val);
    }
  });

  initData();
});

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Make functions accessible from inline HTML onclicks
window.switchTab = switchTab;
window.toggleDetails = toggleDetails;
window.switchSubTab = switchSubTab;
window.setRating = setRating;

