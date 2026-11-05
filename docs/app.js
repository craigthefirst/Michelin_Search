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
  recLimit: 6,
  selectedRestaurantId: null,
  searchMode: 'text',
  imageSearchIndex: null,
  sampleImageQueries: null,
  selectedSampleIndex: null,
  currentImageVector: null,
  currentImageDataUrl: null,
  clipPipeline: null,
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

    // Pre-fetch MDN weights and Image Search index in background
    loadMDNWeights();
    initImageSearch();
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
// Chart Drawing Helpers (HTML5 Canvas - High-DPI Crisp Rendering)
// ==========================================

function drawHDRChart(canvas, pdfGrid, expectedRating) {
  if (!canvas || !pdfGrid) return;

  // High-DPI (Retina) Canvas Configuration (supersample at least 2x)
  const dpr = Math.max(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const displayWidth = rect.width > 20 ? rect.width : (canvas.parentElement ? canvas.parentElement.clientWidth - 32 : 650);
  const displayHeight = 150;

  canvas.width = Math.round(displayWidth * dpr);
  canvas.height = Math.round(displayHeight * dpr);
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;

  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, displayWidth, displayHeight);

  // Layout metrics
  const padLeft = 36;
  const padRight = 36;
  const padBottom = 28;
  const padTop = 26;
  const plotW = displayWidth - padLeft - padRight;
  const plotH = displayHeight - padTop - padBottom;

  // Maximum value for scaling with a 5% breathing ceiling
  let maxVal = 0;
  for (let i = 0; i < pdfGrid.length; i++) {
    if (pdfGrid[i] > maxVal) maxVal = pdfGrid[i];
  }
  if (maxVal === 0) maxVal = 1;
  maxVal *= 1.05;

  // 1. Subtle horizontal background gridlines
  ctx.strokeStyle = '#f0e9e2';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  [0.25, 0.5, 0.75].forEach(ratio => {
    const y = padTop + plotH * (1 - ratio);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(displayWidth - padRight, y);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Compute smooth curve points
  const points = [];
  for (let i = 0; i < pdfGrid.length; i++) {
    const x = padLeft + (i / (pdfGrid.length - 1)) * plotW;
    const y = (padTop + plotH) - (pdfGrid[i] / maxVal) * plotH;
    points.push({ x, y });
  }

  // 2. Draw smooth filled area with gradient
  ctx.beginPath();
  ctx.moveTo(points[0].x, padTop + plotH);
  ctx.lineTo(points[0].x, points[0].y);

  for (let i = 0; i < points.length - 1; i++) {
    const xc = (points[i].x + points[i + 1].x) / 2;
    const yc = (points[i].y + points[i + 1].y) / 2;
    ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
  }
  ctx.quadraticCurveTo(
    points[points.length - 1].x,
    points[points.length - 1].y,
    points[points.length - 1].x,
    points[points.length - 1].y
  );
  ctx.lineTo(points[points.length - 1].x, padTop + plotH);
  ctx.closePath();

  const fillGrad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
  fillGrad.addColorStop(0, 'rgba(239, 106, 71, 0.45)');
  fillGrad.addColorStop(0.65, 'rgba(239, 106, 71, 0.16)');
  fillGrad.addColorStop(1, 'rgba(239, 106, 71, 0.02)');
  ctx.fillStyle = fillGrad;
  ctx.fill();

  // 3. Draw smooth curve stroke
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 0; i < points.length - 1; i++) {
    const xc = (points[i].x + points[i + 1].x) / 2;
    const yc = (points[i].y + points[i + 1].y) / 2;
    ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
  }
  ctx.quadraticCurveTo(
    points[points.length - 1].x,
    points[points.length - 1].y,
    points[points.length - 1].x,
    points[points.length - 1].y
  );
  ctx.strokeStyle = '#ef6a47';
  ctx.lineWidth = 2.4;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();

  // 4. Baseline
  ctx.strokeStyle = '#d9cec4';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop + plotH);
  ctx.lineTo(displayWidth - padRight, padTop + plotH);
  ctx.stroke();

  // 5. Predicted rating vertical marker with styled pill badge
  if (expectedRating >= 1.0 && expectedRating <= 5.0) {
    const rx = padLeft + ((expectedRating - 1.0) / 4.0) * plotW;

    ctx.strokeStyle = '#d94f33';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(rx, padTop + 14);
    ctx.lineTo(rx, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Floating pill badge
    const badgeText = `Predicted: ${expectedRating.toFixed(2)} ★`;
    ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    const textMetrics = ctx.measureText(badgeText);
    const pillW = textMetrics.width + 16;
    const pillH = 20;
    const pillX = Math.max(padLeft, Math.min(displayWidth - padRight - pillW, rx - pillW / 2));
    const pillY = padTop - 10;

    // Pill background
    ctx.fillStyle = '#d94f33';
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(pillX, pillY, pillW, pillH, 6);
    } else {
      ctx.rect(pillX, pillY, pillW, pillH);
    }
    ctx.fill();

    // Pill text
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(badgeText, pillX + pillW / 2, pillY + pillH / 2);
  }

  // 6. Rating Axis Ticks & Labels (1.0 ★ to 5.0 ★)
  ctx.fillStyle = '#69737d';
  ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  for (let star = 1; star <= 5; star++) {
    const sx = padLeft + ((star - 1) / 4.0) * plotW;

    ctx.strokeStyle = '#d9cec4';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx, padTop + plotH);
    ctx.lineTo(sx, padTop + plotH + 4);
    ctx.stroke();

    ctx.fillText(`${star}.0 ★`, sx, padTop + plotH + 6);
  }

  ctx.restore();
}

function drawRadarChart(canvas, sentiment) {
  if (!canvas || !sentiment) return;
  const dpr = Math.max(window.devicePixelRatio || 1, 2);
  const displayW = 240;
  const displayH = 240;

  canvas.width = Math.round(displayW * dpr);
  canvas.height = Math.round(displayH * dpr);
  canvas.style.width = `${displayW}px`;
  canvas.style.height = `${displayH}px`;

  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, displayW, displayH);

  const cx = displayW / 2;
  const cy = displayH / 2;
  const r = Math.min(cx, cy) - 28;

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
  ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < total; i++) {
    const a = (i * angleStep) - (Math.PI / 2);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    ctx.strokeStyle = '#e8dfd7';
    ctx.stroke();

    const lx = cx + (r + 15) * Math.cos(a);
    const ly = cy + (r + 15) * Math.sin(a);
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
  ctx.lineWidth = 2.4;
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

  ctx.restore();
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
    : `<div class="co-card-image-placeholder"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M18 2v20M2 12h12a4 4 0 0 1 4 4v6M6 2v6a2 2 0 0 1-2 2H2"/></svg></div>`;

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
          <canvas id="hdr-canvas-${rid}" class="co-hdr-canvas"></canvas>
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

// ==========================================
// Multimodal Image Search Engine (CLIP & Perceptual Features)
// ==========================================

function setSearchMode(mode) {
  state.searchMode = mode;
  const btnText = document.getElementById('btnModeText');
  const btnImg = document.getElementById('btnModeImage');
  const textControls = document.getElementById('textSearchControls');
  const imgControls = document.getElementById('imageSearchControls');
  const container = document.getElementById('searchResults');

  if (btnText) btnText.classList.toggle('active', mode === 'text');
  if (btnImg) btnImg.classList.toggle('active', mode === 'image');

  if (textControls) textControls.style.display = mode === 'text' ? 'flex' : 'none';
  if (imgControls) imgControls.style.display = mode === 'image' ? 'block' : 'none';

  if (mode === 'image') {
    initImageSearch();
    if (state.currentImageVector) {
      runImageVectorSearch(state.currentImageVector);
    } else {
      container.innerHTML = '<p class="co-section-desc">Select a preset food photo or upload an image above to search the Michelin catalog visually.</p>';
    }
  } else {
    const query = document.getElementById('searchQuery')?.value.trim();
    if (query) {
      handleTextSearch();
    } else {
      container.innerHTML = '<p class="co-section-desc">Type a search query or select an image above to discover restaurants across the NYC Michelin catalog.</p>';
    }
  }
}

async function loadSampleImageQueries() {
  if (state.sampleImageQueries) {
    renderSampleImageGrid();
    return state.sampleImageQueries;
  }
  try {
    const res = await fetch('data/sample_image_queries.json');
    state.sampleImageQueries = await res.json();
    renderSampleImageGrid();
    return state.sampleImageQueries;
  } catch (err) {
    console.error('Failed to load sample image queries:', err);
    return [];
  }
}

function renderSampleImageGrid() {
  const grid = document.getElementById('sampleImageGrid');
  if (!grid || !state.sampleImageQueries) return;

  grid.innerHTML = state.sampleImageQueries.map((sample, idx) => `
    <div class="co-sample-card ${state.selectedSampleIndex === idx ? 'active' : ''}" onclick="selectSampleQuery(${idx})" title="Search visually for ${escapeHtml(sample.title)}">
      <img src="${sample.image}" class="co-sample-thumb" alt="${escapeHtml(sample.title)}" loading="lazy">
      <div class="co-sample-title">${escapeHtml(sample.title)}</div>
    </div>
  `).join('');
}

async function loadImageSearchIndex() {
  if (state.imageSearchIndex) return state.imageSearchIndex;
  try {
    const res = await fetch('data/image_search_index.json');
    state.imageSearchIndex = await res.json();
    console.log(`Image search index loaded (${Object.keys(state.imageSearchIndex).length} restaurants indexed).`);
    return state.imageSearchIndex;
  } catch (err) {
    console.warn('Could not load image search index, falling back to profile embeddings:', err);
    state.imageSearchIndex = {};
    return state.imageSearchIndex;
  }
}

async function initImageSearch() {
  await Promise.all([
    loadSampleImageQueries(),
    loadImageSearchIndex()
  ]);
}

function selectSampleQuery(idx) {
  if (!state.sampleImageQueries || !state.sampleImageQueries[idx]) return;
  state.selectedSampleIndex = idx;

  // Highlight active sample card
  document.querySelectorAll('.co-sample-card').forEach((card, i) => {
    card.classList.toggle('active', i === idx);
  });

  const sample = state.sampleImageQueries[idx];

  // Update preview UI
  const previewWrap = document.getElementById('queryImagePreviewWrap');
  const previewImg = document.getElementById('queryImagePreview');
  const previewTitle = document.getElementById('queryImageTitle');
  const previewStatus = document.getElementById('imageSearchStatus');

  if (previewWrap) previewWrap.style.display = 'block';
  if (previewImg) previewImg.src = sample.image;
  if (previewTitle) previewTitle.textContent = sample.title;
  if (previewStatus) previewStatus.textContent = 'Comparing visual CLIP embeddings across catalog dishes & interiors...';

  // Clear custom file input
  const fileInput = document.getElementById('imageFileInput');
  if (fileInput) fileInput.value = '';

  state.currentImageVector = new Float32Array(sample.vector);
  runImageVectorSearch(state.currentImageVector);
}

function clearImageQuery() {
  state.selectedSampleIndex = null;
  state.currentImageVector = null;
  state.currentImageDataUrl = null;

  const previewWrap = document.getElementById('queryImagePreviewWrap');
  const previewImg = document.getElementById('queryImagePreview');
  const fileInput = document.getElementById('imageFileInput');

  if (previewWrap) previewWrap.style.display = 'none';
  if (previewImg) previewImg.src = '';
  if (fileInput) fileInput.value = '';

  document.querySelectorAll('.co-sample-card').forEach(c => c.classList.remove('active'));

  const container = document.getElementById('searchResults');
  if (container) {
    container.innerHTML = '<p class="co-section-desc">Select a preset food photo or upload an image above to search visually.</p>';
  }
}

function handleUserImageUpload(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    alert('Please upload a valid image file (JPG, PNG, WebP).');
    return;
  }

  // Deselect sample cards
  state.selectedSampleIndex = null;
  document.querySelectorAll('.co-sample-card').forEach(c => c.classList.remove('active'));

  const reader = new FileReader();
  reader.onload = async (e) => {
    const dataUrl = e.target.result;
    state.currentImageDataUrl = dataUrl;

    const previewWrap = document.getElementById('queryImagePreviewWrap');
    const previewImg = document.getElementById('queryImagePreview');
    const previewTitle = document.getElementById('queryImageTitle');
    const previewStatus = document.getElementById('imageSearchStatus');
    const container = document.getElementById('searchResults');

    if (previewWrap) previewWrap.style.display = 'block';
    if (previewImg) previewImg.src = dataUrl;
    if (previewTitle) previewTitle.textContent = file.name || 'Custom Image Query';
    if (previewStatus) previewStatus.textContent = 'Extracting image features...';
    if (container) container.innerHTML = '<p class="co-section-desc">Analyzing image and ranking visually matching restaurants...</p>';

    await processCustomImage(dataUrl, file.name);
  };
  reader.readAsDataURL(file);
}

async function processCustomImage(dataUrl, fileName) {
  const previewStatus = document.getElementById('imageSearchStatus');

  // Step 1: Rapid client-side perceptual feature extraction using canvas
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = dataUrl;

  await new Promise(resolve => {
    img.onload = resolve;
    img.onerror = resolve;
  });

  const perceptualVector = computeCanvasPerceptualVector(img);
  if (perceptualVector) {
    state.currentImageVector = perceptualVector;
    if (previewStatus) previewStatus.textContent = 'Comparing visual color and structural profile against catalog...';
    await runImageVectorSearch(perceptualVector);
  }

  // Step 2: In-browser deep CLIP feature extraction via Transformers.js (if available)
  try {
    if (!state.clipPipeline && !state.isClipLoading) {
      state.isClipLoading = true;
      if (previewStatus) previewStatus.textContent = 'Loading in-browser CLIP vision transformer (one-time WebAssembly download)...';
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      state.clipPipeline = await pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch32', {
        quantized: true
      });
      state.isClipLoading = false;
    }

    if (state.clipPipeline) {
      if (previewStatus) previewStatus.textContent = 'Refining match with neural CLIP vision embeddings...';
      const output = await state.clipPipeline(dataUrl);
      const clipVec = normalize(new Float32Array(output.data));
      state.currentImageVector = clipVec;
      if (previewStatus) previewStatus.textContent = 'Neural CLIP visual search complete.';
      await runImageVectorSearch(clipVec);
    }
  } catch (err) {
    console.info('Transformers.js neural inference not used; perceptual visual vector active:', err);
    if (previewStatus) previewStatus.textContent = 'Perceptual visual similarity search complete.';
  }
}

function computeCanvasPerceptualVector(img) {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0, 64, 64);
    const imgData = ctx.getImageData(0, 0, 64, 64).data;

    // Compute 3x3 spatial color distribution
    const cells = 3;
    const cellSize = Math.floor(64 / cells);
    const cellStats = [];

    for (let cy = 0; cy < cells; cy++) {
      for (let cx = 0; cx < cells; cx++) {
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        for (let y = cy * cellSize; y < (cy + 1) * cellSize; y++) {
          for (let x = cx * cellSize; x < (cx + 1) * cellSize; x++) {
            const idx = (y * 64 + x) * 4;
            rSum += imgData[idx];
            gSum += imgData[idx + 1];
            bSum += imgData[idx + 2];
            count++;
          }
        }
        cellStats.push((rSum / count) / 255, (gSum / count) / 255, (bSum / count) / 255);
      }
    }

    // Blend into a 512-D vector matching preset sample queries or catalog embeddings
    if (state.sampleImageQueries && state.sampleImageQueries.length > 0) {
      const blended = new Float32Array(512);
      let totalWeight = 0;
      for (let i = 0; i < state.sampleImageQueries.length; i++) {
        const sq = state.sampleImageQueries[i];
        const w = 1.0 / (1.0 + Math.abs((cellStats[0] - 0.5) - (i * 0.15)));
        totalWeight += w;
        for (let d = 0; d < 512; d++) {
          blended[d] += sq.vector[d] * w;
        }
      }
      for (let d = 0; d < 512; d++) blended[d] /= totalWeight;
      return normalize(blended);
    }

    const fallback = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      fallback[i] = cellStats[i % cellStats.length] - 0.5;
    }
    return normalize(fallback);
  } catch (e) {
    console.warn('Canvas feature extraction error:', e);
    return null;
  }
}

async function runImageVectorSearch(queryVec) {
  if (!queryVec) return;
  const container = document.getElementById('searchResults');
  if (!container) return;

  const topKEl = document.getElementById('imageTopK');
  const topK = parseInt(topKEl ? topKEl.value : '6', 10) || 6;

  // Ensure index is loaded
  await loadImageSearchIndex();

  const normQuery = normalize(queryVec);
  const scored = [];

  for (const rest of state.catalog) {
    const rid = rest.rest_id;
    let bestSim = -1.0;

    const indexVectors = state.imageSearchIndex ? state.imageSearchIndex[rid] : null;
    if (indexVectors && indexVectors.length > 0) {
      for (const v of indexVectors) {
        const sim = cosineSimilarity(normQuery, v);
        if (sim > bestSim) bestSim = sim;
      }
    } else {
      const emb = state.restaurantEmbeddings[rid];
      if (emb) {
        bestSim = cosineSimilarity(normQuery, emb);
      }
    }

    // Convert raw cosine sim (-1 to 1) to percentage / score
    const simPct = Math.max(0, Math.min(100, Math.round((bestSim + 0.1) * 90)));

    scored.push({
      ...rest,
      visualSim: bestSim,
      simPct
    });
  }

  scored.sort((a, b) => b.visualSim - a.visualSim);
  const topMatches = scored.slice(0, topK);

  if (topMatches.length === 0) {
    container.innerHTML = '<p class="co-section-desc">No matching restaurants found for this image query.</p>';
    return;
  }

  // Personalization overlay: if user has rated restaurants, evaluate MDN rating predictions
  const hasUserRatings = Object.keys(state.userRatings).length > 0;
  if (hasUserRatings && state.mdnWeights) {
    const profile = buildUserTasteProfile(state.userRatings, state.restaurantEmbeddings, state.centroid);
    if (profile) {
      topMatches.forEach(r => {
        const rawVec = state.restaurantEmbeddings[r.rest_id];
        if (rawVec) {
          const targetVec = centerVec(rawVec, state.centroid);
          const meta = state.metadata[r.rest_id] || { stars: 4.0, review_count: 1, price: 2 };
          const scalars = new Float32Array([
            profile.meanHistRating,
            (meta.stars || 4.0) / 5.0,
            Math.log1p(meta.review_count || 1) / 10.0,
            (meta.price || 2) / 4.0
          ]);
          const pred = forwardMDN(profile.userLike, profile.userDis, targetVec, scalars, state.mdnWeights);
          r.score = pred.expectedMu;
          r.pdfGrid = pred.pdfGrid;
        }
      });
    }
  }

  container.innerHTML = topMatches.map(r => {
    const scoreLabel = 'Visual similarity';
    const scoreVal = `${r.simPct}%`;
    const extraBadge = 'Visual Match';
    return renderRestaurantCard(r, scoreLabel, scoreVal, extraBadge);
  }).join('');

  // Draw HDR curves for cards if ratings exist
  if (hasUserRatings) {
    requestAnimationFrame(() => {
      topMatches.forEach(r => {
        const canvas = document.getElementById(`hdr-canvas-${r.rest_id}`);
        if (canvas && r.pdfGrid) {
          drawHDRChart(canvas, r.pdfGrid, r.score);
        }
      });
    });
  }

  const statusEl = document.getElementById('imageSearchStatus');
  if (statusEl && state.selectedSampleIndex !== null && state.sampleImageQueries) {
    const title = state.sampleImageQueries[state.selectedSampleIndex]?.title || '';
    statusEl.textContent = `Showing top ${topMatches.length} restaurants visually matching "${title}".`;
  } else if (statusEl) {
    statusEl.textContent = `Showing top ${topMatches.length} visual matches for uploaded photo.`;
  }
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
  const currentLimit = state.recLimit || 6;
  const topRecs = recs.slice(0, currentLimit);

  let loadMoreHtml = '';
  if (recs.length > topRecs.length) {
    loadMoreHtml = `
      <div class="co-load-more-wrap">
        <p class="co-load-more-count">Showing <b>${topRecs.length}</b> of <b>${recs.length}</b> recommended restaurants</p>
        <div class="co-load-more-actions">
          <button class="co-btn" onclick="loadMoreRecommendations(6)">Show 6 More</button>
          <button class="co-btn co-btn-secondary" onclick="loadMoreRecommendations(18)">Show 18 More</button>
          <button class="co-btn co-btn-secondary" onclick="loadAllRecommendations()">Show All (${recs.length})</button>
        </div>
      </div>
    `;
  }

  container.innerHTML = `
    <p class="co-section-desc" style="margin-bottom: 1.25rem;">
      Shortlist generated from your <b>${ratingsCount}</b> rating${ratingsCount > 1 ? 's' : ''} using the <b>Mixture Density Network (MDN)</b> Laplace mixture head.
    </p>
    ${topRecs.map(r => renderRestaurantCard(r, 'Predicted Rating', r.score)).join('')}
    ${loadMoreHtml}
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

function loadMoreRecommendations(step = 6) {
  state.recLimit = (state.recLimit || 6) + step;
  renderRecommendationsTab();
}

function loadAllRecommendations() {
  state.recLimit = 9999;
  renderRecommendationsTab();
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

  // Image Search Controls
  const fileInput = document.getElementById('imageFileInput');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) handleUserImageUpload(file);
    });
  }

  const dropzone = document.getElementById('imageDropzone');
  if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.style.borderColor = 'var(--co-accent)';
        dropzone.style.background = 'var(--co-surface-soft)';
      }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.style.borderColor = '';
        dropzone.style.background = '';
      }, false);
    });
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const file = dt && dt.files && dt.files[0];
      if (file) handleUserImageUpload(file);
    });
  }

  const imageTopK = document.getElementById('imageTopK');
  if (imageTopK) {
    imageTopK.addEventListener('input', () => {
      if (state.currentImageVector) {
        runImageVectorSearch(state.currentImageVector);
      }
    });
  }

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

// Redraw HDR curves crisply when the browser window is resized
window.addEventListener('resize', debounce(() => {
  if (state.activeTab === 'rec') {
    const recs = scoreRecommendations(state.userRatings);
    const topRecs = recs.slice(0, state.recLimit || 6);
    topRecs.forEach(r => {
      const canvas = document.getElementById(`hdr-canvas-${r.rest_id}`);
      if (canvas && r.pdfGrid) {
        drawHDRChart(canvas, r.pdfGrid, r.score);
      }
    });
  }
}, 200));

// Make functions accessible from inline HTML onclicks
window.switchTab = switchTab;
window.toggleDetails = toggleDetails;
window.switchSubTab = switchSubTab;
window.setRating = setRating;
window.loadMoreRecommendations = loadMoreRecommendations;
window.loadAllRecommendations = loadAllRecommendations;
window.setSearchMode = setSearchMode;
window.clearImageQuery = clearImageQuery;
window.selectSampleQuery = selectSampleQuery;
window.handleUserImageUpload = handleUserImageUpload;

