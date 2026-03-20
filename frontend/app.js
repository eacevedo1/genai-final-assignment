// ── Method display metadata ──────────────────────────────────
const METHOD_META = {
  raw:              { label: 'Raw OLA',              thClass: 'col-raw'  },
  ema_linear_a0_30: { label: 'EMA Linear (α=0.30)', thClass: 'col-emal' },
  ema_log_a0_30:    { label: 'EMA Log (α=0.30)',    thClass: 'col-emag' },
  gru:              { label: 'GRU Predicted',        thClass: 'col-gru'  },
};

// Normalise manifest method key → canonical key
function normaliseKey(k) {
  return k.replace(/\./g, '_'); // ema_linear_a0.30 → ema_linear_a0_30
}

// ── Manifest (inlined to avoid fetch/CORS issues with file:// and GitHub Pages) ──
function loadManifest() {
  return {
    "tracks": [
      {
        "code": "1",
        "original": "examples/1.wav",
        "chunks": {
          "0.25": {
            "raw": "examples/1__chunk0.25s__raw.wav",
            "ema_linear_a0.30": "examples/1__chunk0.25s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/1__chunk0.25s__ema_log_a0.30.wav",
            "gru": "examples/1__chunk0.25s__gru.wav"
          },
          "0.5": {
            "raw": "examples/1__chunk0.50s__raw.wav",
            "ema_linear_a0.30": "examples/1__chunk0.50s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/1__chunk0.50s__ema_log_a0.30.wav",
            "gru": "examples/1__chunk0.50s__gru.wav"
          },
          "1.0": {
            "raw": "examples/1__chunk1.00s__raw.wav",
            "ema_linear_a0.30": "examples/1__chunk1.00s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/1__chunk1.00s__ema_log_a0.30.wav",
            "gru": "examples/1__chunk1.00s__gru.wav"
          }
        }
      },
      {
        "code": "2",
        "original": "examples/2.wav",
        "chunks": {
          "0.25": {
            "raw": "examples/2__chunk0.25s__raw.wav",
            "ema_linear_a0.30": "examples/2__chunk0.25s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/2__chunk0.25s__ema_log_a0.30.wav",
            "gru": "examples/2__chunk0.25s__gru.wav"
          },
          "0.5": {
            "raw": "examples/2__chunk0.50s__raw.wav",
            "ema_linear_a0.30": "examples/2__chunk0.50s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/2__chunk0.50s__ema_log_a0.30.wav",
            "gru": "examples/2__chunk0.50s__gru.wav"
          },
          "1.0": {
            "raw": "examples/2__chunk1.00s__raw.wav",
            "ema_linear_a0.30": "examples/2__chunk1.00s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/2__chunk1.00s__ema_log_a0.30.wav",
            "gru": "examples/2__chunk1.00s__gru.wav"
          }
        }
      },
      {
        "code": "3",
        "original": "examples/3.wav",
        "chunks": {
          "0.25": {
            "raw": "examples/3__chunk0.25s__raw.wav",
            "ema_linear_a0.30": "examples/3__chunk0.25s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/3__chunk0.25s__ema_log_a0.30.wav",
            "gru": "examples/3__chunk0.25s__gru.wav"
          },
          "0.5": {
            "raw": "examples/3__chunk0.50s__raw.wav",
            "ema_linear_a0.30": "examples/3__chunk0.50s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/3__chunk0.50s__ema_log_a0.30.wav",
            "gru": "examples/3__chunk0.50s__gru.wav"
          },
          "1.0": {
            "raw": "examples/3__chunk1.00s__raw.wav",
            "ema_linear_a0.30": "examples/3__chunk1.00s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/3__chunk1.00s__ema_log_a0.30.wav",
            "gru": "examples/3__chunk1.00s__gru.wav"
          }
        }
      },
      {
        "code": "002012",
        "original": "examples/002012.mp3",
        "chunks": {
          "0.25": {
            "raw": "examples/002012__chunk0.25s__raw.wav",
            "ema_linear_a0.30": "examples/002012__chunk0.25s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/002012__chunk0.25s__ema_log_a0.30.wav",
            "gru": "examples/002012__chunk0.25s__gru.wav"
          },
          "0.5": {
            "raw": "examples/002012__chunk0.50s__raw.wav",
            "ema_linear_a0.30": "examples/002012__chunk0.50s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/002012__chunk0.50s__ema_log_a0.30.wav",
            "gru": "examples/002012__chunk0.50s__gru.wav"
          },
          "1.0": {
            "raw": "examples/002012__chunk1.00s__raw.wav",
            "ema_linear_a0.30": "examples/002012__chunk1.00s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/002012__chunk1.00s__ema_log_a0.30.wav",
            "gru": "examples/002012__chunk1.00s__gru.wav"
          }
        }
      },
      {
        "code": "002097",
        "original": "examples/002097.mp3",
        "chunks": {
          "0.25": {
            "raw": "examples/002097__chunk0.25s__raw.wav",
            "ema_linear_a0.30": "examples/002097__chunk0.25s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/002097__chunk0.25s__ema_log_a0.30.wav",
            "gru": "examples/002097__chunk0.25s__gru.wav"
          },
          "0.5": {
            "raw": "examples/002097__chunk0.50s__raw.wav",
            "ema_linear_a0.30": "examples/002097__chunk0.50s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/002097__chunk0.50s__ema_log_a0.30.wav",
            "gru": "examples/002097__chunk0.50s__gru.wav"
          },
          "1.0": {
            "raw": "examples/002097__chunk1.00s__raw.wav",
            "ema_linear_a0.30": "examples/002097__chunk1.00s__ema_linear_a0.30.wav",
            "ema_log_a0.30": "examples/002097__chunk1.00s__ema_log_a0.30.wav",
            "gru": "examples/002097__chunk1.00s__gru.wav"
          }
        }
      }
    ]
  };
}

// ── Render a track ───────────────────────────────────────────
function renderTrack(track) {
  const display = document.getElementById('track-display');
  display.innerHTML = '';

  const chunks = track.chunks;
  const chunkKeys = Object.keys(chunks).sort((a, b) => parseFloat(a) - parseFloat(b));

  // Collect all methods present across all chunk sizes
  const methodKeys = [];
  for (const ck of chunkKeys) {
    for (const mk of Object.keys(chunks[ck])) {
      const nk = normaliseKey(mk);
      if (!methodKeys.includes(nk)) methodKeys.push(nk);
    }
  }
  // Sort by canonical order
  const ORDER = ['raw', 'ema_linear_a0_30', 'ema_log_a0_30', 'gru'];
  methodKeys.sort((a, b) => {
    const ia = ORDER.indexOf(a), ib = ORDER.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });

  const card = document.createElement('div');
  card.className = 'track-card';

  // ── Original row ──
  if (track.original) {
    const origRow = document.createElement('div');
    origRow.className = 'original-row';
    origRow.innerHTML = `
      <span class="badge">Original</span>
      <span class="track-title">Track ${track.code}</span>
      <audio controls preload="none" src="${track.original}"></audio>
    `;
    card.appendChild(origRow);
  }

  // ── Chunk table ──
  const table = document.createElement('table');
  table.className = 'chunk-table';

  // thead
  const thead = document.createElement('thead');
  let thHTML = '<tr><th class="col-chunk">Chunk</th>';
  for (const mk of methodKeys) {
    const meta = METHOD_META[mk] || { label: mk, thClass: '' };
    thHTML += `<th class="${meta.thClass}">${meta.label}</th>`;
  }
  thHTML += '</tr>';
  thead.innerHTML = thHTML;
  table.appendChild(thead);

  // tbody
  const tbody = document.createElement('tbody');
  for (const ck of chunkKeys) {
    const tr = document.createElement('tr');

    // Chunk time cell
    const tdChunk = document.createElement('td');
    tdChunk.className = 'col-chunk';
    tdChunk.setAttribute('data-label', 'Chunk');
    tdChunk.innerHTML = `<span>${parseFloat(ck).toFixed(2)} s</span>`;
    tr.appendChild(tdChunk);

    // Method cells
    const methodMap = {};
    for (const [rawKey, path] of Object.entries(chunks[ck])) {
      methodMap[normaliseKey(rawKey)] = path;
    }

    for (const mk of methodKeys) {
      const meta = METHOD_META[mk] || { label: mk, thClass: '' };
      const td = document.createElement('td');
      td.setAttribute('data-label', meta.label);

      const cell = document.createElement('div');
      cell.className = 'audio-cell';

      if (methodMap[mk]) {
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.preload = 'none';
        audio.src = methodMap[mk];
        cell.appendChild(audio);
      } else {
        const miss = document.createElement('span');
        miss.className = 'missing';
        miss.textContent = '—';
        cell.appendChild(miss);
      }

      td.appendChild(cell);
      tr.appendChild(td);
    }

    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  card.appendChild(table);
  display.appendChild(card);
}

// ── Boot ─────────────────────────────────────────────────────
(() => {
  const display = document.getElementById('track-display');
  const sel = document.getElementById('track-select');

  const manifest = loadManifest();

  const tracks = manifest.tracks;
  if (!tracks || tracks.length === 0) {
    display.innerHTML = '<div class="status-msg">No tracks found in manifest.</div>';
    return;
  }

  // Populate selector
  for (const t of tracks) {
    const opt = document.createElement('option');
    opt.value = t.code;
    opt.textContent = t.code;
    sel.appendChild(opt);
  }

  // Render first track
  renderTrack(tracks[0]);

  // On change
  sel.addEventListener('change', () => {
    const track = tracks.find(t => t.code === sel.value);
    if (track) renderTrack(track);
  });
})();
