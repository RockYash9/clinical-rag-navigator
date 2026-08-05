// Clinical RAG Navigator — chat UI client.
// Talks to this same FastAPI app's /health and /query endpoints. No build
// step, no framework — kept intentionally simple to match the project's
// free/local-first constraints.

const chatEl = document.getElementById("chat");
const emptyStateEl = document.getElementById("empty-state");
const formEl = document.getElementById("composer-form");
const inputEl = document.getElementById("question-input");
const sendButtonEl = document.getElementById("send-button");
const statusPillEl = document.getElementById("status-pill");

function setStatus(state, label) {
  statusPillEl.textContent = label;
  statusPillEl.className = `status-pill status-pill--${state}`;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.pipeline_loaded) {
      setStatus("ready", "Ready");
    } else {
      setStatus(
        "error",
        "Index not built — run scripts/ingest.py + build_index.py"
      );
    }
  } catch (err) {
    setStatus("error", "Server unreachable");
  }
}

function confidenceTier(score) {
  if (score >= 0.7) return { key: "good", color: "var(--good)" };
  if (score >= 0.4) return { key: "mid", color: "var(--mid)" };
  return { key: "low", color: "var(--low)" };
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row message-row--user";
  const bubble = document.createElement("div");
  bubble.className = "bubble-user";
  bubble.textContent = text;
  row.appendChild(bubble);
  chatEl.appendChild(row);
  scrollToBottom();
}

function addThinkingCard() {
  const row = document.createElement("div");
  row.className = "message-row message-row--assistant";
  row.innerHTML = `
    <div class="response-card">
      <div class="response-answer is-thinking">
        Retrieving and generating
        <span class="pulse-dot"></span><span class="pulse-dot"></span><span class="pulse-dot"></span>
      </div>
    </div>
  `;
  chatEl.appendChild(row);
  scrollToBottom();
  return row;
}

function renderCitation(citation) {
  const year = citation.year ? `, ${citation.year}` : "";
  return `
    <div class="citation-item">
      <div class="citation-title">${escapeHtml(citation.source_title)}</div>
      <div class="citation-meta">
        <a href="${escapeHtml(citation.url)}" target="_blank" rel="noopener noreferrer">
          ${escapeHtml(citation.organization)}${year}
        </a>
      </div>
      <div class="citation-excerpt">${escapeHtml(citation.excerpt)}</div>
    </div>
  `;
}

function fillCardWithResponse(row, data) {
  const tier = confidenceTier(data.confidence);
  const pct = Math.round(data.confidence * 100);

  const citationsHtml =
    data.citations && data.citations.length
      ? `
        <details class="citations">
          <summary>${data.citations.length} source${data.citations.length === 1 ? "" : "s"}</summary>
          <div class="citation-list">
            ${data.citations.map(renderCitation).join("")}
          </div>
        </details>
      `
      : "";

  row.innerHTML = `
    <div class="response-card">
      <div class="response-answer">${escapeHtml(data.answer)}</div>
      <div class="confidence-row">
        <span class="confidence-label">Confidence</span>
        <div class="confidence-track">
          <div class="confidence-fill" style="width: ${pct}%; background: ${tier.color};"></div>
        </div>
        <span class="confidence-value" style="color: ${tier.color};">${pct}%</span>
      </div>
      ${citationsHtml}
    </div>
  `;
  scrollToBottom();
}

function fillCardWithError(row, message) {
  row.innerHTML = `
    <div class="response-card">
      <div class="response-error">${escapeHtml(message)}</div>
    </div>
  `;
  scrollToBottom();
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function submitQuestion(question) {
  if (emptyStateEl) emptyStateEl.remove();

  addUserMessage(question);
  const row = addThinkingCard();

  sendButtonEl.disabled = true;
  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    fillCardWithResponse(row, data);
  } catch (err) {
    fillCardWithError(row, err.message || "Something went wrong.");
  } finally {
    sendButtonEl.disabled = false;
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  submitQuestion(question);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 140)}px`;
});

document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    submitQuestion(chip.textContent.trim());
  });
});

checkHealth();
