const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const searchModeInput = document.getElementById("search-mode");
const outputModeInput = document.getElementById("output-mode");
const submitButton = document.getElementById("submit-button");
const synthesizeButton = document.getElementById("synthesize-button");
const resetButton = document.getElementById("reset-button");
const resultsContainer = document.getElementById("results");
const synthesisCard = document.getElementById("synthesis-card");
const synthesisAnswer = document.getElementById("synthesis-answer");
const resultsTitle = document.getElementById("results-title");
const resultFlags = document.getElementById("result-flags");
const feedback = document.getElementById("feedback");
const template = document.getElementById("result-template");
const apiStatus = document.getElementById("api-status");
const collectionName = document.getElementById("collection-name");
const activeFilters = document.getElementById("active-filters");
const filtersStatus = document.getElementById("filters-status");

const filterFields = ["language", "priority", "queue", "type"];
const filterLabels = {
  language: "Langue",
  priority: "Priorité",
  queue: "Queue",
  type: "Type",
};

const outputLabels = {
  raw: "Brut",
  translated: "Traduit",
  reranked: "Reranké",
  "reranked-translated": "Rerank + traduit",
  "translated-reranked": "Traduit + rerank",
};

let currentRetrievedPayload = null;
let currentRetrievedQuery = "";
let synthesisInProgress = false;

function setFeedback(message, tone = "") {
  feedback.textContent = message;
  feedback.className = `feedback ${tone}`.trim();
}

function normalizeDisplayText(value) {
  return String(value || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t")
    .trim();
}

function truncate(text, maxLength = 360) {
  const normalized = normalizeDisplayText(text);
  if (!normalized) {
    return "";
  }
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength).trim()}...` : normalized;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>'
    );
}

function renderMarkdownLite(element, text) {
  const normalized = normalizeDisplayText(text);
  if (!normalized) {
    element.textContent = "";
    return;
  }

  const blocks = normalized.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean);
  element.innerHTML = blocks
    .map((block) => {
      const lines = block.split("\n");
      const bulletLines = lines.filter((line) => /^[-*]\s+/.test(line.trim()));

      if (bulletLines.length === lines.length) {
        const items = lines
          .map((line) => `<li>${renderInlineMarkdown(line.trim().replace(/^[-*]\s+/, ""))}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
      }

      return `<p>${renderInlineMarkdown(block).replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
}

function formatScore(value) {
  if (typeof value !== "number") {
    return "Score indisponible";
  }
  return `Score ${value.toFixed(3)}`;
}

function buildFilters() {
  return filterFields.reduce((acc, field) => {
    const value = document.getElementById(field).value.trim();
    if (value) {
      acc[field] = value;
    }
    return acc;
  }, {});
}

function setSelectState(selectId, label, disabled = true) {
  const select = document.getElementById(selectId);
  select.innerHTML = "";
  const option = document.createElement("option");
  option.value = "";
  option.textContent = label;
  select.appendChild(option);
  select.disabled = disabled;
}

function populateSelect(selectId, values) {
  const select = document.getElementById(selectId);
  const defaultLabel = selectId === "type" ? "Tous" : "Toutes";
  select.innerHTML = "";

  const emptyOption = document.createElement("option");
  emptyOption.value = "";
  emptyOption.textContent = defaultLabel;
  select.appendChild(emptyOption);

  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }

  select.disabled = false;
}

function updateActiveFilters() {
  const filters = buildFilters();
  activeFilters.innerHTML = "";

  for (const [key, value] of Object.entries(filters)) {
    const chip = document.createElement("span");
    chip.className = "active-filter-chip";
    chip.textContent = `${filterLabels[key] || key}: ${value}`;
    activeFilters.appendChild(chip);
  }
}

function normalizeResults(data) {
  if (!data || typeof data !== "object") {
    return {
      flags: "Réponse vide",
      items: [],
    };
  }

  const translated = data.translated || "not asked";
  const reranked = data.reranked || "not asked";
  const synthesized = data.synthesized || "not asked";
  const payload = data.payload || {};

  return {
    flags: `Traduit: ${translated} | Reranké: ${reranked} | Synthétisé: ${synthesized}`,
    answer: data.answer || "",
    payload,
    items: Object.entries(payload).map(([id, item]) => ({ id, ...item })),
  };
}

function renderEmptyState(message) {
  resultsContainer.innerHTML = `<div class="empty-state">${message}</div>`;
}

function clearSynthesis() {
  synthesisAnswer.textContent = "";
  synthesisCard.classList.add("hidden");
  synthesisCard.classList.remove("error");
  synthesisCard.classList.remove("loading");
}

function renderResults(items) {
  resultsContainer.innerHTML = "";

  if (!items.length) {
    renderEmptyState("Aucun ticket ne correspond à cette requête avec les filtres actuels.");
    return;
  }

  const fragment = document.createDocumentFragment();

  for (const item of items) {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".result-score").textContent = formatScore(item.score);
    node.querySelector(".result-id").textContent = item.id;
    node.querySelector(".result-subject").textContent = normalizeDisplayText(item.subject) || "Sans sujet";
    renderMarkdownLite(node.querySelector(".result-body"), truncate(item.body || "Aucun contenu de ticket."));
    renderMarkdownLite(node.querySelector(".result-answer"), truncate(item.answer || "Aucune réponse enregistrée."));

    const tagRow = node.querySelector(".tag-row");
    const tags = [
      ["Langue", item.language],
      ["Priorité", item.priority],
      ["Queue", item.queue],
      ["Type", item.type],
    ].filter(([, value]) => value);

    for (const [label, value] of tags) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = `${label}: ${value}`;
      tagRow.appendChild(tag);
    }

    fragment.appendChild(node);
  }

  resultsContainer.appendChild(fragment);
}

function renderSynthesis(answer, tone = "") {
  const normalized = normalizeDisplayText(answer);
  if (!normalized) {
    clearSynthesis();
    return;
  }

  synthesisCard.classList.remove("hidden");
  synthesisCard.classList.toggle("error", tone === "error");
  synthesisCard.classList.toggle("loading", tone === "loading");
  renderMarkdownLite(synthesisAnswer, normalized);

  if (tone === "loading" || tone === "error") {
    synthesisCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function endpointForOutputMode(mode) {
  const base = `/${collectionName.textContent || "support_tickets"}`;
  if (mode === "translated") {
    return `${base}/translated`;
  }
  if (mode === "reranked") {
    return `${base}/reranked`;
  }
  if (mode === "reranked-translated") {
    return `${base}/reranked/translated`;
  }
  if (mode === "translated-reranked") {
    return `${base}/translated/reranked`;
  }
  return base;
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    apiStatus.textContent = "Connectée";
    collectionName.textContent = data.collection || "Inconnue";
  } catch (error) {
    apiStatus.textContent = "Indisponible";
    collectionName.textContent = "support_tickets";
    setFeedback("L'API ne répond pas encore sur /health.", "error");
  }
}

async function loadFilters() {
  for (const field of filterFields) {
    setSelectState(field, "Chargement...");
  }

  try {
    const response = await fetch("/filters");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    let totalOptions = 0;

    for (const field of filterFields) {
      const values = data[field] || [];
      totalOptions += values.length;
      populateSelect(field, values);
    }

    filtersStatus.textContent = totalOptions ? "" : "Aucune valeur distincte détectée pour le moment.";
    updateActiveFilters();
  } catch (error) {
    for (const field of filterFields) {
      setSelectState(field, "Indisponible");
    }
    filtersStatus.textContent = "Les options n'ont pas pu être chargées.";
    setFeedback("Impossible de charger les valeurs de filtres depuis l'index.", "error");
  }
}

document.querySelectorAll("[data-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = button.dataset.target;
    const hiddenInput = document.getElementById(target);
    hiddenInput.value = button.dataset.value;

    document
      .querySelectorAll(`[data-target="${target}"]`)
      .forEach((item) => item.classList.toggle("active", item === button));
  });
});

for (const field of filterFields) {
  document.getElementById(field).addEventListener("change", updateActiveFilters);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  if (!query) {
    setFeedback("La requête est obligatoire.", "error");
    queryInput.focus();
    return;
  }

  const payload = {
    query,
    mode: searchModeInput.value,
    filter: null,
  };

  const filters = buildFilters();
  if (Object.keys(filters).length) {
    payload.filter = filters;
  }

  const endpoint = endpointForOutputMode(outputModeInput.value);

  submitButton.disabled = true;
  currentRetrievedPayload = null;
  currentRetrievedQuery = "";
  clearSynthesis();
  setFeedback("Recherche en cours...", "");
  resultsTitle.textContent = "Recherche en cours";
  resultFlags.textContent = outputLabels[outputModeInput.value] || "Brut";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const normalized = normalizeResults(data);
    currentRetrievedPayload = normalized.payload;
    currentRetrievedQuery = query;

    resultsTitle.textContent = `${normalized.items.length} ticket${normalized.items.length > 1 ? "s" : ""} trouvé${normalized.items.length > 1 ? "s" : ""}`;
    resultFlags.textContent = normalized.flags;
    setFeedback(normalized.items.length ? "Recherche terminée." : "Recherche terminée sans résultat.", normalized.items.length ? "success" : "");
    renderResults(normalized.items);
    renderSynthesis(normalized.answer);
  } catch (error) {
    resultsTitle.textContent = "Erreur de recherche";
    resultFlags.textContent = "Échec";
    setFeedback("Impossible d'interroger l'API de recherche.", "error");
    renderEmptyState("Vérifie que Qdrant est lancé, que la collection existe et que l'API répond.");
  } finally {
    submitButton.disabled = false;
  }
});

synthesizeButton.addEventListener("click", async () => {
  console.info("Synthesis button clicked");

  if (synthesisInProgress) {
    return;
  }

  setFeedback("Préparation de la synthèse...", "");

  const query = currentRetrievedQuery || queryInput.value.trim();
  if (!query || !currentRetrievedPayload || !Object.keys(currentRetrievedPayload).length) {
    renderSynthesis("Lance d'abord une recherche avant de synthétiser.", "error");
    setFeedback("Lance d'abord une recherche avant de synthétiser.", "error");
    return;
  }

  synthesisInProgress = true;
  synthesizeButton.disabled = true;
  synthesizeButton.textContent = "Synthèse en cours...";
  renderSynthesis("Synthèse en cours...", "loading");
  setFeedback("Synthèse des résultats affichés en cours...", "");

  try {
    const response = await fetch(`/${collectionName.textContent || "support_tickets"}/synthesize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        tickets: currentRetrievedPayload,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    if (data.synthesized !== "success" || !data.answer) {
      throw new Error(data.error || "Synthesis failed");
    }

    renderSynthesis(data.answer);
    resultFlags.textContent = `${resultFlags.textContent.replace(/ \| Synthétisé: .+$/, "")} | Synthétisé: success`;
    setFeedback("Synthèse terminée.", "success");
  } catch (error) {
    renderSynthesis(`Impossible de générer la synthèse.\n\n${error.message}`, "error");
    setFeedback(`Impossible de synthétiser les résultats affichés. ${error.message}`, "error");
  } finally {
    synthesisInProgress = false;
    synthesizeButton.disabled = false;
    synthesizeButton.textContent = "Synthétiser les résultats";
  }
});

resetButton.addEventListener("click", () => {
  form.reset();
  searchModeInput.value = "hybrid";
  outputModeInput.value = "raw";
  document.querySelectorAll('[data-target="search-mode"]').forEach((item) => item.classList.toggle("active", item.dataset.value === "hybrid"));
  document.querySelectorAll('[data-target="output-mode"]').forEach((item) => item.classList.toggle("active", item.dataset.value === "raw"));
  resultsTitle.textContent = "Aucune recherche lancée";
  resultFlags.textContent = "Brut";
  setFeedback("");
  clearSynthesis();
  resultsContainer.innerHTML = "";
  currentRetrievedPayload = null;
  currentRetrievedQuery = "";
  updateActiveFilters();
});

loadHealth();
loadFilters();
