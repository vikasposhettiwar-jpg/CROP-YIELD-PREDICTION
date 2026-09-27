const API_BASE = "http://127.0.0.1:5002";

const form = document.getElementById("predictionForm");
const yieldResult = document.getElementById("yieldResult");
const recommendResult = document.getElementById("recommendResult");
const trendBtn = document.getElementById("trendBtn");
const trendChart = document.getElementById("trendChart");
const trendError = document.getElementById("trendError");
const saveEntryBtn = document.getElementById("saveEntryBtn");
const retrainBtn = document.getElementById("retrainBtn");
const trainingMessage = document.getElementById("trainingMessage");

function getPayload() {
  return {
    country: document.getElementById("country").value,
    state: document.getElementById("state").value,
    district: document.getElementById("district").value,
    season: document.getElementById("season").value,
    crop: document.getElementById("crop").value,
    year: Number(document.getElementById("year").value),
    temperature: Number(document.getElementById("temperature").value),
    rainfall: Number(document.getElementById("rainfall").value),
    humidity: Number(document.getElementById("humidity").value),
    soil_type: document.getElementById("soil_type").value,
  };
}

function getActualYieldValue() {
  const raw = document.getElementById("actual_yield_quintal_per_acre")?.value;
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isNaN(parsed) ? null : parsed;
}

async function callApi(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = getPayload();
  yieldResult.textContent = "Processing...";
  recommendResult.textContent = "Processing...";

  try {
    const [predictData, recommendData] = await Promise.all([
      callApi("/api/predict", payload),
      callApi("/api/recommend", payload),
    ]);
    yieldResult.textContent = `Predicted yield: ${predictData.predicted_yield_quintal_per_acre} q/acre`;
    recommendResult.textContent = `Recommended crop: ${recommendData.recommended_crop} | Expected yield: ${recommendData.expected_yield_quintal_per_acre} q/acre`;
    yieldResult.className = "success";
    recommendResult.className = "success";
  } catch (error) {
    yieldResult.textContent = error.message;
    recommendResult.textContent = error.message;
    yieldResult.className = "error";
    recommendResult.className = "error";
  }
});

function getTrendYears() {
  const start = Number(document.getElementById("trend_start_year").value);
  const end = Number(document.getElementById("trend_end_year").value);
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) {
    throw new Error("Enter valid start/end years.");
  }

  const years = [];
  for (let y = start; y <= end; y += 1) {
    years.push(y);
  }
  if (years.length < 2 || years.length > 4) {
    throw new Error("Please choose a range of 2 to 4 years.");
  }
  return years;
}

function renderTrendBars(trendData) {
  trendChart.innerHTML = "";
  if (!trendData.length) return;

  const maxYield = Math.max(...trendData.map((item) => item.predicted_yield_quintal_per_acre));
  trendData.forEach((item) => {
    const barHeight =
      maxYield > 0 ? Math.round((item.predicted_yield_quintal_per_acre / maxYield) * 160) : 8;
    const wrapper = document.createElement("div");
    wrapper.className = "bar-item";
    wrapper.innerHTML = `
      <div class="bar-label">${item.predicted_yield_quintal_per_acre.toFixed(2)} q/acre</div>
      <div class="bar" style="height:${barHeight}px"></div>
      <div class="bar-label">${item.year}</div>
    `;
    trendChart.appendChild(wrapper);
  });
}

trendBtn.addEventListener("click", async () => {
  trendError.textContent = "";
  try {
    const payload = getPayload();
    const years = getTrendYears();
    const trendData = await callApi("/api/predict-trend", { ...payload, years });
    renderTrendBars(trendData.trend || []);
  } catch (error) {
    trendError.textContent = error.message;
    trendChart.innerHTML = "";
  }
});

saveEntryBtn.addEventListener("click", async () => {
  trainingMessage.className = "";
  trainingMessage.textContent = "Saving entry...";
  try {
    const payload = getPayload();
    const actualYield = getActualYieldValue();
    if (actualYield !== null) payload.actual_yield_quintal_per_acre = actualYield;
    const data = await callApi("/api/save-entry", payload);
    trainingMessage.className = "success";
    trainingMessage.textContent = `Saved entry #${data.entry_id}. Label source: ${data.label_source}.`;
  } catch (error) {
    trainingMessage.className = "error";
    trainingMessage.textContent = error.message;
  }
});

retrainBtn.addEventListener("click", async () => {
  trainingMessage.className = "";
  trainingMessage.textContent = "Retraining model from database records...";
  try {
    const data = await callApi("/api/retrain-from-db", { min_records: 5, include_base_dataset: true });
    trainingMessage.className = "success";
    trainingMessage.textContent = `Retraining complete. Best model: ${data.best_model}. Actual records: ${data.actual_records_used}. Total rows: ${data.total_training_rows}.`;
  } catch (error) {
    trainingMessage.className = "error";
    trainingMessage.textContent = error.message;
  }
});

