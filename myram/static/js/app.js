const form = document.getElementById("predictionForm");
const yieldResult = document.getElementById("yieldResult");
const recommendResult = document.getElementById("recommendResult");
const modelCards = document.getElementById("modelCards");
const featureList = document.getElementById("featureList");

function getPayload() {
  return {
    temperature: Number(document.getElementById("temperature").value),
    rainfall: Number(document.getElementById("rainfall").value),
    humidity: Number(document.getElementById("humidity").value),
    soil_type: document.getElementById("soil_type").value,
  };
}

async function callApi(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = getPayload();

  yieldResult.textContent = "Processing...";
  yieldResult.className = "";
  recommendResult.textContent = "Processing...";
  recommendResult.className = "";

  try {
    const [pred, rec] = await Promise.all([callApi("/predict", payload), callApi("/recommend", payload)]);
    yieldResult.textContent = `Predicted yield: ${pred.predicted_yield_quintal_per_acre} q/acre`;
    yieldResult.className = "success";
    recommendResult.textContent = `Recommended crop: ${rec.recommended_crop} (score: ${rec.confidence_score})`;
    recommendResult.className = "success";
  } catch (err) {
    yieldResult.textContent = err.message;
    yieldResult.className = "error";
    recommendResult.textContent = err.message;
    recommendResult.className = "error";
  }
});

async function loadDashboard() {
  try {
    const res = await fetch("/api/dashboard");
    const data = await res.json();
    const metrics = data.metrics?.metrics || {};
    const best = data.metrics?.best_model || "N/A";

    modelCards.innerHTML = Object.entries(metrics)
      .map(
        ([name, m]) => `
          <div class="metric-card">
            <strong>${name}</strong><br />
            RMSE: ${m.rmse.toFixed(3)}<br />
            MAE: ${m.mae.toFixed(3)}<br />
            R²: ${m.r2.toFixed(3)}<br />
            Best Model: ${name === best ? "Yes" : "No"}
          </div>`
      )
      .join("");

    const features = data.feature_importance?.features || [];
    const importances = data.feature_importance?.importances || [];
    featureList.innerHTML = features
      .map((feature, i) => `<li>${feature}: ${Number(importances[i]).toFixed(4)}</li>`)
      .join("");

    if (!features.length) {
      featureList.innerHTML = "<li>No feature importance available (best model may be linear).</li>";
    }
  } catch (err) {
    modelCards.innerHTML = `<div class="metric-card error">${err.message}</div>`;
  }
}

loadDashboard();

