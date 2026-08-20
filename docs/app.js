// Retail Sales Analytics dashboard
// Loads the small json files under data/ and draws the charts with Plotly.
// Plain vanilla JavaScript, no build step, no framework.

var PLOTLY_COLORS = ["#2f5d9c", "#5b8def", "#7ea16b", "#c9884f", "#a15b8f", "#4a9fa8"];

var PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

var state = {
  overview: null,
  sales: null,
  customers: null,
  models: null,
  recommendations: null,
};

function showFatalError(message) {
  var banner = document.getElementById("app-status");
  banner.textContent = "Could not load the dashboard data: " + message;
  banner.hidden = false;
}

function fetchJSON(path) {
  return fetch(path).then(function (response) {
    if (!response.ok) {
      throw new Error(path + " returned HTTP " + response.status);
    }
    return response.json();
  });
}

function formatCurrency(value) {
  if (value === null || value === undefined) return "n/a";
  if (Math.abs(value) >= 1000000) {
    return "GBP " + (value / 1000000).toFixed(2) + "M";
  }
  return "GBP " + Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatNumber(value) {
  if (value === null || value === undefined) return "n/a";
  return Number(value).toLocaleString();
}

function formatPercent(value) {
  if (value === null || value === undefined) return "n/a";
  return Number(value).toFixed(2) + "%";
}

function formatDecimal(value, digits) {
  if (value === null || value === undefined) return "n/a";
  return Number(value).toFixed(digits === undefined ? 2 : digits);
}

function makeCard(label, value, sub) {
  var card = document.createElement("div");
  card.className = "kpi-card";

  var labelEl = document.createElement("span");
  labelEl.className = "kpi-label";
  labelEl.textContent = label;
  card.appendChild(labelEl);

  var valueEl = document.createElement("span");
  valueEl.className = "kpi-value";
  valueEl.textContent = value;
  card.appendChild(valueEl);

  if (sub) {
    var subEl = document.createElement("span");
    subEl.className = "kpi-sub";
    subEl.textContent = sub;
    card.appendChild(subEl);
  }

  return card;
}

// ---------- Overview ----------

function renderOverview(data) {
  var grid = document.getElementById("kpi-cards");
  grid.innerHTML = "";
  grid.appendChild(makeCard("Total Revenue", formatCurrency(data.total_revenue)));
  grid.appendChild(makeCard("Orders", formatNumber(data.orders)));
  grid.appendChild(makeCard("Active Customers", formatNumber(data.active_customers)));
  grid.appendChild(makeCard("Average Order Value", formatCurrency(data.average_order_value)));
  grid.appendChild(makeCard("Repeat Customer Rate", formatPercent(data.repeat_customer_rate)));
  grid.appendChild(makeCard("Cancellation Rate", formatPercent(data.cancellation_rate)));
}

// ---------- Sales ----------

function renderMonthlyRevenueChart(year) {
  var rows = state.sales.monthly_revenue;
  if (year && year !== "all") {
    rows = rows.filter(function (r) { return r.year_month.slice(0, 4) === year; });
  }
  var x = rows.map(function (r) { return r.year_month; });
  var y = rows.map(function (r) { return r.monthly_revenue; });

  Plotly.newPlot("chart-monthly-revenue", [{
    x: x, y: y, type: "scatter", mode: "lines+markers",
    line: { color: PLOTLY_COLORS[0] },
    hovertemplate: "%{x}<br>GBP %{y:,.0f}<extra></extra>",
  }], {
    margin: { t: 10, r: 20, b: 60, l: 70 },
    xaxis: { title: "Month" },
    yaxis: { title: "Revenue (GBP)" },
  }, PLOTLY_CONFIG);
}

function renderMonthlyOrdersChart() {
  var rows = state.sales.monthly_revenue;
  var x = rows.map(function (r) { return r.year_month; });
  var y = rows.map(function (r) { return r.order_count; });

  Plotly.newPlot("chart-monthly-orders", [{
    x: x, y: y, type: "bar",
    marker: { color: PLOTLY_COLORS[1] },
    hovertemplate: "%{x}<br>%{y:,} orders<extra></extra>",
  }], {
    margin: { t: 10, r: 20, b: 60, l: 60 },
    xaxis: { title: "Month" },
    yaxis: { title: "Orders" },
  }, PLOTLY_CONFIG);
}

function renderCountryChart(topN) {
  var rows = state.sales.revenue_by_country.slice(0, topN);
  rows = rows.slice().reverse();
  var y = rows.map(function (r) { return r.country; });
  var x = rows.map(function (r) { return r.total_revenue; });

  Plotly.newPlot("chart-revenue-country", [{
    x: x, y: y, type: "bar", orientation: "h",
    marker: { color: PLOTLY_COLORS[2] },
    hovertemplate: "%{y}<br>GBP %{x:,.0f}<extra></extra>",
  }], {
    margin: { t: 10, r: 20, b: 50, l: 140 },
    xaxis: { title: "Revenue (GBP)" },
  }, PLOTLY_CONFIG);
}

function renderTopProductsChart(metric) {
  var rows = metric === "quantity" ? state.sales.top_products_by_quantity : state.sales.top_products_by_revenue;
  rows = rows.slice().reverse();
  var y = rows.map(function (r) { return r.description; });
  var x = rows.map(function (r) { return metric === "quantity" ? r.total_units_sold : r.total_revenue; });
  var hover = metric === "quantity" ? "%{y}<br>%{x:,} units<extra></extra>" : "%{y}<br>GBP %{x:,.0f}<extra></extra>";

  Plotly.newPlot("chart-top-products", [{
    x: x, y: y, type: "bar", orientation: "h",
    marker: { color: PLOTLY_COLORS[3] },
    hovertemplate: hover,
  }], {
    margin: { t: 10, r: 20, b: 50, l: 260 },
    xaxis: { title: metric === "quantity" ? "Units sold" : "Revenue (GBP)" },
  }, PLOTLY_CONFIG);
}

function renderSales(data) {
  var yearSelect = document.getElementById("sales-year-select");
  yearSelect.innerHTML = "";
  var allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "All";
  yearSelect.appendChild(allOption);
  data.years.forEach(function (year) {
    var opt = document.createElement("option");
    opt.value = year;
    opt.textContent = year;
    yearSelect.appendChild(opt);
  });
  yearSelect.addEventListener("change", function () {
    renderMonthlyRevenueChart(yearSelect.value);
  });

  var countrySelect = document.getElementById("country-top-select");
  countrySelect.addEventListener("change", function () {
    renderCountryChart(Number(countrySelect.value));
  });

  var productSelect = document.getElementById("product-metric-select");
  productSelect.addEventListener("change", function () {
    renderTopProductsChart(productSelect.value);
  });

  renderMonthlyRevenueChart("all");
  renderMonthlyOrdersChart();
  renderCountryChart(Number(countrySelect.value));
  renderTopProductsChart(productSelect.value);
}

// ---------- Customers ----------

function renderSegments(data) {
  var segments = data.segments;

  var labels = segments.map(function (s) { return s.segment_name; });
  var values = segments.map(function (s) { return s.customers; });

  Plotly.newPlot("chart-segments", [{
    labels: labels, values: values, type: "pie", hole: 0.45,
    marker: { colors: PLOTLY_COLORS },
    hovertemplate: "%{label}<br>%{value} customers (%{percent})<extra></extra>",
  }], {
    margin: { t: 10, r: 10, b: 10, l: 10 },
  }, PLOTLY_CONFIG);

  var tbody = document.querySelector("#segment-table tbody");
  tbody.innerHTML = "";
  segments.forEach(function (s) {
    var row = document.createElement("tr");
    row.innerHTML =
      "<td>" + s.segment_name + "</td>" +
      "<td>" + formatNumber(s.customers) + "</td>" +
      "<td>" + formatDecimal(s.avg_recency, 0) + "</td>" +
      "<td>" + formatDecimal(s.avg_frequency, 1) + "</td>" +
      "<td>" + formatCurrency(s.avg_monetary) + "</td>";
    tbody.appendChild(row);
  });
}

function renderChurnSummary(churn) {
  var grid = document.getElementById("churn-summary-cards");
  grid.innerHTML = "";
  grid.appendChild(makeCard("F1 (Logistic Regression)", formatDecimal(churn.logistic_regression.f1, 3)));
  grid.appendChild(makeCard("ROC-AUC (Logistic Regression)", formatDecimal(churn.logistic_regression.roc_auc, 3)));
}

function renderChurnDrivers(rows) {
  var sorted = rows.slice().sort(function (a, b) { return a.coefficient - b.coefficient; });
  var y = sorted.map(function (r) { return r.feature; });
  var x = sorted.map(function (r) { return r.coefficient; });
  var colors = x.map(function (v) { return v >= 0 ? "#b3413a" : "#2e7d4f"; });

  Plotly.newPlot("chart-churn-drivers", [{
    x: x, y: y, type: "bar", orientation: "h",
    marker: { color: colors },
    hovertemplate: "%{y}<br>coefficient %{x:.3f}<extra></extra>",
  }], {
    margin: { t: 10, r: 20, b: 50, l: 170 },
    xaxis: { title: "Coefficient (standardized features, higher = more likely to churn)" },
  }, PLOTLY_CONFIG);
}

function renderCustomers(data) {
  renderSegments(data);
  renderChurnSummary(data.churn);
  renderChurnDrivers(data.churn_feature_importance);
}

// ---------- Models ----------

function renderSalesForecastModel(sf) {
  var grid = document.getElementById("sales-forecast-cards");
  grid.innerHTML = "";
  grid.appendChild(makeCard("Baseline MAE", formatCurrency(sf.baseline_mae)));
  grid.appendChild(makeCard("Model MAE (" + sf.model_name + ")", formatCurrency(sf.model_mae)));
  grid.appendChild(makeCard("Baseline RMSE", formatCurrency(sf.baseline_rmse)));
  grid.appendChild(makeCard("Model RMSE", formatCurrency(sf.model_rmse)));

  var weeks = sf.test_weeks;
  var x = weeks.map(function (w) { return w.week_start; });

  Plotly.newPlot("chart-sales-forecast", [
    { x: x, y: weeks.map(function (w) { return w.actual_revenue; }), name: "Actual", type: "scatter", mode: "lines+markers", line: { color: PLOTLY_COLORS[0] } },
    { x: x, y: weeks.map(function (w) { return w.predicted_revenue; }), name: "Predicted", type: "scatter", mode: "lines+markers", line: { color: PLOTLY_COLORS[1] } },
    { x: x, y: weeks.map(function (w) { return w.naive_prediction; }), name: "Naive baseline", type: "scatter", mode: "lines", line: { color: "#9aa4ae", dash: "dash" } },
  ], {
    margin: { t: 10, r: 20, b: 60, l: 70 },
    xaxis: { title: "Week (final test period)" },
    yaxis: { title: "Revenue (GBP)" },
    legend: { orientation: "h", y: -0.25 },
  }, PLOTLY_CONFIG);
}

function renderDemandChart(productCode) {
  var product = state.models.demand_prediction.products[productCode];
  if (!product) return;

  Plotly.newPlot("chart-demand", [
    { x: product.weeks, y: product.actual, name: "Actual", type: "scatter", mode: "lines+markers", line: { color: PLOTLY_COLORS[0] } },
    { x: product.weeks, y: product.predicted, name: "Predicted", type: "scatter", mode: "lines+markers", line: { color: PLOTLY_COLORS[1] } },
    { x: product.weeks, y: product.naive, name: "Baseline (last week)", type: "scatter", mode: "lines", line: { color: "#9aa4ae", dash: "dash" } },
  ], {
    margin: { t: 10, r: 20, b: 60, l: 60 },
    xaxis: { title: "Week (final test period)" },
    yaxis: { title: "Units" },
    legend: { orientation: "h", y: -0.25 },
  }, PLOTLY_CONFIG);
}

function renderDemandModel(dp) {
  var grid = document.getElementById("demand-cards");
  grid.innerHTML = "";
  grid.appendChild(makeCard("Baseline MAE", formatDecimal(dp.baseline_mae, 1) + " units"));
  grid.appendChild(makeCard("Model MAE (" + dp.model_name + ")", formatDecimal(dp.model_mae, 1) + " units"));
  grid.appendChild(makeCard("Model RMSE", formatDecimal(dp.model_rmse, 1) + " units"));

  var select = document.getElementById("demand-product-select");
  select.innerHTML = "";
  Object.keys(dp.products).forEach(function (code) {
    var opt = document.createElement("option");
    opt.value = code;
    opt.textContent = code + " - " + dp.products[code].description;
    select.appendChild(opt);
  });
  select.addEventListener("change", function () {
    renderDemandChart(select.value);
  });

  if (select.options.length > 0) {
    renderDemandChart(select.options[0].value);
  }
}

function renderChurnFull(churn) {
  var grid = document.getElementById("churn-full-cards");
  grid.innerHTML = "";
  grid.appendChild(makeCard("Precision (Logistic Regression)", formatDecimal(churn.logistic_regression.precision, 3)));
  grid.appendChild(makeCard("Recall (Logistic Regression)", formatDecimal(churn.logistic_regression.recall, 3)));
  grid.appendChild(makeCard("F1 (Logistic Regression)", formatDecimal(churn.logistic_regression.f1, 3)));
  grid.appendChild(makeCard("ROC-AUC (Logistic Regression)", formatDecimal(churn.logistic_regression.roc_auc, 3)));
  grid.appendChild(makeCard("F1 (Random Forest)", formatDecimal(churn.random_forest.f1, 3), "comparison model"));
  grid.appendChild(makeCard("ROC-AUC (Random Forest)", formatDecimal(churn.random_forest.roc_auc, 3), "comparison model"));
}

function renderPriceModel(price) {
  var grid = document.getElementById("price-cards");
  grid.innerHTML = "";
  grid.appendChild(makeCard("Baseline MAE", formatDecimal(price.baseline_mae, 2)));
  grid.appendChild(makeCard("Ridge MAE", formatDecimal(price.model_mae, 2)));
  grid.appendChild(makeCard("Ridge RMSE", formatDecimal(price.model_rmse, 2)));
  grid.appendChild(makeCard("Ridge R2", formatDecimal(price.model_r2, 3)));
}

function renderModels(data) {
  renderSalesForecastModel(data.sales_forecast);
  renderDemandModel(data.demand_prediction);
  renderChurnFull(data.churn);
  renderPriceModel(data.price_estimation);
}

// ---------- Recommendations ----------

function renderRecommendationTable(productCode) {
  var product = state.recommendations.products[productCode];
  var leadText = document.getElementById("recommend-lead-text");
  var tbody = document.querySelector("#recommend-table tbody");
  tbody.innerHTML = "";

  if (!product) return;

  leadText.textContent = "Customers who bought \"" + product.description + "\" also tended to buy:";

  product.recommendations.forEach(function (rec) {
    var row = document.createElement("tr");
    row.innerHTML =
      "<td>" + rec.description + "</td>" +
      "<td>" + formatDecimal(rec.similarity, 3) + "</td>";
    tbody.appendChild(row);
  });
}

function renderRecommendations(data) {
  var grid = document.getElementById("recommendation-hitrate-cards");
  grid.innerHTML = "";
  grid.appendChild(makeCard("Popularity Baseline Hit Rate@5", formatDecimal(data.popularity_baseline_hit_rate, 3)));
  grid.appendChild(makeCard("Item-Item Recommender Hit Rate@5", formatDecimal(data.item_item_hit_rate, 3)));

  var select = document.getElementById("recommend-product-select");
  select.innerHTML = "";
  Object.keys(data.products).forEach(function (code) {
    var opt = document.createElement("option");
    opt.value = code;
    opt.textContent = code + " - " + data.products[code].description;
    select.appendChild(opt);
  });
  select.addEventListener("change", function () {
    renderRecommendationTable(select.value);
  });

  if (select.options.length > 0) {
    renderRecommendationTable(select.options[0].value);
  }
}

// ---------- Boot ----------

function loadAll() {
  var files = ["overview", "sales", "customers", "models", "recommendations"];
  var promises = files.map(function (name) { return fetchJSON("data/" + name + ".json"); });

  Promise.all(promises).then(function (results) {
    files.forEach(function (name, i) { state[name] = results[i]; });

    renderOverview(state.overview);
    renderSales(state.sales);
    renderCustomers(state.customers);
    renderModels(state.models);
    renderRecommendations(state.recommendations);
  }).catch(function (error) {
    showFatalError(error.message);
    console.error(error);
  });
}

window.addEventListener("DOMContentLoaded", loadAll);
