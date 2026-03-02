const pptxgen = require("pptxgenjs");

// Colors
const BG = "0D1229";
const CARD_BG = "1E2341";
const BLUE = "60A5FA";
const PURPLE = "A78BFA";
const GREEN = "4ADE80";
const ORANGE = "FBBF24";
const CYAN = "22D3EE";
const PINK = "F472B6";
const RED = "F87171";
const GOLD = "FBBF24";
const GRAY = "8B95B0";
const LIGHT = "B0B8D0";
const WHITE = "FFFFFF";

function addBadge(slide, text, x, y, colors) {
  const c1 = colors || "22C55E";
  slide.addShape("rect", { x, y, w: 1.8, h: 0.32, fill: { color: c1 }, rectRadius: 0.05 });
  slide.addText(text, { x, y, w: 1.8, h: 0.32, fontSize: 9, fontFace: "Arial", bold: true, color: WHITE, align: "center", valign: "middle", letterSpacing: 1.5 });
}

function addTagline(slide, text, y) {
  y = y || 4.95;
  slide.addShape("rect", { x: 0.4, y, w: 9.2, h: 0.4, fill: { color: "141E32" }, line: { color: "2A3560", width: 0.5 }, rectRadius: 0.05 });
  slide.addText(text, { x: 0.4, y, w: 9.2, h: 0.4, fontSize: 11, fontFace: "Arial", bold: true, color: GOLD, align: "center", valign: "middle" });
}

function addCard(slide, title, body, x, y, w, h, accentColor) {
  slide.addShape("rect", { x, y, w, h, fill: { color: CARD_BG }, rectRadius: 0.08 });
  slide.addShape("rect", { x, y, w: 0.06, h, fill: { color: accentColor || BLUE } });
  slide.addText(title, { x: x + 0.15, y, w: w - 0.2, h: 0.35, fontSize: 13, fontFace: "Arial", bold: true, color: accentColor || BLUE, valign: "top", margin: [4, 0, 0, 0] });
  slide.addText(body, { x: x + 0.15, y: y + 0.32, w: w - 0.2, h: h - 0.36, fontSize: 10, fontFace: "Arial", color: LIGHT, valign: "top", lineSpacingMultiple: 1.3 });
}

function addNumberedItem(slide, num, title, desc, x, y, w, circleColor, titleColor) {
  slide.addShape("rect", { x, y, w, h: 0.55, fill: { color: CARD_BG }, rectRadius: 0.06 });
  slide.addShape("oval", { x: x + 0.1, y: y + 0.1, w: 0.35, h: 0.35, fill: { color: circleColor } });
  slide.addText(String(num), { x: x + 0.1, y: y + 0.1, w: 0.35, h: 0.35, fontSize: 11, fontFace: "Arial", bold: true, color: WHITE, align: "center", valign: "middle" });
  slide.addText(title, { x: x + 0.55, y: y + 0.05, w: w - 0.65, h: 0.22, fontSize: 11, fontFace: "Arial", bold: true, color: titleColor || BLUE });
  slide.addText(desc, { x: x + 0.55, y: y + 0.27, w: w - 0.65, h: 0.23, fontSize: 9, fontFace: "Arial", color: LIGHT });
}

function titleSlide(pres, title, subtitle, week, author, meta) {
  let s = pres.addSlide();
  s.background = { color: BG };
  s.addShape("oval", { x: 2.5, y: -0.5, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
  s.addShape("oval", { x: 6, y: 3.5, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
  s.addText("DEVOPS WITH VIBECODING", { x: 0, y: 1.2, w: 10, h: 0.35, fontSize: 11, fontFace: "Arial", color: "8B5CF6", align: "center", charSpacing: 4 });
  s.addText(title, { x: 0.5, y: 1.7, w: 9, h: 0.6, fontSize: 28, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText(subtitle, { x: 1, y: 2.4, w: 8, h: 0.4, fontSize: 16, fontFace: "Arial", color: BLUE, align: "center" });
  s.addText(week, { x: 3.5, y: 3.1, w: 3, h: 0.35, fontSize: 12, fontFace: "Arial", color: GRAY, align: "center" });
  s.addText(author || "Anirach Mingkhwan", { x: 2, y: 3.6, w: 6, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
  s.addText(meta || "FITM, KMUTNB", { x: 2, y: 3.9, w: 6, h: 0.3, fontSize: 10, fontFace: "Arial", color: GRAY, align: "center" });
  return s;
}

function sectionSlide(pres, num, title, subtitle) {
  let s = pres.addSlide();
  s.background = { color: BG };
  s.addShape("rect", { x: 0, y: 0, w: 10, h: 5.63, fill: { color: "111936" } });
  s.addShape("oval", { x: -1, y: 1, w: 4, h: 4, fill: { color: "8B5CF6", transparency: 92 } });
  s.addShape("oval", { x: 7, y: -0.5, w: 3, h: 3, fill: { color: "3B82F6", transparency: 92 } });
  s.addText(num, { x: 3.5, y: 1.5, w: 3, h: 0.5, fontSize: 14, fontFace: "Arial", color: PURPLE, align: "center", charSpacing: 3 });
  s.addText(title, { x: 1, y: 2.1, w: 8, h: 0.6, fontSize: 28, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
  s.addText(subtitle || "", { x: 1.5, y: 2.8, w: 7, h: 0.4, fontSize: 14, fontFace: "Arial", color: BLUE, align: "center" });
  return s;
}

function contentSlide(pres, title) {
  let s = pres.addSlide();
  s.background = { color: BG };
  s.addShape("rect", { x: 0, y: 0, w: 10, h: 0.9, fill: { color: "111936" } });
  s.addShape("rect", { x: 0, y: 0.88, w: 10, h: 0.03, fill: { color: PURPLE, transparency: 50 } });
  s.addText(title, { x: 0.5, y: 0.15, w: 9, h: 0.6, fontSize: 20, fontFace: "Arial", bold: true, color: WHITE });
  return s;
}

// ============================================================
// BUILD
// ============================================================
let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

// Slide 1: Title
titleSlide(pres, "Monitoring, Logging\n& Observability", "Building Production-Grade Visibility", "Week 7", "Anirach Mingkhwan", "FITM, KMUTNB");

// Slide 2: Learning Objectives
let s = contentSlide(pres, "Learning Objectives");
addNumberedItem(s, 1, "Three Pillars", "Distinguish monitoring, logging & observability concepts", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Prometheus + Grafana", "Deploy metrics collection and visualization stack", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Structured Logging", "Implement centralized log aggregation with Loki", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Distributed Tracing", "Trace request flows across microservices", 0.5, 3.05, 9, CYAN, CYAN);
addNumberedItem(s, 5, "AI for Observability", "Leverage AI for anomaly detection & log analysis", 0.5, 3.7, 9, ORANGE, ORANGE);
addTagline(s, '"You can\'t fix what you can\'t see" - Observability Principle');

// Slide 3: Agenda
s = contentSlide(pres, "Today's Agenda");
addCard(s, "Part 1: Foundations", "Three Pillars of Observability\nOpenTelemetry Standard\nMonitoring vs Observability", 0.5, 1.1, 4.3, 1.5, PURPLE);
addCard(s, "Part 2: Metrics", "Prometheus Architecture\nGrafana Dashboards\nRED & USE Methods", 5.2, 1.1, 4.3, 1.5, BLUE);
addCard(s, "Part 3: Logging", "EFK vs Loki Stack\nStructured Logging\nLogQL Queries", 0.5, 2.8, 4.3, 1.5, GREEN);
addCard(s, "Part 4: AI + Alerting", "AI-Powered Observability\nSLO-Based Alerting\nHands-on Lab", 5.2, 2.8, 4.3, 1.5, ORANGE);

// ============= SECTION 1: THREE PILLARS =============
sectionSlide(pres, "SECTION 01", "The Three Pillars\nof Observability", "Metrics, Logs, and Traces");

// Slide 5: Monitoring vs Observability
s = contentSlide(pres, "Monitoring vs Observability");
addCard(s, "Monitoring", "- Predefined dashboards & alerts\n- Answers known questions\n- \"Is the system up?\"\n- Reactive approach\n- Tracks expected failures", 0.5, 1.1, 4.3, 2.5, BLUE);
addCard(s, "Observability", "- Explore unknown unknowns\n- Answers novel questions\n- \"Why is it slow for user X?\"\n- Proactive debugging\n- High-cardinality data", 5.2, 1.1, 4.3, 2.5, PURPLE);
addTagline(s, "Observability = ability to understand internal state from external outputs");

// Slide 6: Three Pillars Overview
s = contentSlide(pres, "The Three Pillars");
addCard(s, "Metrics", "Numerical measurements at intervals\nCounters, Gauges, Histograms\nEfficient to store and query\nBest for: trends, alerting, SLOs", 0.5, 1.1, 2.8, 2.3, BLUE);
addCard(s, "Logs", "Timestamped event records\nStructured JSON preferred\nDetailed but expensive at scale\nBest for: debugging, audit trails", 3.6, 1.1, 2.8, 2.3, GREEN);
addCard(s, "Traces", "Request propagation paths\nSpans per service hop\nEnd-to-end latency breakdown\nBest for: distributed debugging", 6.7, 1.1, 2.8, 2.3, ORANGE);
addTagline(s, "All three pillars complement each other - use them together");

// Slide 7: Metric Types
s = contentSlide(pres, "Metric Types Deep Dive");
addNumberedItem(s, 1, "Counter", "Monotonically increasing value (e.g., total requests, errors)", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "Gauge", "Value that goes up and down (e.g., CPU usage, memory, queue size)", 0.5, 1.75, 9, GREEN, GREEN);
addNumberedItem(s, 3, "Histogram", "Distribution of values in buckets (e.g., request latency percentiles)", 0.5, 2.4, 9, PURPLE, PURPLE);
addNumberedItem(s, 4, "Summary", "Pre-calculated quantiles on client side (e.g., p50, p95, p99)", 0.5, 3.05, 9, ORANGE, ORANGE);
addTagline(s, "Choose the right metric type for what you're measuring");

// Slide 8: OpenTelemetry
s = contentSlide(pres, "OpenTelemetry (OTel)");
addCard(s, "What is OTel?", "Vendor-neutral, open standard for instrumentation\nMerged from OpenTracing + OpenCensus\nCNCF project - industry standard", 0.5, 1.1, 4.3, 1.7, CYAN);
addCard(s, "Components", "APIs & SDKs for multiple languages\nCollector for processing & export\nSemantic conventions for naming\nAuto-instrumentation support", 5.2, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "Why OTel?", "- One SDK for metrics, logs, and traces\n- Avoid vendor lock-in\n- Export to Prometheus, Jaeger, Zipkin, Datadog, etc.\n- Growing ecosystem and community support", 0.5, 3.0, 9, 1.5, GREEN);

// ============= SECTION 2: PROMETHEUS & GRAFANA =============
sectionSlide(pres, "SECTION 02", "Prometheus & Grafana", "Metrics Collection and Visualization");

// Slide 10: Prometheus Architecture
s = contentSlide(pres, "Prometheus Architecture");
addCard(s, "Core Design", "Pull-based model - scrapes /metrics endpoints\nMulti-dimensional data model\nTime-series database (TSDB)\nPowerful query language: PromQL", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Components", "Prometheus Server (scrape + store)\nAlertmanager (routing + dedup)\nPushgateway (short-lived jobs)\nExporters (node, blackbox, etc.)\nService Discovery (K8s, DNS, file)", 5.2, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Key Features", "- No external dependencies\n- Reliable even when other systems fail\n- 15-day default retention\n- Federation for scaling", 0.5, 3.3, 9, 1.2, GREEN);

// Slide 11: PromQL Essentials
s = contentSlide(pres, "PromQL Query Language");
addCard(s, "Instant Vector", "http_requests_total{method=\"GET\"}\nSelects latest value for matching series", 0.5, 1.1, 4.3, 1.2, BLUE);
addCard(s, "Range Vector", "http_requests_total[5m]\nSelects values over time window", 5.2, 1.1, 4.3, 1.2, GREEN);
addCard(s, "Rate & Aggregation", "rate(http_requests_total[5m])\nsum by (status) (rate(...))\nhistogram_quantile(0.95, ...)", 0.5, 2.5, 4.3, 1.4, PURPLE);
addCard(s, "Common Patterns", "Error rate: rate(errors[5m]) / rate(total[5m])\nLatency p99: histogram_quantile(0.99, rate(duration_bucket[5m]))\nSaturation: container_memory / limit", 5.2, 2.5, 4.3, 1.4, ORANGE);
addTagline(s, "Master PromQL = Master Prometheus");

// Slide 12: Python Instrumentation
s = contentSlide(pres, "Instrumenting Python Apps");
addCard(s, "prometheus_client Library", "from prometheus_client import Counter, Histogram, start_http_server\n\nREQUESTS = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])\nLATENCY = Histogram('app_request_duration_seconds', 'Request latency')\n\n@LATENCY.time()\ndef handle_request():\n    REQUESTS.labels(method='GET', endpoint='/api').inc()", 0.5, 1.1, 9, 2.8, BLUE);
addCard(s, "Expose Metrics", "start_http_server(8000)  # /metrics on port 8000\n# Prometheus scrapes this endpoint automatically", 0.5, 4.1, 9, 0.7, GREEN);

// Slide 13: Grafana
s = contentSlide(pres, "Grafana Dashboards");
addCard(s, "What is Grafana?", "Open-source visualization & analytics\nMulti-source: Prometheus, Loki, ES, etc.\nTemplate variables for dynamic dashboards\nAlerting with notification channels", 0.5, 1.1, 4.3, 1.7, BLUE);
addCard(s, "Dashboard Best Practices", "Use template variables ($namespace, $pod)\nOrganize: Overview > Service > Detail\nInclude documentation in panels\nVersion control with JSON export", 5.2, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "Dashboard-as-Code", "Store dashboard JSON in Git\nUse Grafana provisioning or API\nTerraform grafana_dashboard resource\nGrafonnet (Jsonnet library) for templates", 0.5, 3.0, 9, 1.5, GREEN);

// Slide 14: RED & USE Methods
s = contentSlide(pres, "RED & USE Methods");
addCard(s, "RED Method (Services)", "Rate - requests per second\nErrors - failed requests per second\nDuration - latency distribution\n\nBest for: request-driven services\n\"How is my service performing?\"", 0.5, 1.1, 4.3, 2.5, RED);
addCard(s, "USE Method (Resources)", "Utilization - % time resource busy\nSaturation - queue depth / backlog\nErrors - error count per resource\n\nBest for: infrastructure resources\n\"Is my hardware the bottleneck?\"", 5.2, 1.1, 4.3, 2.5, CYAN);
addTagline(s, "RED for services, USE for resources - cover all your bases");

// Slide 15: Golden Signals
s = contentSlide(pres, "Google's Four Golden Signals");
addNumberedItem(s, 1, "Latency", "Time to serve a request - distinguish successful vs failed", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "Traffic", "Demand on the system - requests/sec, sessions, transactions", 0.5, 1.75, 9, GREEN, GREEN);
addNumberedItem(s, 3, "Errors", "Rate of failed requests - explicit (5xx) and implicit (wrong content)", 0.5, 2.4, 9, RED, RED);
addNumberedItem(s, 4, "Saturation", "How full the system is - CPU, memory, I/O, queue depth", 0.5, 3.05, 9, ORANGE, ORANGE);
addTagline(s, "From Google SRE Book - the signals that matter most");

// ============= SECTION 3: LOGGING =============
sectionSlide(pres, "SECTION 03", "Centralized Logging", "Structured Logs at Scale");

// Slide 17: Structured vs Unstructured
s = contentSlide(pres, "Structured vs Unstructured Logs");
addCard(s, "Unstructured", "2024-01-15 10:23:45 ERROR Failed to connect to database\n\n- Hard to parse and search\n- Inconsistent format\n- Regex-heavy processing\n- Difficult to aggregate", 0.5, 1.1, 4.3, 2.3, RED);
addCard(s, "Structured (JSON)", '{"timestamp":"2024-01-15T10:23:45Z",\n "level":"error",\n "msg":"Failed to connect to database",\n "service":"api",\n "host":"pod-abc"}\n\n- Machine-parseable\n- Consistent, queryable fields\n- Easy to filter and aggregate', 5.2, 1.1, 4.3, 2.3, GREEN);
addTagline(s, "Always use structured logging in production");

// Slide 18: EFK Stack
s = contentSlide(pres, "EFK Stack");
addCard(s, "Elasticsearch", "Full-text search engine\nDistributed, scalable indexing\nPowerful query DSL\nResource-heavy (CPU + RAM)", 0.5, 1.1, 2.8, 1.8, BLUE);
addCard(s, "Fluent Bit", "Lightweight log collector\nDaemonSet in Kubernetes\nEnriches with K8s metadata\nParsing, filtering, routing", 3.6, 1.1, 2.8, 1.8, GREEN);
addCard(s, "Kibana", "Visualization for Elasticsearch\nLog exploration & dashboards\nSaved searches & alerts\nDiscover view for debugging", 6.7, 1.1, 2.8, 1.8, PURPLE);
addCard(s, "When to Use EFK", "- Full-text search requirements\n- Complex log analytics\n- Large teams with dedicated ops\n- Compliance / audit trail needs", 0.5, 3.1, 9, 1.3, ORANGE);

// Slide 19: Loki
s = contentSlide(pres, "Grafana Loki - Lightweight Logging");
addCard(s, "Design Philosophy", "\"Like Prometheus, but for logs\"\nIndexes only labels, not content\nMuch cheaper than Elasticsearch\nHorizontally scalable", 0.5, 1.1, 4.3, 1.7, BLUE);
addCard(s, "LogQL", "Stream selector: {app=\"api\"}\nFilter: |= \"error\" != \"timeout\"\nParser: | json | line_format\nMetrics: rate({app=\"api\"} |= \"error\" [5m])", 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "Grafana + Loki", "- Single UI for metrics + logs\n- Click from alert to related logs\n- Correlate metrics and log patterns\n- Sufficient for most DevOps teams", 0.5, 3.0, 9, 1.4, PURPLE);

// Slide 20: EFK vs Loki comparison
s = contentSlide(pres, "EFK vs Loki: When to Choose");
addCard(s, "Choose EFK When", "- Need full-text search across all fields\n- Complex analytics & aggregations\n- Large team, dedicated ops budget\n- Compliance requirements\n- Already invested in Elastic ecosystem", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "Choose Loki When", "- Label-based filtering is sufficient\n- Cost-sensitive environment\n- Already using Grafana for metrics\n- Want simple deployment\n- Small-medium team", 5.2, 1.1, 4.3, 2.3, GREEN);
addTagline(s, "Start with Loki, move to EFK only when you need full-text search");

// Slide 21: Distributed Tracing
s = contentSlide(pres, "Distributed Tracing");
addCard(s, "Why Tracing?", "Microservices = complex request paths\nOne user request may hit 10+ services\nLatency could come from any hop\nTracing shows the complete picture", 0.5, 1.1, 4.3, 1.7, CYAN);
addCard(s, "Key Concepts", "Trace: end-to-end request journey\nSpan: single operation within a trace\nContext Propagation: passing trace ID\nSampling: not every request (head/tail)", 5.2, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "Tools", "Jaeger (CNCF, Uber) - full-featured, good UI | Zipkin (Twitter) - simpler, mature\nTempo (Grafana) - integrates with Loki+Prometheus | OpenTelemetry Collector - vendor-neutral pipeline", 0.5, 3.0, 9, 1.4, ORANGE);

// ============= SECTION 4: AI FOR OBSERVABILITY =============
sectionSlide(pres, "SECTION 04", "AI-Powered Observability", "Intelligent Monitoring & Analysis");

// Slide 23: AI Applications
s = contentSlide(pres, "AI for Observability");
addNumberedItem(s, 1, "Anomaly Detection", "ML models detect unusual patterns in metrics automatically", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Metric Forecasting", "Predict capacity needs and trend violations before they happen", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Alert Correlation", "Group related alerts to reduce noise and find root causes", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Root Cause Analysis", "AI traces through dependencies to identify failure origins", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Natural Language Queries", "Ask questions about your system in plain English", 0.5, 3.7, 9, CYAN, CYAN);

// Slide 24: AI + VibeCoding for Observability
s = contentSlide(pres, "VibeCoding for Observability");
addCard(s, "PromQL Generation", "Prompt: \"Show me error rate by service\"\nAI: rate(http_requests_total{status=~\"5..\"}[5m])\n       / rate(http_requests_total[5m])\n       * 100", 0.5, 1.1, 4.3, 1.7, BLUE);
addCard(s, "Dashboard Generation", "Prompt: \"Create a Grafana dashboard\n         for my Python API\"\nAI: Complete dashboard JSON with\n     RED metrics, resource panels, alerts", 5.2, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "Alert Rules from SLOs", "Prompt: \"Alert when availability drops below 99.9%\"\nAI: Generates multi-window burn rate alerts following Google SRE approach", 0.5, 3.0, 4.3, 1.4, GREEN);
addCard(s, "Log Analysis", "Prompt: \"Why are we seeing latency spikes?\"\nAI: Analyzes log patterns, correlates\n     with metric anomalies, suggests causes", 5.2, 3.0, 4.3, 1.4, ORANGE);

// ============= SECTION 5: ALERTING =============
sectionSlide(pres, "SECTION 05", "Alerting Best Practices", "SLO-Based Alerting & Noise Reduction");

// Slide 26: Alerting Philosophy
s = contentSlide(pres, "Alerting Done Right");
addCard(s, "Alert on Symptoms", "User-facing impact, not internal causes\nExample: \"Error rate > 1%\" not \"CPU > 80%\"\nSymptom-based alerts are more actionable", 0.5, 1.1, 4.3, 1.5, RED);
addCard(s, "Reduce Noise", "Every alert should be actionable\nIf you ignore an alert, delete it\nGroup related alerts together\nTune thresholds from real data", 5.2, 1.1, 4.3, 1.5, ORANGE);
addCard(s, "Alert Requirements", "- Severity levels: critical / warning / info\n- Routing: PagerDuty / Slack / email based on severity\n- Runbooks: every alert links to a runbook\n- Regular tuning: review alerts monthly", 0.5, 2.8, 9, 1.6, BLUE);

// Slide 27: SLO-Based Alerting
s = contentSlide(pres, "SLO-Based Alerting");
addCard(s, "Define SLOs", "SLI: actual measurement (e.g., latency < 200ms)\nSLO: target (e.g., 99.9% of requests < 200ms)\nError Budget: 100% - SLO = allowed failures\n30-day budget @ 99.9% = 43.2 min downtime", 0.5, 1.1, 4.3, 1.8, BLUE);
addCard(s, "Burn Rate Alerts", "How fast are you consuming error budget?\nBurn rate 1x = exactly on pace\nBurn rate 10x = will exhaust in 3 days\nMulti-window: 1h/6h for fast, 3d for slow burn", 5.2, 1.1, 4.3, 1.8, PURPLE);
addCard(s, "Google SRE Approach", "- Dramatically reduces alert noise (up to 90%)\n- Focuses on what matters: user impact\n- Aligns engineering with business objectives\n- Error budget drives reliability vs velocity decisions", 0.5, 3.1, 9, 1.3, GREEN);
addTagline(s, "SLO-based alerting is the gold standard - adopt it");

// Slide 28: Alertmanager
s = contentSlide(pres, "Prometheus Alertmanager");
addCard(s, "Features", "- Deduplication of alerts\n- Grouping related alerts\n- Silencing during maintenance\n- Inhibition (suppress if related alert fires)\n- Multi-receiver routing", 0.5, 1.1, 4.3, 2.2, BLUE);
addCard(s, "Routing Example", "route:\n  group_by: [alertname, namespace]\n  receiver: slack-default\n  routes:\n  - match: {severity: critical}\n    receiver: pagerduty\n  - match: {severity: warning}\n    receiver: slack-warning", 5.2, 1.1, 4.3, 2.2, PURPLE);
addTagline(s, "Alertmanager prevents alert fatigue with intelligent routing");

// ============= SECTION 6: HANDS-ON LAB =============
sectionSlide(pres, "SECTION 06", "Hands-on Lab", "Building an Observability Stack (90 min)");

// Slide 30: Lab Overview
s = contentSlide(pres, "Lab: Observability Stack");
addCard(s, "Part 1: Monitoring (30 min)", "1. Helm install kube-prometheus-stack\n2. Add custom metrics (prometheus_client)\n3. Verify scraping targets\n4. Write PromQL queries", 0.5, 1.1, 2.8, 2.0, BLUE);
addCard(s, "Part 2: Dashboards (30 min)", "1. Import K8s overview dashboard\n2. Create custom app dashboard\n3. Build 6+ panels (rate, errors, latency)\n4. AI-generate complex PromQL", 3.6, 1.1, 2.8, 2.0, GREEN);
addCard(s, "Part 3: Logging (30 min)", "1. Deploy Loki + Promtail\n2. Structured logging in Flask\n3. LogQL queries & exploration\n4. Alert rules for SLO violations", 6.7, 1.1, 2.8, 2.0, PURPLE);
addTagline(s, "By the end: a complete monitoring + logging + alerting stack");

// Slide 31: Lab Part 1 - Prometheus Setup
s = contentSlide(pres, "Lab Part 1: Prometheus Setup");
addCard(s, "Install kube-prometheus-stack", "helm repo add prometheus-community \\\n  https://prometheus-community.github.io/helm-charts\nhelm install monitoring prometheus-community/kube-prometheus-stack \\\n  --namespace monitoring --create-namespace", 0.5, 1.1, 9, 1.3, BLUE);
addCard(s, "Add Custom Metrics", "from prometheus_client import Counter, Histogram\nfrom prometheus_client import start_http_server\n\nREQUESTS = Counter('myapp_requests_total',\n  'Total requests', ['method', 'status'])\nstart_http_server(8000)", 0.5, 2.6, 9, 1.6, GREEN);
addTagline(s, "Verify: kubectl port-forward svc/monitoring-prometheus 9090");

// Slide 32: Lab Part 1 - PromQL Practice
s = contentSlide(pres, "Lab Part 1: PromQL Practice");
addCard(s, "Basic Queries", "# Total requests in last 5 min\nrate(myapp_requests_total[5m])\n\n# Error rate percentage\nrate(myapp_requests_total{status=\"500\"}[5m])\n/ rate(myapp_requests_total[5m]) * 100", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Advanced Queries", "# 95th percentile latency\nhistogram_quantile(0.95,\n  rate(myapp_request_duration_bucket[5m]))\n\n# Top 5 endpoints by request rate\ntopk(5, sum by (endpoint)\n  (rate(myapp_requests_total[5m])))", 5.2, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "AI-Assisted PromQL", "Try asking your AI assistant:\n\"Write PromQL for: show me the services with error rate above 1% in the last hour\"", 0.5, 3.3, 9, 1.1, GREEN);

// Slide 33: Lab Part 2 - Grafana Dashboard
s = contentSlide(pres, "Lab Part 2: Grafana Dashboard");
addCard(s, "Required Panels (6+)", "1. Request Rate (rate/sec) - Graph\n2. Error Rate (%) - Stat + threshold colors\n3. Latency p50/p95/p99 - Graph\n4. CPU Usage - Gauge\n5. Memory Usage - Gauge\n6. Active Connections - Stat", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "Dashboard Tips", "Use variables: $namespace, $service\nSet meaningful thresholds\nAdd descriptions to every panel\nUse appropriate visualization types\nExport JSON and commit to Git", 5.2, 1.1, 4.3, 2.3, PURPLE);
addTagline(s, "A good dashboard tells a story at a glance");

// Slide 34: Lab Part 3 - Loki
s = contentSlide(pres, "Lab Part 3: Loki + Structured Logging");
addCard(s, "Deploy Loki", "helm install loki grafana/loki-stack \\\n  --set promtail.enabled=true \\\n  --set loki.persistence.enabled=true \\\n  --namespace monitoring", 0.5, 1.1, 9, 1.1, BLUE);
addCard(s, "Flask Structured Logging", "import logging, json_log_formatter\n\nformatter = json_log_formatter.JSONFormatter()\nhandler = logging.StreamHandler()\nhandler.setFormatter(formatter)\nlogger = logging.getLogger('myapp')\nlogger.addHandler(handler)\nlogger.info('Request processed', extra={'method': 'GET', 'path': '/api', 'duration_ms': 42})", 0.5, 2.4, 9, 1.8, GREEN);
addTagline(s, "Structured logs + Loki = searchable, correlated debugging");

// Slide 35: Lab Part 3 - LogQL & Alerts
s = contentSlide(pres, "Lab Part 3: LogQL & Alert Rules");
addCard(s, "LogQL Queries", '{app="myapp"} |= "error"\n{app="myapp"} | json | level="error" | line_format "{{.msg}}"\nsum(rate({app="myapp"} |= "error" [5m])) by (level)', 0.5, 1.1, 9, 1.2, BLUE);
addCard(s, "Alert Rule Example", "groups:\n- name: slo-alerts\n  rules:\n  - alert: HighErrorRate\n    expr: rate(http_errors_total[5m]) / rate(http_total[5m]) > 0.001\n    for: 5m\n    labels: { severity: critical }\n    annotations:\n      summary: Error rate exceeds SLO (99.9%)", 0.5, 2.5, 9, 2.0, RED);

// ============= SECTION 7: ASSESSMENT & WRAP-UP =============
sectionSlide(pres, "SECTION 07", "Assessment & Next Steps", "Lab Deliverables + Midterm");

// Slide 37: Assessment
s = contentSlide(pres, "Week 7 Assessment");
addCard(s, "Lab Deliverables", "1. Prometheus collecting custom app metrics\n2. Grafana dashboard with 6+ panels\n3. Loki receiving structured logs\n4. At least 2 alert rules configured\n5. Screenshot of complete stack", 0.5, 1.1, 4.3, 2.2, BLUE);
addCard(s, "Midterm Checkpoint", "Due this week: midterm progress report\n\n- CI/CD pipeline running\n- Containerized app deployed\n- Monitoring stack operational\n- Document AI collaboration process", 5.2, 1.1, 4.3, 2.2, PURPLE);
addTagline(s, "Midterm = proof that your DevOps pipeline works end-to-end");

// Slide 38: Tool Summary
s = contentSlide(pres, "Tool Summary");
addCard(s, "Metrics", "Prometheus - collection & storage\nGrafana - visualization\nprometheus_client - Python SDK\nnode_exporter - host metrics", 0.5, 1.1, 4.3, 1.5, BLUE);
addCard(s, "Logging", "Loki - lightweight log aggregation\nPromtail - log collector agent\nFluent Bit - alternative collector\nElasticsearch - full-text (if needed)", 5.2, 1.1, 4.3, 1.5, GREEN);
addCard(s, "Tracing & Alerting", "Jaeger / Tempo - distributed tracing\nOpenTelemetry - instrumentation standard\nAlertmanager - alert routing & dedup\nPagerDuty / Slack - notification channels", 0.5, 2.8, 4.3, 1.5, PURPLE);
addCard(s, "AI Tools", "AI for PromQL generation\nAI for dashboard creation\nAnomaly detection models\nLog pattern analysis", 5.2, 2.8, 4.3, 1.5, ORANGE);

// Slide 39: Recommended Reading
s = contentSlide(pres, "Recommended Reading");
addNumberedItem(s, 1, "Distributed Systems Observability", "Sridharan, C. (2018) - O'Reilly - Free ebook", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "Observability Engineering", "Majors, C. et al. (2022) - O'Reilly - Comprehensive guide", 0.5, 1.75, 9, GREEN, GREEN);
addNumberedItem(s, 3, "Prometheus Documentation", "prometheus.io/docs - Official reference", 0.5, 2.4, 9, PURPLE, PURPLE);
addNumberedItem(s, 4, "Alerting on SLOs", "Google SRE Workbook - sre.google/workbook/alerting-on-slos", 0.5, 3.05, 9, ORANGE, ORANGE);
addTagline(s, "Start with Sridharan's book - it's free and excellent");

// Slide 40: Key Takeaways
s = contentSlide(pres, "Key Takeaways");
addNumberedItem(s, 1, "Three Pillars", "Metrics + Logs + Traces = complete observability", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Start Simple", "Prometheus + Grafana + Loki covers 90% of needs", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Structured Everything", "JSON logs, labeled metrics, traced requests", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Alert Smart", "SLO-based alerting reduces noise by up to 90%", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "AI Assists", "Let AI generate PromQL, dashboards, and alert rules", 0.5, 3.7, 9, CYAN, CYAN);

// Slide 41: Next Week Preview
s = contentSlide(pres, "Next Week: DevSecOps + AI Code Review");
addCard(s, "Week 8 Preview", "- Security scanning in CI/CD pipelines\n- SAST, DAST, SCA tools\n- AI-powered code review\n- Supply chain security\n- Secret management", 0.5, 1.1, 9, 2.0, PURPLE);
addTagline(s, "Security is everyone's job - shift left with DevSecOps");

// Slide 42: Q&A
s = pres.addSlide();
s.background = { color: BG };
s.addShape("oval", { x: 2.5, y: -0.5, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
s.addShape("oval", { x: 6, y: 3.5, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
s.addText("Questions?", { x: 0, y: 1.8, w: 10, h: 0.7, fontSize: 36, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
s.addText("Week 7: Monitoring, Logging & Observability", { x: 1, y: 2.7, w: 8, h: 0.4, fontSize: 14, fontFace: "Arial", color: BLUE, align: "center" });
s.addText("Anirach Mingkhwan | FITM, KMUTNB", { x: 2, y: 3.3, w: 6, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
addTagline(s, '"Observability is not a tool you buy, it\'s a practice you build"');

// SAVE
const outPath = "/home/clawdbot/clawd/tmp/Week07_raw.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Saved ${outPath} (${pres.slides.length} slides)`);
}).catch(err => {
  console.error("Error:", err);
});
