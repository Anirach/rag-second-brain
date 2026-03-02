const pptxgen = require("pptxgenjs");
const BG="0D1229",CARD_BG="1E2341",BLUE="60A5FA",PURPLE="A78BFA",GREEN="4ADE80",ORANGE="FBBF24",CYAN="22D3EE",PINK="F472B6",RED="F87171",GOLD="FBBF24",GRAY="8B95B0",LIGHT="B0B8D0",WHITE="FFFFFF";
function addTagline(s,t,y){y=y||4.95;s.addShape("rect",{x:0.4,y,w:9.2,h:0.4,fill:{color:"141E32"},line:{color:"2A3560",width:0.5},rectRadius:0.05});s.addText(t,{x:0.4,y,w:9.2,h:0.4,fontSize:11,fontFace:"Arial",bold:true,color:GOLD,align:"center",valign:"middle"});}
function addCard(s,title,body,x,y,w,h,ac){s.addShape("rect",{x,y,w,h,fill:{color:CARD_BG},rectRadius:0.08});s.addShape("rect",{x,y,w:0.06,h,fill:{color:ac||BLUE}});s.addText(title,{x:x+0.15,y,w:w-0.2,h:0.35,fontSize:13,fontFace:"Arial",bold:true,color:ac||BLUE,valign:"top",margin:[4,0,0,0]});s.addText(body,{x:x+0.15,y:y+0.32,w:w-0.2,h:h-0.36,fontSize:10,fontFace:"Arial",color:LIGHT,valign:"top",lineSpacingMultiple:1.3});}
function addNumberedItem(s,n,title,desc,x,y,w,cc,tc){s.addShape("rect",{x,y,w,h:0.55,fill:{color:CARD_BG},rectRadius:0.06});s.addShape("oval",{x:x+0.1,y:y+0.1,w:0.35,h:0.35,fill:{color:cc}});s.addText(String(n),{x:x+0.1,y:y+0.1,w:0.35,h:0.35,fontSize:11,fontFace:"Arial",bold:true,color:WHITE,align:"center",valign:"middle"});s.addText(title,{x:x+0.55,y:y+0.05,w:w-0.65,h:0.22,fontSize:11,fontFace:"Arial",bold:true,color:tc||BLUE});s.addText(desc,{x:x+0.55,y:y+0.27,w:w-0.65,h:0.23,fontSize:9,fontFace:"Arial",color:LIGHT});}
function titleSlide(p,t,sub,wk){let s=p.addSlide();s.background={color:BG};s.addShape("oval",{x:2.5,y:-0.5,w:3,h:3,fill:{color:"8B5CF6",transparency:88}});s.addShape("oval",{x:6,y:3.5,w:2.5,h:2.5,fill:{color:"3B82F6",transparency:88}});s.addText("DEVOPS WITH VIBECODING",{x:0,y:1.2,w:10,h:0.35,fontSize:11,fontFace:"Arial",color:"8B5CF6",align:"center",charSpacing:4});s.addText(t,{x:0.5,y:1.7,w:9,h:0.6,fontSize:28,fontFace:"Arial",bold:true,color:WHITE,align:"center"});s.addText(sub,{x:1,y:2.4,w:8,h:0.4,fontSize:16,fontFace:"Arial",color:BLUE,align:"center"});s.addText(wk,{x:3.5,y:3.1,w:3,h:0.35,fontSize:12,fontFace:"Arial",color:GRAY,align:"center"});s.addText("Anirach Mingkhwan",{x:2,y:3.6,w:6,h:0.3,fontSize:11,fontFace:"Arial",color:GRAY,align:"center"});s.addText("FITM, KMUTNB",{x:2,y:3.9,w:6,h:0.3,fontSize:10,fontFace:"Arial",color:GRAY,align:"center"});}
function sectionSlide(p,num,t,sub){let s=p.addSlide();s.background={color:BG};s.addShape("rect",{x:0,y:0,w:10,h:5.63,fill:{color:"111936"}});s.addShape("oval",{x:-1,y:1,w:4,h:4,fill:{color:"8B5CF6",transparency:92}});s.addShape("oval",{x:7,y:-0.5,w:3,h:3,fill:{color:"3B82F6",transparency:92}});s.addText(num,{x:3.5,y:1.5,w:3,h:0.5,fontSize:14,fontFace:"Arial",color:PURPLE,align:"center",charSpacing:3});s.addText(t,{x:1,y:2.1,w:8,h:0.6,fontSize:28,fontFace:"Arial",bold:true,color:WHITE,align:"center"});s.addText(sub||"",{x:1.5,y:2.8,w:7,h:0.4,fontSize:14,fontFace:"Arial",color:BLUE,align:"center"});}
function contentSlide(p,t){let s=p.addSlide();s.background={color:BG};s.addShape("rect",{x:0,y:0,w:10,h:0.9,fill:{color:"111936"}});s.addShape("rect",{x:0,y:0.88,w:10,h:0.03,fill:{color:PURPLE,transparency:50}});s.addText(t,{x:0.5,y:0.15,w:9,h:0.6,fontSize:20,fontFace:"Arial",bold:true,color:WHITE});return s;}

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

// 1: Title
titleSlide(pres, "MLOps & AI Model\nDeployment", "DevOps for Machine Learning", "Week 10");

// 2: Learning Objectives
let s = contentSlide(pres, "Learning Objectives");
addNumberedItem(s, 1, "MLOps Lifecycle", "Understand MLOps and its relationship to DevOps", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Experiment Tracking", "Implement model versioning and reproducible experiments", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Model Serving", "Build containerized model serving infrastructure", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "ML Pipelines", "Automate training pipelines with MLflow, DVC, Kubeflow", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Monitoring & Drift", "Apply model monitoring and drift detection in production", 0.5, 3.7, 9, RED, RED);
addTagline(s, '"ML models degrade silently - MLOps makes degradation visible"');

// 3: Agenda
s = contentSlide(pres, "Today's Agenda");
addCard(s, "Part 1: MLOps Foundations", "What is MLOps?\nML vs Software challenges\nMaturity levels", 0.5, 1.1, 4.3, 1.5, PURPLE);
addCard(s, "Part 2: Tracking & Versioning", "MLflow experiments\nDVC for data versioning\nModel registry", 5.2, 1.1, 4.3, 1.5, BLUE);
addCard(s, "Part 3: Serving", "Deployment patterns\nContainerized serving\nA/B testing models", 0.5, 2.8, 4.3, 1.5, GREEN);
addCard(s, "Part 4: Pipelines & Monitoring", "ML automation\nDrift detection\nHands-on Lab", 5.2, 2.8, 4.3, 1.5, ORANGE);

// === SECTION 1: MLOPS FOUNDATIONS ===
sectionSlide(pres, "SECTION 01", "Introduction to MLOps", "Why ML Needs Its Own Ops");

// 5: ML vs Software
s = contentSlide(pres, "ML Systems vs Traditional Software");
addCard(s, "Traditional Software", "- Code is the artifact\n- Deterministic behavior\n- Version control is mature\n- Testing is well-understood\n- Bugs are reproducible\n- Deploy once, runs forever", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "ML Systems", "- Code + Data + Model = artifact\n- Probabilistic behavior\n- Need to version data too\n- Testing is much harder\n- Bugs may be statistical\n- Models degrade over time (drift)", 5.2, 1.1, 4.3, 2.3, PURPLE);
addTagline(s, "ML has all the challenges of software plus data and model management");

// 6: Hidden Technical Debt
s = contentSlide(pres, "Hidden Technical Debt in ML (Google, 2015)");
addCard(s, "The Iceberg", "ML Code is only a tiny fraction\nof a real ML system!\n\nSurrounding infrastructure:\n- Data collection & validation\n- Feature extraction & stores\n- Configuration management\n- Process management tools\n- Analysis tools & monitoring\n- Serving infrastructure\n- Machine resource management", 0.5, 1.1, 4.3, 3.0, RED);
addCard(s, "Common Debt", "- Glue code (connecting systems)\n- Pipeline jungles (tangled data flows)\n- Dead experimental codepaths\n- Undeclared data dependencies\n- Configuration debt\n- Feedback loops (model output affects input)\n- Reproducibility gaps\n- Monitoring blind spots\n\nMLOps addresses ALL of these", 5.2, 1.1, 4.3, 3.0, ORANGE);

// 7: MLOps Maturity
s = contentSlide(pres, "MLOps Maturity Levels");
addNumberedItem(s, 0, "Level 0: Manual", "Manual training, manual deployment, no monitoring, notebook-driven", 0.5, 1.1, 9, RED, RED);
addNumberedItem(s, 1, "Level 1: Pipeline Automation", "Automated training pipeline, manual deployment, basic monitoring", 0.5, 1.75, 9, ORANGE, ORANGE);
addNumberedItem(s, 2, "Level 2: CI/CD Automation", "Automated training + deployment, A/B testing, experiment tracking", 0.5, 2.4, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Level 3: Full Automation", "Monitoring-triggered retraining, auto-rollback, self-healing", 0.5, 3.05, 9, GREEN, GREEN);
addTagline(s, "Most teams are at Level 0-1. Goal: reach Level 2 minimum.");

// 8: MLOps Lifecycle
s = contentSlide(pres, "MLOps Lifecycle");
addCard(s, "Development", "1. Data collection & labeling\n2. Feature engineering\n3. Experiment tracking\n4. Model training & tuning\n5. Model evaluation\n6. Model selection", 0.5, 1.1, 2.8, 2.3, PURPLE);
addCard(s, "Deployment", "7. Model packaging\n8. Containerization\n9. Serving infrastructure\n10. A/B testing\n11. Canary deployment\n12. Full rollout", 3.6, 1.1, 2.8, 2.3, BLUE);
addCard(s, "Operations", "13. Prediction monitoring\n14. Data drift detection\n15. Performance tracking\n16. Alerting & rollback\n17. Retraining triggers\n18. Continuous improvement", 6.7, 1.1, 2.8, 2.3, GREEN);
addTagline(s, "MLOps is a continuous cycle, not a one-time deployment");

// === SECTION 2: EXPERIMENT TRACKING ===
sectionSlide(pres, "SECTION 02", "Experiment Tracking\n& Model Versioning", "Reproducibility is Non-Negotiable");

// 10: MLflow Overview
s = contentSlide(pres, "MLflow: The MLOps Swiss Army Knife");
addCard(s, "MLflow Tracking", "Log parameters, metrics, artifacts\nCompare runs side-by-side\nSearch & filter experiments\nUI for visualization", 0.5, 1.1, 4.3, 1.5, BLUE);
addCard(s, "MLflow Projects", "Reproducible runs with conda/docker\nGit-backed project definitions\nShared execution environment\nParameter definitions", 5.2, 1.1, 4.3, 1.5, GREEN);
addCard(s, "MLflow Models", "Standard model packaging format\nMultiple flavors (sklearn, pytorch, tf)\nOne-line deployment\nServing via REST API", 0.5, 2.8, 4.3, 1.5, PURPLE);
addCard(s, "MLflow Registry", "Model versioning & stages\nApproval workflows\nStage transitions\nNone -> Staging -> Production -> Archived", 5.2, 2.8, 4.3, 1.5, ORANGE);

// 11: MLflow Tracking Code
s = contentSlide(pres, "MLflow Tracking in Practice");
addCard(s, "Training with Tracking", "import mlflow\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, f1_score\n\nmlflow.set_experiment('customer-churn')\n\nwith mlflow.start_run(run_name='rf-baseline'):\n    # Log parameters\n    mlflow.log_param('n_estimators', 100)\n    mlflow.log_param('max_depth', 10)\n    mlflow.log_param('data_version', 'v2.1')\n    \n    # Train\n    model = RandomForestClassifier(n_estimators=100, max_depth=10)\n    model.fit(X_train, y_train)\n    \n    # Log metrics\n    preds = model.predict(X_test)\n    mlflow.log_metric('accuracy', accuracy_score(y_test, preds))\n    mlflow.log_metric('f1', f1_score(y_test, preds))\n    \n    # Log model\n    mlflow.sklearn.log_model(model, 'model')", 0.5, 1.1, 9, 3.5, BLUE);

// 12: What to Track
s = contentSlide(pres, "What to Track in Every Experiment");
addNumberedItem(s, 1, "Code Version", "Git commit hash - exact code that produced the model", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Data Version", "DVC hash or dataset version - exact data used for training", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Hyperparameters", "All training config - learning rate, epochs, batch size, architecture", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Metrics", "Accuracy, F1, AUC, latency, memory - quantify model quality", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Artifacts & Environment", "Model files, plots, requirements.txt, Dockerfile", 0.5, 3.7, 9, CYAN, CYAN);

// 13: DVC
s = contentSlide(pres, "DVC: Git for Data");
addCard(s, "The Problem", "Git can't handle large files (datasets, models)\nData needs versioning too\nTeam needs to share exact datasets\nReproducibility requires data lineage", 0.5, 1.1, 4.3, 1.7, RED);
addCard(s, "DVC Solution", "# Track a large dataset\ndvc add data/training.csv\ngit add data/training.csv.dvc .gitignore\ngit commit -m 'Add training data v1'\n\n# Push data to remote storage\ndvc remote add s3 s3://my-bucket/dvc\ndvc push\n\n# Reproduce exact dataset\ngit checkout v1.0\ndvc pull", 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "How It Works", "Git stores: .dvc files (hashes, metadata) | DVC stores: actual data in S3/GCS/Azure\nResult: complete version history for code AND data together", 0.5, 3.0, 9, 1.4, BLUE);

// 14: Model Registry
s = contentSlide(pres, "Model Registry Lifecycle");
addNumberedItem(s, 1, "None (Development)", "Model registered, not yet evaluated", 0.5, 1.1, 9, GRAY, GRAY);
addNumberedItem(s, 2, "Staging", "Passed automated tests, ready for manual review", 0.5, 1.75, 9, ORANGE, ORANGE);
addNumberedItem(s, 3, "Production", "Approved, serving live traffic", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Archived", "Replaced by newer version, kept for rollback", 0.5, 3.05, 9, GRAY, GRAY);
addCard(s, "Registry Code", "# Register model\nresult = mlflow.register_model('runs:/abc123/model', 'churn-predictor')\n\n# Transition stage\nclient.transition_model_version_stage('churn-predictor', version=3, stage='Production')", 0.5, 3.7, 9, 0.8, PURPLE);

// === SECTION 3: MODEL SERVING ===
sectionSlide(pres, "SECTION 03", "Model Serving &\nDeployment", "From Training to Production");

// 16: Deployment Patterns
s = contentSlide(pres, "Model Deployment Patterns");
addCard(s, "Batch Prediction", "Process large datasets offline\nScheduled (hourly/daily)\nHigh throughput, high latency OK\nUse case: recommendations,\nrisk scoring, reports", 0.5, 1.1, 4.3, 1.7, BLUE);
addCard(s, "Real-Time (REST/gRPC)", "On-demand predictions via API\nLow latency required (<100ms)\nAuto-scaling for load\nUse case: fraud detection,\nchatbots, search ranking", 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "Streaming", "Process events in real-time\nKafka/Kinesis integration\nContinuous predictions\nUse case: anomaly detection,\nreal-time personalization", 0.5, 3.0, 4.3, 1.4, PURPLE);
addCard(s, "Edge Deployment", "Model runs on device\nNo network dependency\nTF Lite, ONNX Runtime\nUse case: mobile, IoT,\nautonomous vehicles", 5.2, 3.0, 4.3, 1.4, ORANGE);

// 17: Serving Frameworks
s = contentSlide(pres, "Model Serving Frameworks");
addCard(s, "TF Serving", "TensorFlow models\nHigh performance C++\ngRPC + REST API\nModel versioning built-in", 0.5, 1.1, 2.1, 1.7, BLUE);
addCard(s, "TorchServe", "PyTorch models\nMulti-model serving\nMetrics + logging\nModel archiver tool", 2.9, 1.1, 2.1, 1.7, PURPLE);
addCard(s, "Triton", "NVIDIA, multi-framework\nGPU optimized\nDynamic batching\nEnsemble support", 5.3, 1.1, 2.1, 1.7, GREEN);
addCard(s, "BentoML", "Framework-agnostic\nEasy packaging\nBuilt-in API server\nDocker/K8s deploy", 7.4, 1.1, 2.1, 1.7, ORANGE);
addCard(s, "MLflow Models", "# Serve any MLflow model\nmlflow models serve -m models:/churn-predictor/Production -p 5000\n\n# Or build Docker container\nmlflow models build-docker -m models:/churn-predictor/Production -n churn-api", 0.5, 3.0, 9, 1.4, CYAN);

// 18: Containerized Serving
s = contentSlide(pres, "Containerized Model Serving");
addCard(s, "Dockerfile", "FROM python:3.11-slim\n\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\n\nCOPY model/ /app/model/\nCOPY serve.py /app/\n\nWORKDIR /app\nEXPOSE 8000\nCMD [\"uvicorn\", \"serve:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]", 0.5, 1.1, 4.3, 2.8, BLUE);
addCard(s, "FastAPI Serving", "from fastapi import FastAPI\nimport mlflow\nimport numpy as np\n\napp = FastAPI()\nmodel = mlflow.pyfunc.load_model('model/')\n\n@app.post('/predict')\nasync def predict(features: dict):\n    X = np.array([features['data']])\n    prediction = model.predict(X)\n    return {\n        'prediction': prediction.tolist(),\n        'model_version': '3',\n        'timestamp': datetime.now().isoformat()\n    }", 5.2, 1.1, 4.3, 2.8, GREEN);

// 19: A/B Testing Models
s = contentSlide(pres, "A/B Testing for Models");
addCard(s, "Why A/B Test Models?", "- Offline metrics don't always match online\n- Compare old vs new model on real traffic\n- Measure business impact (not just accuracy)\n- Gradual rollout reduces risk\n- Data-driven deployment decisions", 0.5, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Implementation", "Traffic splitting approaches:\n- Load balancer routing (Istio, NGINX)\n- Application-level (feature flags)\n- Shadow mode (log predictions, no serving)\n\nMetrics to compare:\n- Prediction accuracy on live data\n- Latency & throughput\n- Business KPIs (conversion, engagement)\n- User satisfaction signals", 5.2, 1.1, 4.3, 2.0, BLUE);
addTagline(s, "Never deploy a new model without A/B testing on real traffic first");

// 20: K8s Deployment
s = contentSlide(pres, "Model Deployment on Kubernetes");
addCard(s, "K8s Benefits for ML", "- HPA for auto-scaling\n- GPU scheduling (NVIDIA plugin)\n- Rolling updates for model versions\n- Resource limits per model\n- Health checks & self-healing\n- Multi-model serving", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "Deployment Manifest", "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: churn-model\nspec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n      - name: model\n        image: churn-api:v3\n        resources:\n          limits:\n            nvidia.com/gpu: 1\n            memory: 4Gi\n        ports:\n        - containerPort: 8000", 5.2, 1.1, 4.3, 2.3, GREEN);
addTagline(s, "K8s handles scaling and reliability so you can focus on models");

// === SECTION 4: PIPELINES & MONITORING ===
sectionSlide(pres, "SECTION 04", "ML Pipelines &\nModel Monitoring", "Automation & Drift Detection");

// 22: ML Pipeline
s = contentSlide(pres, "Automated ML Pipeline");
addNumberedItem(s, 1, "Data Validation", "Great Expectations checks schema, distributions, quality", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Feature Engineering", "Transform raw data, use feature store (Feast) for consistency", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Hyperparameter Optimization", "Optuna/Ray Tune for automated search", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Model Evaluation", "Compare against baseline, check for bias, validate metrics", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Conditional Deployment", "Only deploy if metrics improve over current production model", 0.5, 3.7, 9, CYAN, CYAN);

// 23: Feature Store
s = contentSlide(pres, "Feature Stores");
addCard(s, "The Problem: Training-Serving Skew", "Training uses batch-computed features\nServing computes features on-the-fly\nDifferent code paths = different results\nModel accuracy drops silently in production", 0.5, 1.1, 4.3, 1.7, RED);
addCard(s, "Feature Store Solution", "Single source of truth for features\nSame features for training AND serving\nTools: Feast (open-source), Tecton\n\nFeast Example:\nfeast apply  # register features\nfeast materialize  # load to online store\n\n# In serving code:\nfeatures = store.get_online_features(\n  entity_rows=[{'user_id': 123}]\n).to_dict()", 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "Key Benefit", "Eliminates the #1 cause of ML bugs in production: training-serving skew\nSame feature definitions, same computations, everywhere", 0.5, 3.0, 9, 1.4, BLUE);

// 24: Pipeline Orchestrators
s = contentSlide(pres, "ML Pipeline Orchestrators");
addCard(s, "Kubeflow Pipelines", "Kubernetes-native ML workflows\nComponent-based DAGs\nExperiment tracking built-in\nGPU scheduling", 0.5, 1.1, 4.3, 1.5, BLUE);
addCard(s, "Airflow / Prefect", "General-purpose DAG orchestration\nRich scheduling & monitoring\nLarge plugin ecosystem\nPrefect: modern, Python-native", 5.2, 1.1, 4.3, 1.5, GREEN);
addCard(s, "GitHub Actions for ML", "- name: Retrain on data change\n  on:\n    push: { paths: ['data/**'] }\n  jobs:\n    train:\n      steps:\n      - run: python train.py\n      - run: mlflow models build-docker\n      - run: kubectl set image deployment/model model=new-image", 0.5, 2.8, 9, 1.6, PURPLE);

// 25: Model Monitoring
s = contentSlide(pres, "Model Monitoring in Production");
addCard(s, "What to Monitor", "- Prediction quality (if labels available)\n- Prediction distribution shifts\n- Input data drift\n- Feature value distributions\n- Latency & throughput\n- Error rates\n- Resource utilization (GPU/CPU/memory)", 0.5, 1.1, 4.3, 2.5, BLUE);
addCard(s, "Types of Drift", "Data Drift: input distribution changes\n  (new user demographics, seasonal shifts)\n\nConcept Drift: relationship between\n  input and output changes\n  (user behavior evolves, market shifts)\n\nBoth cause silent model degradation!\nOnly monitoring can detect them.", 5.2, 1.1, 4.3, 2.5, RED);
addTagline(s, "A model without monitoring is a ticking time bomb");

// 26: Drift Detection
s = contentSlide(pres, "Drift Detection Methods");
addCard(s, "Statistical Tests", "KS Test: compare distributions\nPSI (Population Stability Index):\n  <0.1 stable, 0.1-0.25 moderate, >0.25 significant\nJS Divergence: symmetric KL divergence\nChi-squared: categorical features", 0.5, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Evidently AI", "from evidently.report import Report\nfrom evidently.metric_preset import DataDriftPreset\n\nreport = Report(metrics=[DataDriftPreset()])\nreport.run(\n    reference_data=training_df,\n    current_data=production_df\n)\nreport.save_html('drift_report.html')\n\n# Visualize drift per feature\n# Set up alerts when PSI > threshold", 5.2, 1.1, 4.3, 2.0, GREEN);
addCard(s, "Response to Drift", "Alert -> Investigate -> Retrain (if data drift) or Rollback (if concept drift)\nAdjust thresholds -> Monitor -> Continuous improvement", 0.5, 3.3, 9, 1.2, ORANGE);

// 27: Grafana ML Dashboard
s = contentSlide(pres, "Model Monitoring Dashboard (Grafana)");
addCard(s, "Essential Panels", "1. Prediction volume over time\n2. Prediction distribution (histogram)\n3. Latency p50/p95/p99\n4. Error rate by endpoint\n5. Feature drift scores (PSI per feature)\n6. Model version currently serving\n7. Data quality metrics\n8. Resource utilization (GPU if applicable)", 0.5, 1.1, 4.3, 2.8, BLUE);
addCard(s, "Alert Rules", "Prediction drift: PSI > 0.25 for any feature\nLatency spike: p99 > 500ms for 5 minutes\nError rate: > 1% for 5 minutes\nAccuracy drop: below baseline - 5%\nData quality: null rate > threshold\n\nRoute:\n- Critical: PagerDuty\n- Warning: Slack\n- Info: Dashboard only", 5.2, 1.1, 4.3, 2.8, ORANGE);

// === SECTION 5: LAB ===
sectionSlide(pres, "SECTION 05", "Hands-on Lab", "Building an MLOps Pipeline (90 min)");

// 29: Lab Overview
s = contentSlide(pres, "Lab: MLOps Pipeline");
addCard(s, "Part 1: Experiment Tracking (30 min)", "1. Start MLflow tracking server\n2. Train scikit-learn classifier\n3. Log params, metrics, artifacts\n4. Compare 5+ runs in UI\n5. Select best model", 0.5, 1.1, 2.8, 2.0, BLUE);
addCard(s, "Part 2: Model Deployment (30 min)", "1. Register best model in registry\n2. Build Docker container\n3. Deploy to Kubernetes\n4. Test prediction API\n5. Set up health checks", 3.6, 1.1, 2.8, 2.0, GREEN);
addCard(s, "Part 3: Monitoring (30 min)", "1. Implement prediction logging\n2. Evidently drift monitoring\n3. Grafana model dashboard\n4. GitHub Actions retraining\n5. Document pipeline", 6.7, 1.1, 2.8, 2.0, PURPLE);
addTagline(s, "By the end: a complete MLOps pipeline from experiment to monitoring");

// 30: Lab Part 1
s = contentSlide(pres, "Lab Part 1: MLflow Experiment Tracking");
addCard(s, "Setup & Training", "# Start MLflow server\nmlflow server --host 0.0.0.0 --port 5000\n\n# Train with different hyperparameters\nfor n_est in [50, 100, 200]:\n    for depth in [5, 10, 20]:\n        with mlflow.start_run():\n            mlflow.log_param('n_estimators', n_est)\n            mlflow.log_param('max_depth', depth)\n            model = RandomForestClassifier(\n                n_estimators=n_est, max_depth=depth)\n            model.fit(X_train, y_train)\n            acc = accuracy_score(y_test, model.predict(X_test))\n            mlflow.log_metric('accuracy', acc)\n            mlflow.sklearn.log_model(model, 'model')", 0.5, 1.1, 9, 3.3, BLUE);

// 31: Lab Part 2
s = contentSlide(pres, "Lab Part 2: Model Deployment");
addCard(s, "Register & Container", "# Register best model\nmlflow.register_model(\n    'runs:/best_run_id/model',\n    'customer-churn'\n)\n\n# Build Docker image\nmlflow models build-docker \\\n    -m models:/customer-churn/Production \\\n    -n churn-api:v1\n\n# Test locally\ndocker run -p 5001:8080 churn-api:v1", 0.5, 1.1, 4.3, 3.0, GREEN);
addCard(s, "Deploy to K8s", "# Deploy\nkubectl apply -f model-deployment.yaml\n\n# Test prediction API\ncurl -X POST http://model-svc:8080/invocations \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"inputs\": [[25, 1, 50000, 3]]}'\n\n# Response:\n{\"predictions\": [0]}\n\n# Set up HPA\nkubectl autoscale deployment churn-model \\\n  --min=2 --max=10 --cpu-percent=70", 5.2, 1.1, 4.3, 3.0, PURPLE);

// 32: Lab Part 3
s = contentSlide(pres, "Lab Part 3: Monitoring & Automation");
addCard(s, "Drift Monitoring", "from evidently.report import Report\nfrom evidently.metric_preset import DataDriftPreset\n\n# Compare training vs production data\nreport = Report(metrics=[DataDriftPreset()])\nreport.run(\n    reference_data=train_df,\n    current_data=prod_df\n)\nreport.save_html('drift.html')\n\n# Check drift score\nif report.as_dict()['metrics'][0]\\\n    ['result']['dataset_drift']:\n    trigger_retraining()", 0.5, 1.1, 4.3, 2.8, ORANGE);
addCard(s, "Retraining Pipeline", "# .github/workflows/retrain.yml\non:\n  schedule:\n    - cron: '0 2 * * 1'  # Weekly\n  workflow_dispatch:  # Manual trigger\n\njobs:\n  retrain:\n    steps:\n    - run: python train.py\n    - run: python evaluate.py\n    - run: |\n        if [ $IMPROVED = true ]; then\n          mlflow models build-docker ...\n          kubectl set image ...\n        fi", 5.2, 1.1, 4.3, 2.8, GREEN);

// === SECTION 6: WRAP-UP ===
sectionSlide(pres, "SECTION 06", "Assessment & Next Steps", "Quiz 2 + Key Takeaways");

// 34: Assessment
s = contentSlide(pres, "Week 10 Assessment");
addCard(s, "Lab Deliverables", "1. MLflow with 5+ tracked runs\n2. Registered model in registry\n3. Containerized model serving\n4. Monitoring dashboard (Grafana)\n5. Retraining pipeline (GitHub Actions)", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Quiz 2 (Weeks 6-10)", "Covers:\n- IaC & Terraform (Week 6)\n- Monitoring & Observability (Week 7)\n- DevSecOps (Week 8)\n- Testing with AI (Week 9)\n- MLOps (Week 10)\n\nFormat: MCQ + short answer + scenario", 5.2, 1.1, 4.3, 2.0, PURPLE);
addTagline(s, "Midterm checkpoint: demonstrate your complete DevOps pipeline");

// 35: Tool Summary
s = contentSlide(pres, "MLOps Tool Landscape");
addCard(s, "Tracking & Versioning", "MLflow - experiment tracking\nDVC - data versioning\nWeights & Biases - cloud tracking\nNeptune - team collaboration", 0.5, 1.1, 4.3, 1.5, BLUE);
addCard(s, "Serving & Deploy", "BentoML - easy packaging\nTriton - NVIDIA GPU optimized\nSeldon Core - K8s native\nMLflow Models - framework agnostic", 5.2, 1.1, 4.3, 1.5, GREEN);
addCard(s, "Pipelines", "Kubeflow - K8s ML workflows\nAirflow/Prefect - orchestration\nGitHub Actions - CI/CD\nVertex AI - Google Cloud", 0.5, 2.8, 4.3, 1.5, PURPLE);
addCard(s, "Monitoring", "Evidently AI - drift detection\nGrafana - dashboards\nPrometheus - metrics\nWhylogs - data profiling", 5.2, 2.8, 4.3, 1.5, ORANGE);

// 36: Recommended Reading
s = contentSlide(pres, "Recommended Reading");
addNumberedItem(s, 1, "Designing ML Systems", "Huyen, C. (2022) O'Reilly - The definitive MLOps book", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "MLflow Documentation", "mlflow.org/docs - Official reference and tutorials", 0.5, 1.75, 9, GREEN, GREEN);
addNumberedItem(s, 3, "Hidden Technical Debt in ML", "Google (2015) NeurIPS - Classic paper on ML system challenges", 0.5, 2.4, 9, PURPLE, PURPLE);
addNumberedItem(s, 4, "MLOps: A Taxonomy", "Kreuzberger et al. (2023) IEEE Access - Comprehensive survey", 0.5, 3.05, 9, ORANGE, ORANGE);
addTagline(s, "Chip Huyen's book is essential reading for any ML engineer");

// 37: Key Takeaways
s = contentSlide(pres, "Key Takeaways");
addNumberedItem(s, 1, "Track Everything", "Code + data + params + metrics = reproducible ML", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "Version Data Too", "DVC makes data versioning as easy as git", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Containerize Models", "Docker + K8s = scalable, reliable serving", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Monitor for Drift", "Models degrade silently - Evidently + Grafana catch it", 0.5, 3.05, 9, RED, RED);
addNumberedItem(s, 5, "Automate Retraining", "Pipeline automation closes the ML feedback loop", 0.5, 3.7, 9, ORANGE, ORANGE);

// 38: Next Week Preview
s = contentSlide(pres, "Next Week: Multi-Agent Coding");
addCard(s, "Week 11 Preview", "- Multi-agent AI coding systems\n- Agent collaboration patterns\n- Task decomposition strategies\n- Code generation at scale\n- Human oversight frameworks\n- Building with CrewAI / AutoGen", 0.5, 1.1, 9, 2.0, PURPLE);
addTagline(s, "From single AI assistant to teams of specialized coding agents");

// 39: Q&A
s = pres.addSlide();
s.background = { color: BG };
s.addShape("oval", { x: 2.5, y: -0.5, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
s.addShape("oval", { x: 6, y: 3.5, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
s.addText("Questions?", { x: 0, y: 1.8, w: 10, h: 0.7, fontSize: 36, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
s.addText("Week 10: MLOps & AI Model Deployment", { x: 1, y: 2.7, w: 8, h: 0.4, fontSize: 14, fontFace: "Arial", color: BLUE, align: "center" });
s.addText("Anirach Mingkhwan | FITM, KMUTNB", { x: 2, y: 3.3, w: 6, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
addTagline(s, '"The best model is the one you can deploy, monitor, and retrain"');

const outPath = "/home/clawdbot/clawd/tmp/Week10_raw.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Saved ${outPath} (${pres.slides.length} slides)`);
}).catch(err => console.error("Error:", err));
