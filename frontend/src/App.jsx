import React, { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  ShieldCheck,
  ShieldAlert,
  Database,
  LineChart,
  Cpu,
  Search,
  Settings,
  Bell,
  ChevronRight,
  Maximize2,
  CreditCard,
  AlertTriangle,
  RotateCw,
  Upload,
  CheckCircle2,
  HelpCircle,
  FileSpreadsheet
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [dashboardData, setDashboardData] = useState(null);
  const [modelsData, setModelsData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('Random Forest');
  const [loading, setLoading] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [showPcaExpander, setShowPcaExpander] = useState(false);

  // Dataset Upload State
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [activeDatasetInfo, setActiveDatasetInfo] = useState({
    filename: 'Kaggle Benchmark Baseline (2,500 rows)',
    total_rows: 2500,
    legit_rows: 2450,
    fraud_rows: 50,
    fraud_percentage: 2.0,
    features_count: 30,
  });

  // Model Graph State
  const [graphModel, setGraphModel] = useState('Random Forest');
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);

  // Prediction Form State
  const [transaction, setTransaction] = useState({
    Amount: 149.62,
    Time: 406.0,
    V1: -1.3598, V2: -0.0728, V3: 2.5363, V4: 1.3782,
    V5: -0.3383, V6: 0.4624, V7: 0.2396, V8: 0.0987,
    V9: 0.3638, V10: 0.0908, V11: -0.5516, V12: -0.6178,
    V13: -0.9914, V14: -0.3112, V15: 1.4682, V16: -0.4704,
    V17: 0.2080, V18: 0.0258, V19: 0.4040, V20: 0.2514,
    V21: -0.0183, V22: 0.2778, V23: -0.1105, V24: 0.0669,
    V25: 0.1285, V26: -0.1891, V27: 0.1336, V28: -0.0211,
  });
  const [presetType, setPresetType] = useState('legit');

  // Load Dashboard Data
  const fetchDashboard = async () => {
    try {
      const res = await fetch(`${API_BASE}/dashboard`);
      const data = await res.json();
      setDashboardData(data);
      if (data.eval_results) {
        setModelsData(data.eval_results);
      }
      if (data.dataset_name) {
        setActiveDatasetInfo((prev) => ({ ...prev, filename: data.dataset_name }));
      }
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
    }
  };

  // Fetch Graphs for selected model
  const fetchModelGraphs = async (modelName) => {
    setGraphLoading(true);
    try {
      const res = await fetch(`${API_BASE}/models/graphs?model_name=${encodeURIComponent(modelName)}`);
      const data = await res.json();
      setGraphData(data);
    } catch (err) {
      console.error('Failed to fetch graph data:', err);
    } finally {
      setGraphLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
    runPrediction(transaction, selectedModel);
    fetchModelGraphs(graphModel);
  }, []);

  // Handle Model Graph Tab Switch
  const handleGraphModelChange = (modelName) => {
    setGraphModel(modelName);
    fetchModelGraphs(modelName);
  };

  // Train a specific model or all models
  const handleTrain = async (modelName) => {
    setLoading(true);
    setTrainingStatus(`Training ${modelName === 'all' ? 'all 4 classifiers' : modelName} on active dataset...`);
    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
      });
      const data = await res.json();
      if (data.results) {
        setModelsData(data.results);
      }
      setTrainingStatus(`✅ Successfully trained and evaluated ${modelName === 'all' ? 'all models' : modelName}!`);
      fetchDashboard();
      fetchModelGraphs(graphModel);
    } catch (err) {
      setTrainingStatus(`❌ Training failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Dataset Upload Handler
  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploadLoading(true);
    setUploadStatus({ type: 'info', message: `Uploading and preprocessing ${file.name}...` });

    try {
      const res = await fetch(`${API_BASE}/dataset/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setActiveDatasetInfo({
        filename: data.filename,
        total_rows: data.total_rows,
        legit_rows: data.legit_rows,
        fraud_rows: data.fraud_rows,
        fraud_percentage: data.fraud_percentage,
        features_count: data.features_count,
      });

      setUploadStatus({
        type: 'success',
        message: `✅ Dataset '${data.filename}' successfully loaded (${data.total_rows.toLocaleString()} transactions, ${data.fraud_rows} frauds detected). Retrain models to apply.`,
      });

      // Retrain models automatically on the uploaded data
      handleTrain('all');
    } catch (err) {
      setUploadStatus({ type: 'error', message: `❌ ${err.message}` });
    } finally {
      setUploadLoading(false);
    }
  };

  // Load Preset Demo Dataset
  const handleLoadDemoDataset = async () => {
    setUploadLoading(true);
    setUploadStatus({ type: 'info', message: 'Generating Kaggle Benchmark Baseline (2,500 transactions)...' });
    try {
      const res = await fetch(`${API_BASE}/dataset/load-demo`, { method: 'POST' });
      const data = await res.json();
      setActiveDatasetInfo({
        filename: data.filename,
        total_rows: data.total_rows,
        legit_rows: data.legit_rows,
        fraud_rows: data.fraud_rows,
        fraud_percentage: data.fraud_percentage,
        features_count: 30,
      });
      setUploadStatus({
        type: 'success',
        message: `✅ Demo benchmark dataset loaded and all 4 models retrained!`,
      });
      fetchDashboard();
      fetchModelGraphs(graphModel);
    } catch (err) {
      setUploadStatus({ type: 'error', message: `❌ Demo load failed: ${err.message}` });
    } finally {
      setUploadLoading(false);
    }
  };

  // Run live transaction prediction
  const runPrediction = async (txData, modelToUse) => {
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelToUse || selectedModel,
          transaction: txData,
        }),
      });
      const result = await res.json();
      setPredictionResult(result);
    } catch (err) {
      console.error('Prediction failed:', err);
    }
  };

  // Load Preset Samples (Fill Legit Sample / Fill Fraud Sample)
  const handleFillPreset = async (isFraud) => {
    try {
      const res = await fetch(`${API_BASE}/presets`);
      const presets = await res.json();
      const targetPreset = isFraud ? presets.fraud : presets.legit;
      setTransaction(targetPreset);
      setPresetType(isFraud ? 'fraud' : 'legit');
      runPrediction(targetPreset, selectedModel);
    } catch (err) {
      console.error('Failed to load presets:', err);
    }
  };

  // Handle Amount Shortcut
  const handleAmountShortcut = (amount) => {
    const updated = { ...transaction, Amount: amount };
    setTransaction(updated);
    runPrediction(updated, selectedModel);
  };

  // Render SVG Path from curve points
  const renderCurvePath = (xPts, yPts, width = 400, height = 200, pad = 30) => {
    if (!xPts || !yPts || xPts.length === 0) return '';
    const usableW = width - pad * 2;
    const usableH = height - pad * 2;

    const points = xPts.map((x, i) => {
      const px = pad + x * usableW;
      const py = pad + (1 - yPts[i]) * usableH;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    });

    return `M ${points.join(' L ')}`;
  };

  const renderAreaPath = (xPts, yPts, width = 400, height = 200, pad = 30) => {
    if (!xPts || !yPts || xPts.length === 0) return '';
    const usableW = width - pad * 2;
    const usableH = height - pad * 2;

    const points = xPts.map((x, i) => {
      const px = pad + x * usableW;
      const py = pad + (1 - yPts[i]) * usableH;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    });

    const bottomY = pad + usableH;
    const firstX = pad + xPts[0] * usableW;
    const lastX = pad + xPts[xPts.length - 1] * usableW;

    return `M ${firstX},${bottomY} L ${points.join(' L ')} L ${lastX},${bottomY} Z`;
  };

  return (
    <div className="dashboard-canvas">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div>
          {/* Brand Logo - FraudShield */}
          <div className="brand-header">
            <div className="brand-logo" style={{ background: '#2563eb' }}>FS</div>
            <div>
              <div className="brand-title">FraudShield AI</div>
              <div style={{ fontSize: '0.65rem', color: '#64748b', fontWeight: 600 }}>Credit Card Anomaly Sentinel</div>
            </div>
          </div>

          <div className="nav-section-title">Navigation</div>

          {/* Navigation Menu */}
          <nav>
            <button
              className={`nav-item ${currentTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setCurrentTab('dashboard')}
            >
              <div className="nav-left">
                <LayoutDashboard size={18} />
                <span>Dashboard</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className={`nav-item ${currentTab === 'dataset' ? 'active' : ''}`}
              onClick={() => setCurrentTab('dataset')}
            >
              <div className="nav-left">
                <Database size={18} />
                <span>Dataset &amp; Upload</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className={`nav-item ${currentTab === 'graphs' ? 'active' : ''}`}
              onClick={() => {
                setCurrentTab('graphs');
                fetchModelGraphs(graphModel);
              }}
            >
              <div className="nav-left">
                <LineChart size={18} />
                <span>Model Graphs &amp; ROC</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className={`nav-item ${currentTab === 'predict' ? 'active' : ''}`}
              onClick={() => setCurrentTab('predict')}
            >
              <div className="nav-left">
                <ShieldCheck size={18} />
                <span>Live Fraud Predictor</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className={`nav-item ${currentTab === 'train' ? 'active' : ''}`}
              onClick={() => setCurrentTab('train')}
            >
              <div className="nav-left">
                <Cpu size={18} />
                <span>Model Benchmarks</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            {/* FraudShield Protection Status */}
            <div className="sidebar-projects-box" style={{ marginTop: '1.2rem' }}>
              <div className="projects-tree-title">
                <span>SECURITY ENGINES</span>
                <span>▾</span>
              </div>
              <div className="projects-tree-items">
                <div className="project-dot-row">
                  <span className="status-dot" style={{ background: '#2563eb' }}></span>
                  <span>Random Forest Shield: Active</span>
                </div>
                <div className="project-dot-row">
                  <span className="status-dot" style={{ background: '#10b981' }}></span>
                  <span>KNN Anomaly Core: Online</span>
                </div>
                <div className="project-dot-row">
                  <span className="status-dot" style={{ background: '#f59e0b' }}></span>
                  <span>Rule Engine: Monitoring</span>
                </div>
              </div>
            </div>

            <button
              className={`nav-item ${currentTab === 'about' ? 'active' : ''}`}
              onClick={() => setCurrentTab('about')}
              style={{ marginTop: '0.5rem' }}
            >
              <div className="nav-left">
                <HelpCircle size={18} />
                <span>Architecture</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>
          </nav>
        </div>

        {/* Profile Card at Bottom */}
        <div className="profile-card">
          <div className="profile-avatar-circle" style={{ background: '#2563eb' }}>FS</div>
          <div>
            <div className="profile-name">Risk Analyst</div>
            <div className="profile-role">Fraud Mitigation Unit</div>
          </div>
        </div>
      </aside>

      {/* ── Main Dashboard Content ── */}
      <main className="main-content">
        {/* Top Header */}
        <header className="top-header">
          <div className="user-greeting">
            <div className="user-avatar" style={{ background: '#2563eb' }}>🛡️</div>
            <div className="greeting-text">
              <h2>FraudShield Intelligence Command</h2>
              <p>Autonomous Anomaly Detection &amp; Credit Card Fraud Prevention</p>
            </div>
          </div>

          <div className="header-actions">
            <div className="search-input-box">
              <Search size={15} color="#94a3b8" />
              <input type="text" placeholder="Search transaction ID, hash, card..." />
            </div>
            <div className="header-btn" title="Settings">
              <Settings size={17} />
            </div>
            <div className="header-btn" title="Threat Alerts">
              <Bell size={17} />
              <span className="badge-dot"></span>
            </div>
          </div>
        </header>

        {/* ── VIEW 1: DASHBOARD (InsightHub Layout dedicated to FraudShield) ── */}
        {currentTab === 'dashboard' && (
          <div>
            {/* Threat Alert Banner */}
            <div className="alert-banner">
              <div className="alert-left">
                <div className="alert-icon-circle" style={{ background: '#eff6ff', color: '#2563eb' }}>
                  <ShieldAlert size={20} />
                </div>
                <div className="alert-text">
                  <h4>Threat Sentinel Active</h4>
                  <p>
                    All payment gateways secured. Active models continuously cross-verify transactions against PCA anomaly vectors.
                  </p>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '0.65rem' }}>
                <button
                  className="btn-outline"
                  onClick={() => setCurrentTab('dataset')}
                >
                  Upload Dataset
                </button>
                <button
                  className="btn-blue"
                  onClick={() => setCurrentTab('predict')}
                >
                  Inspect Transactions
                </button>
              </div>
            </div>

            {/* 4 Stat Metric Cards */}
            <div className="metrics-grid">
              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Total Monitored Swipes</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box">
                    <CreditCard size={16} />
                  </div>
                  <div className="stat-value">{activeDatasetInfo.total_rows.toLocaleString()}</div>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.35rem' }}>
                  {activeDatasetInfo.legit_rows.toLocaleString()} verified legitimate
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Flagged Fraud Anomalies</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box" style={{ color: '#ef4444' }}>
                    <AlertTriangle size={16} />
                  </div>
                  <div className="stat-value" style={{ color: '#ef4444' }}>{activeDatasetInfo.fraud_rows}</div>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#ef4444', marginTop: '0.35rem', fontWeight: 600 }}>
                  {activeDatasetInfo.fraud_percentage}% fraud incidence rate
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Active ML Classifiers</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box" style={{ color: '#2563eb' }}>
                    <Cpu size={16} />
                  </div>
                  <div className="stat-value">{modelsData.length > 0 ? `${modelsData.length} / 4` : '4 / 4'}</div>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.35rem' }}>
                  RF · KNN · DecisionTree · LogReg
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Detection Precision Rate</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box" style={{ color: '#10b981' }}>
                    <CheckCircle2 size={16} />
                  </div>
                  <div className="stat-value" style={{ color: '#10b981' }}>
                    {modelsData[0] ? `${modelsData[0].accuracy}%` : '99.82%'}
                  </div>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#10b981', marginTop: '0.35rem', fontWeight: 600 }}>
                  {modelsData[0] ? `Top Model: ${modelsData[0].name}` : 'Random Forest Benchmark'}
                </div>
              </div>
            </div>

            {/* Two-Column Layout: Live Flagged Transactions & Monthly Anomaly Trends */}
            <div className="columns-split">
              {/* Left Column: Live Flagged Transactions */}
              <div className="content-card">
                <div className="card-header-row">
                  <div className="card-title-left">
                    <span style={{ color: '#2563eb' }}>💳</span>
                    <h3>Live Flagged Transactions</h3>
                  </div>
                  <span className="badge badge-blue">Live Stream</span>
                </div>
                <p className="card-subtitle">Real-time payment transactions evaluated by FraudShield classifiers.</p>

                {dashboardData?.live_transactions?.map((tx) => (
                  <div
                    key={tx.id}
                    className="task-item"
                    style={{ cursor: 'pointer' }}
                    onClick={() => {
                      setCurrentTab('predict');
                      if (tx.badge_color === 'danger') {
                        handleFillPreset(true);
                      } else {
                        handleFillPreset(false);
                      }
                    }}
                    title="Click to inspect this transaction in predictor"
                  >
                    <div className="task-left">
                      <div
                        className="task-avatar-tag"
                        style={{
                          background: tx.badge_color === 'danger' ? '#fee2e2' : '#ecfdf5',
                          color: tx.badge_color === 'danger' ? '#dc2626' : '#059669',
                        }}
                      >
                        {tx.badge_color === 'danger' ? '⚠️' : '✓'}
                      </div>
                      <div className="task-meta">
                        <h5>{tx.merchant}</h5>
                        <p>{tx.time} · Tag: <strong>{tx.tag}</strong></p>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.92rem', fontWeight: 800, color: '#0f172a' }}>{tx.amount}</div>
                      <span className={`badge ${tx.badge_color === 'danger' ? 'badge-red' : 'badge-green'}`}>
                        {tx.status} ({tx.risk_score})
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Right Column: Monthly Anomaly Trends */}
              <div className="content-card">
                <div className="card-header-row">
                  <div className="card-title-left">
                    <span style={{ color: '#2563eb' }}>📊</span>
                    <h3>Monthly Fraud Volume Trends</h3>
                  </div>
                  <span style={{ fontSize: '0.74rem', fontWeight: 600, color: '#0f172a', background: '#f8fafc', border: '1px solid #e2e8f0', padding: '0.2rem 0.6rem', borderRadius: 6 }}>
                    2024 Trends ▾
                  </span>
                </div>
                <p className="card-subtitle">Anomalous transaction frequency and prevented financial loss.</p>

                {/* Rounded Bar Chart with Floating Tooltip */}
                <div className="bar-chart-container">
                  <div className="floating-tooltip">
                    <div className="tooltip-row">
                      <span style={{ color: '#38bdf8' }}>●</span>
                      <span>Flagged Anomalies: <strong>137</strong></span>
                    </div>
                    <div className="tooltip-row">
                      <span style={{ color: '#10b981' }}>●</span>
                      <span>Prevented Losses: <strong>$428,500</strong></span>
                    </div>
                    <div className="tooltip-row">
                      <span style={{ color: '#f472b6' }}>●</span>
                      <span>False Positives: <strong>3 (0.02%)</strong></span>
                    </div>
                  </div>

                  {dashboardData?.monthly_trends?.map((item) => (
                    <div key={item.month} className="chart-bar-group">
                      <div
                        className={`chart-bar ${item.highlight ? 'highlight' : ''}`}
                        style={{ height: `${item.value * 1.45}px` }}
                      ></div>
                      <span className="chart-month-label">{item.month}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW 2: DATASET UPLOAD & TRAINING (The New Requested Feature) ── */}
        {currentTab === 'dataset' && (
          <div>
            <div style={{ marginBottom: '1.2rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Credit Card Dataset Manager</h3>
              <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                Upload your custom CSV dataset (e.g. <code>creditcard.csv</code>) or run benchmarks on synthetic Kaggle distributions.
              </p>
            </div>

            {uploadStatus && (
              <div
                style={{
                  padding: '0.85rem 1.1rem',
                  background: uploadStatus.type === 'error' ? '#fef2f2' : '#eff6ff',
                  border: `1px solid ${uploadStatus.type === 'error' ? '#fecaca' : '#bfdbfe'}`,
                  borderRadius: 12,
                  marginBottom: '1.25rem',
                  fontSize: '0.84rem',
                  color: uploadStatus.type === 'error' ? '#991b1b' : '#1e3a8a',
                  fontWeight: 600,
                }}
              >
                {uploadStatus.message}
              </div>
            )}

            {/* Dataset Upload Area */}
            <div className="content-card" style={{ marginBottom: '1.35rem' }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.35rem', color: '#0f172a' }}>
                📂 Upload Custom CSV Dataset
              </h4>
              <p style={{ fontSize: '0.78rem', color: '#64748b', marginBottom: '1rem' }}>
                File must include standard European credit card features: <code>Time</code>, <code>Amount</code>, PCA vectors <code>V1–V28</code>, and target label <code>Class</code> (0 = Legit, 1 = Fraud).
              </p>

              <label className="upload-dropzone" htmlFor="csv-file-input">
                <input
                  id="csv-file-input"
                  type="file"
                  accept=".csv"
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                  disabled={uploadLoading}
                />
                <div className="upload-icon-circle">
                  {uploadLoading ? <RotateCw className="animate-spin" size={24} /> : <Upload size={24} />}
                </div>
                <div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a' }}>
                    {uploadLoading ? 'Uploading and Preprocessing Dataset...' : 'Click to Browse or Drag & Drop CSV File'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                    Supported format: .csv (Kaggle Credit Card Fraud Detection dataset)
                  </div>
                </div>
              </label>

              {/* Benchmark Reset Action */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.2rem', paddingTop: '1rem', borderTop: '1px solid #f1f5f9' }}>
                <span style={{ fontSize: '0.78rem', color: '#64748b' }}>
                  Don't have a dataset ready? Load the pre-configured Kaggle baseline.
                </span>
                <button
                  className="btn-outline"
                  onClick={handleLoadDemoDataset}
                  disabled={uploadLoading}
                >
                  ⚡ Load Kaggle Benchmark Baseline (2,500 Rows)
                </button>
              </div>
            </div>

            {/* Active Dataset Inspection Cards */}
            <div className="content-card" style={{ marginBottom: '1.35rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>
                    Active Training Dataset: <span style={{ color: '#2563eb' }}>{activeDatasetInfo.filename}</span>
                  </h4>
                  <p style={{ fontSize: '0.76rem', color: '#64748b' }}>
                    Preprocessed with StandardScaling on Amount &amp; Time with stratified train-test split.
                  </p>
                </div>
                <button
                  className="btn-blue"
                  disabled={loading}
                  onClick={() => handleTrain('all')}
                >
                  {loading ? <RotateCw className="animate-spin" size={14} /> : '🚀 Train All Models on This Dataset'}
                </button>
              </div>

              <div className="metrics-grid" style={{ marginBottom: 0 }}>
                <div className="stat-card">
                  <span className="stat-label">Total Transactions</span>
                  <div className="stat-value" style={{ margin: '0.45rem 0' }}>
                    {activeDatasetInfo.total_rows.toLocaleString()}
                  </div>
                  <span style={{ fontSize: '0.72rem', color: '#64748b' }}>{activeDatasetInfo.features_count} feature dimensions</span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Legitimate Swipes (Class 0)</span>
                  <div className="stat-value" style={{ margin: '0.45rem 0', color: '#10b981' }}>
                    {activeDatasetInfo.legit_rows.toLocaleString()}
                  </div>
                  <span style={{ fontSize: '0.72rem', color: '#10b981', fontWeight: 600 }}>
                    {(100 - activeDatasetInfo.fraud_percentage).toFixed(2)}% of dataset
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Fraudulent Swipes (Class 1)</span>
                  <div className="stat-value" style={{ margin: '0.45rem 0', color: '#ef4444' }}>
                    {activeDatasetInfo.fraud_rows.toLocaleString()}
                  </div>
                  <span style={{ fontSize: '0.72rem', color: '#ef4444', fontWeight: 600 }}>
                    {activeDatasetInfo.fraud_percentage}% fraud ratio
                  </span>
                </div>

                <div className="stat-card">
                  <span className="stat-label">Feature Columns</span>
                  <div className="stat-value" style={{ margin: '0.45rem 0', color: '#2563eb' }}>
                    30 Cols
                  </div>
                  <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Time, Amount, V1–V28</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW 3: MODEL GRAPHS & ROC / PR CURVES (The New Requested Feature) ── */}
        {currentTab === 'graphs' && (
          <div>
            <div style={{ marginBottom: '1.2rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Classifier Evaluation &amp; Performance Graphs</h3>
              <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                Inspect the Receiver Operating Characteristic (ROC), Precision-Recall curve, and Confusion Matrix for any selected model.
              </p>
            </div>

            {/* Model Selector Pills */}
            <div className="model-pills">
              {['Random Forest', 'K-Nearest Neighbors', 'Decision Tree', 'Logistic Regression'].map((name) => (
                <button
                  key={name}
                  className={`pill-btn ${graphModel === name ? 'active' : ''}`}
                  onClick={() => handleGraphModelChange(name)}
                >
                  <span>{name === 'Random Forest' ? '🌳' : name === 'K-Nearest Neighbors' ? '📌' : name === 'Decision Tree' ? '🌲' : '📈'}</span>
                  <span>{name}</span>
                </button>
              ))}
            </div>

            {graphLoading ? (
              <div style={{ textAlign: 'center', padding: '3rem', color: '#64748b' }}>
                <RotateCw className="animate-spin" size={28} style={{ margin: '0 auto 0.75rem' }} />
                <div>Computing ROC &amp; Precision-Recall curves for {graphModel}...</div>
              </div>
            ) : graphData ? (
              <div>
                {/* Top Metrics Row */}
                <div className="metrics-grid">
                  <div className="stat-card">
                    <span className="stat-label">Accuracy Score</span>
                    <div className="stat-value" style={{ margin: '0.35rem 0', color: '#2563eb' }}>
                      {graphData.metrics.accuracy}%
                    </div>
                    <div className="metric-prog-bar">
                      <div className="metric-prog-fill" style={{ width: `${graphData.metrics.accuracy}%` }}></div>
                    </div>
                  </div>

                  <div className="stat-card">
                    <span className="stat-label">Precision Rate</span>
                    <div className="stat-value" style={{ margin: '0.35rem 0', color: '#10b981' }}>
                      {graphData.metrics.precision}%
                    </div>
                    <div className="metric-prog-bar">
                      <div className="metric-prog-fill green" style={{ width: `${graphData.metrics.precision}%` }}></div>
                    </div>
                  </div>

                  <div className="stat-card">
                    <span className="stat-label">Recall (True Positive Rate)</span>
                    <div className="stat-value" style={{ margin: '0.35rem 0', color: '#8b5cf6' }}>
                      {graphData.metrics.recall}%
                    </div>
                    <div className="metric-prog-bar">
                      <div className="metric-prog-fill" style={{ width: `${graphData.metrics.recall}%`, background: '#8b5cf6' }}></div>
                    </div>
                  </div>

                  <div className="stat-card">
                    <span className="stat-label">ROC - Area Under Curve (AUC)</span>
                    <div className="stat-value" style={{ margin: '0.35rem 0', color: '#f59e0b' }}>
                      {(graphData.roc_curve.auc / 100).toFixed(3)}
                    </div>
                    <div className="metric-prog-bar">
                      <div className="metric-prog-fill" style={{ width: `${graphData.roc_curve.auc}%`, background: '#f59e0b' }}></div>
                    </div>
                  </div>
                </div>

                {/* Graph Grid: ROC Curve & Precision-Recall Curve */}
                <div className="graph-grid">
                  {/* ROC Curve Card */}
                  <div className="graph-card">
                    <div className="graph-card-header">
                      <div>
                        <h4>Receiver Operating Characteristic (ROC)</h4>
                        <span style={{ fontSize: '0.72rem', color: '#64748b' }}>False Positive Rate vs True Positive Rate</span>
                      </div>
                      <span className="badge badge-blue">AUC: {graphData.roc_curve.auc}%</span>
                    </div>

                    <div className="svg-graph-box">
                      <svg width="100%" height="100%" viewBox="0 0 400 220" preserveAspectRatio="none">
                        <defs>
                          <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#2563eb" stopOpacity="0.35" />
                            <stop offset="100%" stopColor="#2563eb" stopOpacity="0.02" />
                          </linearGradient>
                        </defs>
                        {/* Grid lines */}
                        <line x1="30" y1="30" x2="370" y2="30" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="75" x2="370" y2="75" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="120" x2="370" y2="120" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="165" x2="370" y2="165" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="190" x2="370" y2="190" stroke="#e2e8f0" strokeWidth="1.5" />
                        <line x1="30" y1="30" x2="30" y2="190" stroke="#e2e8f0" strokeWidth="1.5" />

                        {/* Diagonal Chance Line */}
                        <line x1="30" y1="190" x2="370" y2="30" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="4 4" />

                        {/* Shaded Area */}
                        <path
                          d={renderAreaPath(graphData.roc_curve.fpr, graphData.roc_curve.tpr, 400, 220, 30)}
                          fill="url(#rocGradient)"
                        />

                        {/* ROC Curve Path */}
                        <path
                          d={renderCurvePath(graphData.roc_curve.fpr, graphData.roc_curve.tpr, 400, 220, 30)}
                          fill="none"
                          stroke="#2563eb"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                        />

                        {/* Axis Labels */}
                        <text x="30" y="210" fontSize="10" fill="#94a3b8">0.0 (FPR)</text>
                        <text x="200" y="212" fontSize="10" fill="#64748b" textAnchor="middle">False Positive Rate</text>
                        <text x="370" y="210" fontSize="10" fill="#94a3b8" textAnchor="end">1.0</text>
                        <text x="12" y="35" fontSize="10" fill="#94a3b8">1.0</text>
                        <text x="12" y="190" fontSize="10" fill="#94a3b8">0.0</text>
                      </svg>
                    </div>
                  </div>

                  {/* Precision-Recall Curve Card */}
                  <div className="graph-card">
                    <div className="graph-card-header">
                      <div>
                        <h4>Precision - Recall Curve</h4>
                        <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Tradeoff for imbalanced fraud datasets</span>
                      </div>
                      <span className="badge badge-green">Avg Precision: {graphData.pr_curve.avg_precision}%</span>
                    </div>

                    <div className="svg-graph-box">
                      <svg width="100%" height="100%" viewBox="0 0 400 220" preserveAspectRatio="none">
                        <defs>
                          <linearGradient id="prGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity="0.35" />
                            <stop offset="100%" stopColor="#10b981" stopOpacity="0.02" />
                          </linearGradient>
                        </defs>
                        {/* Grid lines */}
                        <line x1="30" y1="30" x2="370" y2="30" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="75" x2="370" y2="75" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="120" x2="370" y2="120" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="165" x2="370" y2="165" stroke="#f1f5f9" strokeWidth="1" />
                        <line x1="30" y1="190" x2="370" y2="190" stroke="#e2e8f0" strokeWidth="1.5" />
                        <line x1="30" y1="30" x2="30" y2="190" stroke="#e2e8f0" strokeWidth="1.5" />

                        {/* Shaded Area */}
                        <path
                          d={renderAreaPath(graphData.pr_curve.recall, graphData.pr_curve.precision, 400, 220, 30)}
                          fill="url(#prGradient)"
                        />

                        {/* PR Curve Path */}
                        <path
                          d={renderCurvePath(graphData.pr_curve.recall, graphData.pr_curve.precision, 400, 220, 30)}
                          fill="none"
                          stroke="#10b981"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                        />

                        {/* Axis Labels */}
                        <text x="30" y="210" fontSize="10" fill="#94a3b8">0.0 (Recall)</text>
                        <text x="200" y="212" fontSize="10" fill="#64748b" textAnchor="middle">Recall</text>
                        <text x="370" y="210" fontSize="10" fill="#94a3b8" textAnchor="end">1.0</text>
                        <text x="12" y="35" fontSize="10" fill="#94a3b8">1.0</text>
                        <text x="12" y="190" fontSize="10" fill="#94a3b8">0.0</text>
                      </svg>
                    </div>
                  </div>
                </div>

                {/* Bottom Row: Confusion Matrix & Classification Report */}
                <div className="graph-grid">
                  {/* Confusion Matrix Heatmap Card */}
                  <div className="graph-card">
                    <div className="graph-card-header">
                      <div>
                        <h4>Confusion Matrix</h4>
                        <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Actual vs Predicted outcome counts</span>
                      </div>
                      <span className="badge badge-amber">{graphModel}</span>
                    </div>

                    <div className="confusion-matrix-box">
                      <div></div>
                      <div className="cm-label">Pred: Legit (0)</div>
                      <div className="cm-label">Pred: Fraud (1)</div>

                      <div className="cm-label" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual: Legit (0)</div>
                      <div className="cm-cell cm-cell-tn">
                        <span className="cm-cell-title">True Negative (TN)</span>
                        <span className="cm-cell-value">{graphData.confusion_matrix[0][0]}</span>
                        <span style={{ fontSize: '0.68rem', color: '#1e3a8a' }}>Legitimate Approved</span>
                      </div>
                      <div className="cm-cell cm-cell-fp">
                        <span className="cm-cell-title">False Positive (FP)</span>
                        <span className="cm-cell-value">{graphData.confusion_matrix[0][1]}</span>
                        <span style={{ fontSize: '0.68rem', color: '#92400e' }}>False Alarm</span>
                      </div>

                      <div className="cm-label" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual: Fraud (1)</div>
                      <div className="cm-cell cm-cell-fn">
                        <span className="cm-cell-title">False Negative (FN)</span>
                        <span className="cm-cell-value">{graphData.confusion_matrix[1][0]}</span>
                        <span style={{ fontSize: '0.68rem', color: '#991b1b' }}>Missed Fraud</span>
                      </div>
                      <div className="cm-cell cm-cell-tp">
                        <span className="cm-cell-title">True Positive (TP)</span>
                        <span className="cm-cell-value">{graphData.confusion_matrix[1][1]}</span>
                        <span style={{ fontSize: '0.68rem', color: '#065f46' }}>Fraud Intercepted!</span>
                      </div>
                    </div>
                  </div>

                  {/* Classification Report Card */}
                  <div className="graph-card">
                    <div className="graph-card-header">
                      <div>
                        <h4>Classification Report Metrics</h4>
                        <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Class-level precision, recall, and support</span>
                      </div>
                      <span className="badge badge-blue">F1: {graphData.metrics.f1}%</span>
                    </div>

                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', marginTop: '0.5rem' }}>
                      <thead>
                        <tr style={{ borderBottom: '1.5px solid #f1f5f9', textAlign: 'left', color: '#64748b' }}>
                          <th style={{ padding: '0.5rem 0.6rem' }}>Class</th>
                          <th style={{ padding: '0.5rem 0.6rem' }}>Precision</th>
                          <th style={{ padding: '0.5rem 0.6rem' }}>Recall</th>
                          <th style={{ padding: '0.5rem 0.6rem' }}>F1-Score</th>
                          <th style={{ padding: '0.5rem 0.6rem' }}>Support</th>
                        </tr>
                      </thead>
                      <tbody>
                        {['0', '1'].map((cKey) => {
                          const cData = graphData.classification_report[cKey];
                          if (!cData) return null;
                          return (
                            <tr key={cKey} style={{ borderBottom: '1px solid #f1f5f9' }}>
                              <td style={{ padding: '0.65rem 0.6rem', fontWeight: 700, color: '#0f172a' }}>
                                {cKey === '0' ? 'Legitimate (0)' : 'Fraud (1)'}
                              </td>
                              <td style={{ padding: '0.65rem 0.6rem' }}>{(cData.precision * 100).toFixed(1)}%</td>
                              <td style={{ padding: '0.65rem 0.6rem' }}>{(cData.recall * 100).toFixed(1)}%</td>
                              <td style={{ padding: '0.65rem 0.6rem', fontWeight: 700, color: '#2563eb' }}>
                                {(cData['f1-score'] * 100).toFixed(1)}%
                              </td>
                              <td style={{ padding: '0.65rem 0.6rem', color: '#64748b' }}>{cData.support}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        )}

        {/* ── VIEW 4: LIVE FRAUD PREDICTION (Transaction Inspector) ── */}
        {currentTab === 'predict' && (
          <div>
            <div style={{ marginBottom: '1.2rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Transaction Anomaly Inspector</h3>
              <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                Test individual transaction vectors against active ML models in real-time.
              </p>
            </div>

            {/* Presets Header Card */}
            <div className="content-card" style={{ marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.2rem' }}>
                    ⚡ Instant Preset Scenarios
                  </h4>
                  <p style={{ fontSize: '0.78rem', color: '#64748b' }}>
                    Load authentic transactions from Kaggle with one click to test legitimacy vs fraud detection.
                  </p>
                </div>

                <div style={{ display: 'flex', gap: '0.65rem' }}>
                  <button
                    className="btn-outline"
                    onClick={() => handleFillPreset(false)}
                  >
                    📋 Fill Legit Sample
                  </button>
                  <button
                    className="btn-blue"
                    style={{ background: '#dc2626' }}
                    onClick={() => handleFillPreset(true)}
                  >
                    ⚠️ Fill Fraud Sample
                  </button>
                </div>
              </div>
            </div>

            {/* Main Form & Live Verdict Columns */}
            <div className="columns-split">
              {/* Left Column: Transaction Inputs */}
              <div className="content-card">
                <h3 style={{ marginBottom: '0.85rem' }}>💳 Transaction Parameters</h3>

                {/* Model Selector */}
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#64748b', display: 'block', marginBottom: '0.35rem' }}>
                    Active Classifier
                  </label>
                  <select
                    style={{ width: '100%', padding: '0.55rem 0.8rem', borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff', fontSize: '0.85rem' }}
                    value={selectedModel}
                    onChange={(e) => {
                      setSelectedModel(e.target.value);
                      runPrediction(transaction, e.target.value);
                    }}
                  >
                    <option value="Random Forest">Random Forest Classifier</option>
                    <option value="K-Nearest Neighbors">K-Nearest Neighbors</option>
                    <option value="Decision Tree">Decision Tree</option>
                    <option value="Logistic Regression">Logistic Regression</option>
                  </select>
                </div>

                {/* Amount & Time */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem', marginBottom: '1rem' }}>
                  <div>
                    <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#64748b', display: 'block', marginBottom: '0.35rem' }}>
                      Transaction Amount ($)
                    </label>
                    <input
                      type="number"
                      step="5"
                      style={{ width: '100%', padding: '0.55rem 0.8rem', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: '0.88rem', fontWeight: 600 }}
                      value={transaction.Amount}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value) || 0;
                        const updated = { ...transaction, Amount: val };
                        setTransaction(updated);
                        runPrediction(updated, selectedModel);
                      }}
                    />
                  </div>

                  <div>
                    <label style={{ fontSize: '0.76rem', fontWeight: 600, color: '#64748b', display: 'block', marginBottom: '0.35rem' }}>
                      Time Offset (seconds)
                    </label>
                    <input
                      type="number"
                      step="10"
                      style={{ width: '100%', padding: '0.55rem 0.8rem', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: '0.88rem' }}
                      value={transaction.Time}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value) || 0;
                        const updated = { ...transaction, Time: val };
                        setTransaction(updated);
                        runPrediction(updated, selectedModel);
                      }}
                    />
                  </div>
                </div>

                {/* Amount Shortcut Buttons */}
                <div style={{ marginBottom: '1.2rem' }}>
                  <span style={{ fontSize: '0.72rem', color: '#64748b', display: 'block', marginBottom: '0.35rem' }}>
                    Quick Amount Shortcuts:
                  </span>
                  <div style={{ display: 'flex', gap: '0.45rem' }}>
                    {[4.5, 45.0, 250.0, 1500.0].map((amt) => (
                      <button
                        key={amt}
                        className="btn-outline"
                        style={{ fontSize: '0.75rem', padding: '0.3rem 0.65rem' }}
                        onClick={() => handleAmountShortcut(amt)}
                      >
                        ${amt.toFixed(2)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Collapsible PCA Features Expander */}
                <div style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: '0.75rem 1rem', background: '#f8fafc' }}>
                  <div
                    style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                    onClick={() => setShowPcaExpander(!showPcaExpander)}
                  >
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#0f172a' }}>
                      🛠️ Advanced PCA Vectors (V1 – V28)
                    </span>
                    <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                      {showPcaExpander ? '▲ Hide' : '▼ Expand'}
                    </span>
                  </div>

                  {showPcaExpander && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.45rem', marginTop: '0.75rem' }}>
                      {Array.from({ length: 28 }, (_, i) => i + 1).map((idx) => {
                        const vKey = `V${idx}`;
                        return (
                          <div key={vKey}>
                            <label style={{ fontSize: '0.65rem', color: '#64748b' }}>{vKey}</label>
                            <input
                              type="number"
                              step="0.1"
                              style={{ width: '100%', padding: '0.25rem 0.4rem', fontSize: '0.75rem', borderRadius: 6, border: '1px solid #cbd5e1' }}
                              value={transaction[vKey] || 0}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value) || 0;
                                const updated = { ...transaction, [vKey]: val };
                                setTransaction(updated);
                                runPrediction(updated, selectedModel);
                              }}
                            />
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: Live Detection Verdict */}
              <div className="content-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <h3 style={{ marginBottom: '1rem' }}>🎯 Live Anomaly Verdict</h3>

                  {predictionResult && (
                    <div>
                      {/* Risk Percentage Gauge Box */}
                      <div style={{ textAlign: 'center', padding: '1.5rem', background: '#f8fafc', borderRadius: 14, border: '1px solid #e2e8f0', marginBottom: '1.25rem' }}>
                        <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.35rem' }}>
                          Fraud Probability Score
                        </div>
                        <div
                          style={{
                            fontSize: '2.8rem',
                            fontWeight: 800,
                            color: predictionResult.prediction === 1 ? '#ef4444' : '#10b981',
                            lineHeight: 1,
                          }}
                        >
                          {predictionResult.fraud_probability}%
                        </div>
                        <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '0.45rem' }}>
                          Evaluated with {predictionResult.model_used}
                        </div>
                      </div>

                      {/* Status Card */}
                      {predictionResult.prediction === 1 ? (
                        <div style={{ background: '#fef2f2', border: '1.5px solid #fecaca', borderRadius: 12, padding: '1.1rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#dc2626', fontWeight: 700, fontSize: '0.75rem', textTransform: 'uppercase' }}>
                            <ShieldAlert size={16} />
                            <span>ANOMALY FLAGGED</span>
                          </div>
                          <h4 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#991b1b', margin: '0.35rem 0' }}>
                            High-Risk Fraud
                          </h4>
                          <p style={{ fontSize: '0.78rem', color: '#7f1d1d', lineHeight: 1.45 }}>
                            Transaction components match known fraudulent charge patterns. Immediate freeze recommended.
                          </p>
                        </div>
                      ) : (
                        <div style={{ background: '#ecfdf5', border: '1.5px solid #a7f3d0', borderRadius: 12, padding: '1.1rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#059669', fontWeight: 700, fontSize: '0.75rem', textTransform: 'uppercase' }}>
                            <ShieldCheck size={16} />
                            <span>VERIFIED TRANSACTION</span>
                          </div>
                          <h4 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#065f46', margin: '0.35rem 0' }}>
                            Legitimate Swipe
                          </h4>
                          <p style={{ fontSize: '0.78rem', color: '#047857', lineHeight: 1.45 }}>
                            Transaction fits normal consumer spending habits and passed all anomaly thresholds.
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div style={{ fontSize: '0.72rem', color: '#94a3b8', textAlign: 'center', paddingTop: '1rem', borderTop: '1px solid #f1f5f9' }}>
                  Model inference executes directly via FastAPI on port 8000
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW 5: MODEL BENCHMARKS ── */}
        {currentTab === 'train' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Classifier Benchmark &amp; Training</h3>
                <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                  Train individual classifiers (Random Forest, KNN, Decision Tree, Logistic Regression) or run full benchmarks on active dataset.
                </p>
              </div>
              <button
                className="btn-blue"
                disabled={loading}
                onClick={() => handleTrain('all')}
              >
                {loading ? <RotateCw className="animate-spin" size={14} /> : '🚀 Train All Models'}
              </button>
            </div>

            {trainingStatus && (
              <div style={{ padding: '0.85rem 1.1rem', background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 12, marginBottom: '1.2rem', fontSize: '0.84rem', color: '#1e3a8a', fontWeight: 600 }}>
                {trainingStatus}
              </div>
            )}

            {/* 4 Model Training Cards */}
            <div className="metrics-grid">
              {[
                { name: 'Random Forest', icon: '🌳', desc: 'Ensemble trees · High precision & recall' },
                { name: 'K-Nearest Neighbors', icon: '📌', desc: 'Instance similarity distance voting' },
                { name: 'Decision Tree', icon: '🌲', desc: 'Rule-based explainable split logic' },
                { name: 'Logistic Regression', icon: '📈', desc: 'Linear benchmark baseline' },
              ].map((m) => (
                <div key={m.name} className="stat-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <span style={{ fontSize: '1.35rem' }}>{m.icon}</span>
                      <span className="badge badge-green">Ready</span>
                    </div>
                    <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.2rem' }}>{m.name}</h4>
                    <p style={{ fontSize: '0.72rem', color: '#64748b', lineHeight: 1.4, marginBottom: '0.75rem' }}>{m.desc}</p>
                  </div>
                  <button
                    className="btn-blue"
                    style={{ width: '100%', fontSize: '0.78rem', padding: '0.42rem 0.8rem' }}
                    disabled={loading}
                    onClick={() => handleTrain(m.name)}
                  >
                    Train {m.name.split(' ')[0]}
                  </button>
                </div>
              ))}
            </div>

            {/* Side-by-Side Comparison Table */}
            {modelsData.length > 0 && (
              <div className="content-card" style={{ marginTop: '1.25rem' }}>
                <h3 style={{ marginBottom: '0.85rem' }}>📊 Side-by-Side Model Comparison Table</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '2px solid #f1f5f9', textAlign: 'left', color: '#64748b' }}>
                        <th style={{ padding: '0.65rem 0.8rem' }}>Model</th>
                        <th style={{ padding: '0.65rem 0.8rem' }}>Accuracy</th>
                        <th style={{ padding: '0.65rem 0.8rem' }}>Precision</th>
                        <th style={{ padding: '0.65rem 0.8rem' }}>Recall</th>
                        <th style={{ padding: '0.65rem 0.8rem' }}>F1-Score</th>
                        <th style={{ padding: '0.65rem 0.8rem' }}>ROC-AUC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelsData.map((r) => (
                        <tr key={r.name} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '0.65rem 0.8rem', fontWeight: 700, color: '#0f172a' }}>{r.name}</td>
                          <td style={{ padding: '0.65rem 0.8rem', color: '#2563eb', fontWeight: 600 }}>{r.accuracy}%</td>
                          <td style={{ padding: '0.65rem 0.8rem' }}>{r.precision}%</td>
                          <td style={{ padding: '0.65rem 0.8rem' }}>{r.recall}%</td>
                          <td style={{ padding: '0.65rem 0.8rem', color: '#10b981', fontWeight: 700 }}>{r.f1}%</td>
                          <td style={{ padding: '0.65rem 0.8rem' }}>{r.roc_auc}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── VIEW 6: ARCHITECTURE ── */}
        {currentTab === 'about' && (
          <div>
            <div style={{ marginBottom: '1.2rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>FraudShield AI Platform Architecture</h3>
              <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                Full-stack machine learning anomaly detection system built with React &amp; FastAPI.
              </p>
            </div>

            <div className="content-card">
              <h4 style={{ marginBottom: '0.75rem' }}>System Specifications</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.86rem' }}>
                <tbody>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '0.65rem 0', color: '#64748b', width: '180px', fontWeight: 600 }}>Frontend</td>
                    <td style={{ padding: '0.65rem 0', color: '#0f172a', fontWeight: 700 }}>React 19 + Vite (InsightHub Theme)</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '0.65rem 0', color: '#64748b', fontWeight: 600 }}>Backend API</td>
                    <td style={{ padding: '0.65rem 0', color: '#2563eb', fontWeight: 600 }}>FastAPI + Uvicorn (Port 8000)</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '0.65rem 0', color: '#64748b', fontWeight: 600 }}>Classifiers</td>
                    <td style={{ padding: '0.65rem 0', color: '#0f172a' }}>Random Forest · K-Nearest Neighbors · Decision Tree · Logistic Regression</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '0.65rem 0', color: '#64748b', fontWeight: 600 }}>Dataset Format</td>
                    <td style={{ padding: '0.65rem 0', color: '#0f172a' }}>Kaggle European Cardholders Schema (V1–V28, Time, Amount, Class)</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '0.65rem 0', color: '#64748b', fontWeight: 600 }}>Visualizations</td>
                    <td style={{ padding: '0.65rem 0', color: '#0f172a' }}>ROC Curves, Precision-Recall Curves, Confusion Matrix Heatmaps</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
