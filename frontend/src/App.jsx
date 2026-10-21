import React, { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  Inbox,
  TrendingUp,
  FolderGit2,
  CheckSquare,
  CalendarX2,
  LineChart,
  Search,
  Settings,
  Bell,
  ChevronRight,
  Maximize2,
  User,
  Folder,
  Paperclip,
  Percent,
  Play,
  RotateCw,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Upload,
  Database
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
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
    }
  };

  useEffect(() => {
    fetchDashboard();
    // Run initial prediction
    runPrediction(transaction, selectedModel);
  }, []);

  // Train a specific model or all models
  const handleTrain = async (modelName) => {
    setLoading(true);
    setTrainingStatus(`Training ${modelName === 'all' ? 'all 4 classifiers' : modelName}...`);
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
    } catch (err) {
      setTrainingStatus(`❌ Training failed: ${err.message}`);
    } finally {
      setLoading(false);
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

  return (
    <div className="dashboard-canvas">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div>
          {/* Brand Logo */}
          <div className="brand-header">
            <div className="brand-logo">G</div>
            <div className="brand-title">InsightHub</div>
          </div>

          <div className="nav-section-title">Dashboard</div>

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
              className={`nav-item ${currentTab === 'train' ? 'active' : ''}`}
              onClick={() => setCurrentTab('train')}
            >
              <div className="nav-left">
                <TrendingUp size={18} />
                <span>Performances</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className={`nav-item ${currentTab === 'predict' ? 'active' : ''}`}
              onClick={() => setCurrentTab('predict')}
            >
              <div className="nav-left">
                <ShieldCheck size={18} />
                <span>Prediction</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className={`nav-item ${currentTab === 'dataset' ? 'active' : ''}`}
              onClick={() => setCurrentTab('dataset')}
            >
              <div className="nav-left">
                <Database size={18} />
                <span>Dataset &amp; Prep</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            {/* Projects Tree Section (Matching Screenshot) */}
            <div className="sidebar-projects-box">
              <div className="projects-tree-title">
                <span>PROJECTS</span>
                <span>▾</span>
              </div>
              <div className="projects-tree-items">
                <div className="project-dot-row">
                  <span className="status-dot" style={{ background: '#2563eb' }}></span>
                  <span>Active Project</span>
                </div>
                <div className="project-dot-row">
                  <span className="status-dot" style={{ background: '#10b981' }}></span>
                  <span>Project Done</span>
                </div>
                <div className="project-dot-row">
                  <span className="status-dot" style={{ background: '#ef4444' }}></span>
                  <span>Project On Hold</span>
                </div>
              </div>
            </div>

            <button
              className={`nav-item ${currentTab === 'about' ? 'active' : ''}`}
              onClick={() => setCurrentTab('about')}
            >
              <div className="nav-left">
                <CheckSquare size={18} />
                <span>Employee Task</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className="nav-item"
              onClick={() => setCurrentTab('about')}
            >
              <div className="nav-left">
                <CalendarX2 size={18} />
                <span>Absence</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>

            <button
              className={`nav-item ${currentTab === 'train' ? 'active' : ''}`}
              onClick={() => setCurrentTab('train')}
            >
              <div className="nav-left">
                <LineChart size={18} />
                <span>Analytics</span>
              </div>
              <ChevronRight size={14} className="nav-chevron" />
            </button>
          </nav>
        </div>

        {/* Profile Card at Bottom */}
        <div className="profile-card">
          <div className="profile-avatar-circle">MK</div>
          <div>
            <div className="profile-name">Markus Keller</div>
            <div className="profile-role">Risk &amp; Fraud Lead</div>
          </div>
        </div>
      </aside>

      {/* ── Main Dashboard Content ── */}
      <main className="main-content">
        {/* Top Header */}
        <header className="top-header">
          <div className="user-greeting">
            <div className="user-avatar">👨‍💼</div>
            <div className="greeting-text">
              <h2>Hey, Markus</h2>
              <p>Sunday, June 25, 2024</p>
            </div>
          </div>

          <div className="header-actions">
            <div className="search-input-box">
              <Search size={15} color="#94a3b8" />
              <input type="text" placeholder="Start searching here..." />
            </div>
            <div className="header-btn" title="Settings">
              <Settings size={17} />
            </div>
            <div className="header-btn" title="Notifications">
              <Bell size={17} />
              <span className="badge-dot"></span>
            </div>
          </div>
        </header>

        {/* ── VIEW 1: DASHBOARD (Home matching screenshot) ── */}
        {currentTab === 'dashboard' && (
          <div>
            {/* Alert Notification Banner ("Dear Manager") */}
            <div className="alert-banner">
              <div className="alert-left">
                <div className="alert-icon-circle">🔔</div>
                <div className="alert-text">
                  <h4>Dear Manager</h4>
                  <p>
                    We have observed a decline in{' '}
                    <span style={{ color: '#ef4444', fontWeight: 600 }}>[Hermawan]</span>'s performance over the past 2 weeks.
                  </p>
                </div>
              </div>
              <button
                className="btn-blue"
                onClick={() => setCurrentTab('train')}
              >
                View Detail
              </button>
            </div>

            {/* 4 Stat Metric Cards (547, 339, 147, 89.75%) */}
            <div className="metrics-grid">
              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Active Employees</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box">
                    <User size={16} />
                  </div>
                  <div className="stat-value">547</div>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Number of Projects</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box">
                    <Folder size={16} />
                  </div>
                  <div className="stat-value">339</div>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Number of Task</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box">
                    <Paperclip size={16} />
                  </div>
                  <div className="stat-value">147</div>
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <span className="stat-label">Target Percentage Completed</span>
                  <Maximize2 size={13} className="stat-expand" />
                </div>
                <div className="stat-bottom">
                  <div className="stat-icon-box">
                    <Percent size={16} />
                  </div>
                  <div className="stat-value">89.75%</div>
                </div>
              </div>
            </div>

            {/* Two-Column Layout: On Going Task & Graphs/Analysis */}
            <div className="columns-split">
              {/* Left Column: On Going Task */}
              <div className="content-card">
                <div className="card-header-row">
                  <div className="card-title-left">
                    <span style={{ color: '#2563eb' }}>🎯</span>
                    <h3>On Going Task</h3>
                  </div>
                  <div style={{ display: 'flex', gap: '0.45rem', color: '#94a3b8' }}>
                    <Search size={15} style={{ cursor: 'pointer' }} />
                  </div>
                </div>
                <p className="card-subtitle">Best performing employee ranking.</p>

                {/* Task 1: Journey Scarves */}
                <div className="task-item">
                  <div className="task-left">
                    <div className="task-avatar-tag" style={{ background: '#0f172a' }}>JS</div>
                    <div className="task-meta">
                      <h5>Journey Scarves</h5>
                      <p>Rebranding and Website Design</p>
                    </div>
                  </div>
                  <div className="avatar-stack">
                    <div className="mini-avatar" style={{ background: '#3b82f6', color: '#fff' }}>👩‍💻</div>
                    <div className="mini-avatar" style={{ background: '#10b981', color: '#fff' }}>👨‍🎨</div>
                    <div className="mini-avatar" style={{ background: '#f59e0b', color: '#fff' }}>👩‍💼</div>
                  </div>
                </div>

                {/* Task 2: Edfier */}
                <div className="task-item" style={{ flexDirection: 'column', alignItems: 'stretch', gap: '0.65rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div className="task-left">
                      <div className="task-avatar-tag" style={{ background: '#38bdf8' }}>ED</div>
                      <div className="task-meta">
                        <h5>Edfier</h5>
                        <p>Web Design &amp; Development</p>
                      </div>
                    </div>
                    <div className="avatar-stack">
                      <div className="mini-avatar" style={{ background: '#8b5cf6', color: '#fff' }}>👩‍💻</div>
                      <div className="mini-avatar" style={{ background: '#ec4899', color: '#fff' }}>👨‍💻</div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#64748b', paddingTop: '0.45rem', borderTop: '1px solid #f1f5f9' }}>
                    <div>Status: <strong style={{ color: '#0f172a' }}>On Going</strong></div>
                    <div>Progression: <strong style={{ color: '#2563eb' }}>51%</strong></div>
                    <div>Due Date: <strong style={{ color: '#0f172a' }}>Aug, 17 2024</strong></div>
                  </div>
                </div>

                {/* Task 3: Random Forest ML Shield */}
                <div className="task-item">
                  <div className="task-left">
                    <div className="task-avatar-tag" style={{ background: '#2563eb' }}>FS</div>
                    <div className="task-meta">
                      <h5>Random Forest Anomaly Shield</h5>
                      <p>Real-time Credit Card Anomaly Engine</p>
                    </div>
                  </div>
                  <span className="badge badge-green">99.8% Acc</span>
                </div>
              </div>

              {/* Right Column: Graphs and Analysis */}
              <div className="content-card">
                <div className="card-header-row">
                  <div className="card-title-left">
                    <span style={{ color: '#2563eb' }}>📊</span>
                    <h3>Graphs and Analysis</h3>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
                    <span style={{ fontSize: '0.74rem', fontWeight: 600, color: '#0f172a', background: '#f8fafc', border: '1px solid #e2e8f0', padding: '0.2rem 0.6rem', borderRadius: 6 }}>
                      Month ▾
                    </span>
                    <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>↑</span>
                  </div>
                </div>
                <p className="card-subtitle">Projects completed per month based on trends.</p>

                {/* Rounded Bar Chart with Floating Tooltip (Matching Screenshot) */}
                <div className="bar-chart-container">
                  {/* Floating Dark Tooltip above Mei */}
                  <div className="floating-tooltip">
                    <div className="tooltip-row">
                      <span style={{ color: '#38bdf8' }}>●</span>
                      <span>Project Done: <strong>137</strong></span>
                    </div>
                    <div className="tooltip-row">
                      <span style={{ color: '#818cf8' }}>●</span>
                      <span>Project New: <strong>123</strong></span>
                    </div>
                    <div className="tooltip-row">
                      <span style={{ color: '#f472b6' }}>●</span>
                      <span>Project Cost: <strong>84</strong></span>
                    </div>
                  </div>

                  {dashboardData?.monthly_trends?.map((item) => (
                    <div key={item.month} className="chart-bar-group">
                      <div
                        className={`chart-bar ${item.highlight ? 'highlight' : ''}`}
                        style={{ height: `${item.value * 1.5}px` }}
                      ></div>
                      <span className="chart-month-label">{item.month}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW 2: TRAIN & PERFORMANCES ── */}
        {currentTab === 'train' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Classifier Benchmark &amp; Training</h3>
                <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                  Train individual classifiers (Random Forest, KNN, Decision Tree, Logistic Regression) or run full benchmarks.
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
                { name: 'K-Nearest Neighbors', icon: '📌', desc: 'Instance similarity voting' },
                { name: 'Decision Tree', icon: '🌲', desc: 'Rule-based explainable logic' },
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

        {/* ── VIEW 3: LIVE FRAUD PREDICTION ── */}
        {currentTab === 'predict' && (
          <div>
            <div style={{ marginBottom: '1.2rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Transaction Fraud Inspector</h3>
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
                    Fill authentic transaction patterns from the Kaggle dataset with one click.
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

                {/* Collapsible PCA Features Expander (Hidden by default!) */}
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

        {/* ── VIEW 4: DATASET & PREP ── */}
        {currentTab === 'dataset' && (
          <div>
            <div style={{ marginBottom: '1.2rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Transaction Dataset</h3>
              <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
                Inspect, generate, and explore transactions for machine learning models.
              </p>
            </div>

            <div className="content-card" style={{ marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.2rem' }}>
                    ⚡ One-Click Demo Dataset (2,500 Rows)
                  </h4>
                  <p style={{ fontSize: '0.78rem', color: '#64748b' }}>
                    Generate synthetic credit card transactions matching the exact Kaggle PCA schema.
                  </p>
                </div>
                <button
                  className="btn-blue"
                  onClick={async () => {
                    const res = await fetch(`${API_BASE}/dataset/load-demo`, { method: 'POST' });
                    const data = await res.json();
                    alert(`✅ Generated ${data.rows} rows of transactions!`);
                    fetchDashboard();
                  }}
                >
                  ⚡ Load Demo Dataset
                </button>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="stat-card">
                <span className="stat-label">Total Transactions</span>
                <div className="stat-value" style={{ margin: '0.45rem 0' }}>2,500</div>
                <span style={{ fontSize: '0.72rem', color: '#64748b' }}>30 total feature columns</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Legitimate Ratio</span>
                <div className="stat-value" style={{ margin: '0.45rem 0', color: '#2563eb' }}>98.0%</div>
                <span style={{ fontSize: '0.72rem', color: '#64748b' }}>2,450 clean records</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Anomaly Ratio</span>
                <div className="stat-value" style={{ margin: '0.45rem 0', color: '#ef4444' }}>2.0%</div>
                <span style={{ fontSize: '0.72rem', color: '#ef4444' }}>50 fraud records</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">PCA Features</span>
                <div className="stat-value" style={{ margin: '0.45rem 0' }}>V1–V28</div>
                <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Time &amp; Amount scaled</span>
              </div>
            </div>
          </div>
        )}

        {/* ── VIEW 5: ABOUT ── */}
        {currentTab === 'about' && (
          <div>
            <div style={{ marginBottom: '1.2rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>Platform Architecture</h3>
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
                  <tr>
                    <td style={{ padding: '0.65rem 0', color: '#64748b', fontWeight: 600 }}>Dataset Format</td>
                    <td style={{ padding: '0.65rem 0', color: '#0f172a' }}>Kaggle European Cardholders Schema (V1–V28, Time, Amount, Class)</td>
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
