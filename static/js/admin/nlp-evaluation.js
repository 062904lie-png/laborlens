/* Admin-only benchmark reports. All dataset and prediction text is escaped. */
window.AdminNlpEvaluation = (() => {
  let selectedRun = null;
  let busy = false;
  let notifyCurrent = () => {};
  const pct = value => value == null ? 'Not evaluated' : `${(Number(value) * 100).toFixed(2)}%`;
  const names = {language: 'Language detection', intent: 'Intent detection', entities: 'Entity extraction'};
  async function api(path, options = {}) {
    const response = await apiFetch('/admin/nlp-evaluation' + path, options);
    const data = await response.json();
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Unable to complete NLP evaluation request.');
    return data;
  }
  function download(filename, value, type = 'application/json') {
    const url = URL.createObjectURL(new Blob([value], {type}));
    const link = document.createElement('a'); link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(url);
  }
  function metricTable(metrics) {
    return AdminUI.table(['Task', 'Accuracy', 'Precision', 'Recall', 'F1', 'Scored cases'], Object.entries(metrics).map(([task, values]) => [
      names[task], task === 'entities' ? 'N/A' : pct(values.accuracy), pct(values.precision), pct(values.recall), pct(values.f1), Number(values.samples)
    ]));
  }
  function classDetail(task, metric) {
    if (!metric.samples) return '';
    const rows = metric.classes.map(row => [esc(row.label), row.support, pct(row.precision), pct(row.recall), pct(row.f1)]);
    const matrix = AdminUI.table(['Expected / Predicted', ...metric.labels], metric.confusion.map((row, index) => [esc(metric.labels[index]), ...row]));
    return `<details><summary>${names[task]} by class and confusion matrix</summary>${AdminUI.table(['Label', 'Support', 'Precision', 'Recall', 'F1'], rows)}${matrix}</details>`;
  }
  function resultHtml(run) {
    const configuration = run.configuration;
    return `<header><div><h3>${esc(run.dataset_name)} · ${esc(run.dataset_version)}</h3><p>${esc(new Date(run.created_at).toLocaleString())} · ${run.completed} completed · ${run.failed} failed · ${run.mismatch_cases} cases with mismatches</p></div><div class="nlp-actions"><button type="button" class="btn btn-sm" data-export="summary">Export summary CSV</button><button type="button" class="btn btn-sm" data-export="cases">Export cases CSV</button></div></header>
      ${run.failed ? '<p class="nlp-warning">Some cases failed. Scores exclude failed cases; review their errors before interpreting the results.</p>' : ''}
      ${metricTable(run.metrics)}<p>Classification precision, recall and F1 are macro averages. Entity scores use micro counts. These scores measure the labeled NLP tasks, not legal answer accuracy.</p>
      <details><summary>Scoring method and configuration</summary><p>${esc(run.method)}</p><p>spaCy ${esc(configuration.spacy_version)} · ${esc(configuration.model)} ${esc(configuration.model_version || 'Version unavailable')} · scoring version ${esc(configuration.scoring_version)}</p>${configuration.learned_classifier ? `<p>Classifier: ${esc(configuration.learned_classifier.encoder)} · ${esc(configuration.learned_classifier.training_count)} training examples · label status: ${esc(configuration.learned_classifier.training_status)}</p><p>Model artifact SHA256: <code>${esc(configuration.learned_classifier.artifact_sha256)}</code><br>Training data SHA256: <code>${esc(configuration.learned_classifier.training_sha256)}</code></p>` : ''}<p>Dataset SHA256: <code>${esc(run.dataset_sha256)}</code><br>NLP source SHA256: <code>${esc(configuration.nlp_source_sha256)}</code></p></details>
      ${classDetail('language', run.metrics.language)}${classDetail('intent', run.metrics.intent)}
      <details><summary>Entity extraction by type</summary>${AdminUI.table(['Type', 'Annotated cases', 'Correct', 'Extra', 'Missed', 'Precision', 'Recall', 'F1'], run.metrics.entities.types.map(row => [esc(row.type), row.samples, row.tp, row.fp, row.fn, pct(row.precision), pct(row.recall), pct(row.f1)]))}</details>
      <details><summary>Results by reviewed language</summary>${Object.entries(run.by_language).map(([language, metrics]) => `<h4>${esc(language)}</h4>${metricTable(metrics)}`).join('')}<p>Cases without a reviewed language label are included in overall scores only.</p></details>
      <details open><summary>Case results</summary><label>Show <select data-case-filter><option value="issues">Mismatches and failures</option><option value="all">All cases</option></select></label><div data-case-results></div></details>`;
  }
  function renderCases(host, run) {
    const all = host.querySelector('[data-case-filter]').value === 'all';
    const rows = run.cases.filter(row => all || row.status === 'failed' || row.mismatches.length);
    host.querySelector('[data-case-results]').innerHTML = AdminUI.table(['Case / Question', 'Expected', 'Predicted', 'Result'], rows.map(row => [
      `<strong>${esc(row.id)}</strong><p>${esc(row.question)}</p>`, `<pre>${esc(JSON.stringify(row.expected, null, 2))}</pre>`,
      `<pre>${esc(JSON.stringify(row.predicted, null, 2))}</pre>`, esc(row.status === 'failed' ? row.error : row.mismatches.join(', ') || 'All annotated outputs match')
    ]));
  }
  function mount(parent) {
    const host = parent.querySelector('[data-nlp-evaluation]');
    if (!host) return;
    host.innerHTML = `<header><div><h2>NLP Evaluation</h2><p>Measure the existing language detector, intent detector and entity extractor against reviewed labels.</p></div><button class="btn btn-sm" type="button" data-example>Download format example</button></header>
      <details><summary>Dataset format and scoring rules</summary><p>Upload a UTF-8 JSON object with <code>name</code>, <code>version</code> and 1–200 <code>cases</code> (maximum 1 MB). Each case needs a unique <code>id</code>, a <code>question</code> and at least one of <code>expected_language</code>, <code>expected_intent</code> or <code>expected_entities</code>.</p><p>Omit an unreviewed task or entity type to exclude it from scoring. An empty entity list means no entity of that type is expected. Entity matching ignores case and repeated whitespace. Label mixed languages and multiple intents consistently; the current NLP returns one language and one intent. Review labels before uploading. The format example is not a research benchmark.</p><div data-labels></div></details>
      <div class="nlp-upload"><label>Reviewed dataset JSON<input type="file" accept=".json,application/json" data-file></label><div data-preview></div><button class="btn" type="button" data-upload disabled>Save dataset</button></div>
      <div class="nlp-actions"><label>Saved dataset<select data-dataset aria-label="Saved evaluation dataset"></select></label><button class="btn btn-primary" type="button" data-run disabled>Run NLP evaluation</button></div>
      <p data-status role="status" aria-live="polite">Loading evaluation records…</p><label>Run history (latest 50)<select data-history aria-label="Saved evaluation run"></select></label><div data-result></div>`;
    let pending = null;
    let loadingRun = 0;
    const status = message => { if (host.isConnected) host.querySelector('[data-status]').textContent = message; };
    const lockControls = () => {
      host.querySelector('[data-run]').disabled = busy || !host.querySelector('[data-dataset]').value;
      host.querySelector('[data-upload]').disabled = busy || !pending;
      for (const selector of ['[data-file]', '[data-dataset]', '[data-history]']) host.querySelector(selector).disabled = busy;
    };
    notifyCurrent = () => { if (host.isConnected) reload(host.querySelector('[data-dataset]').value).catch(error => status(error.message)); };
    const finish = () => { busy = false; lockControls(); if (!host.isConnected) notifyCurrent(); };
    function display(run) {
      if (!host.isConnected) return;
      const output = host.querySelector('[data-result]');
      if (!run) { output.innerHTML = '<p class="nlp-empty">No evaluation results yet. Save a reviewed dataset and run an evaluation.</p>'; return; }
      output.innerHTML = resultHtml(run);
      renderCases(output, run);
      output.querySelector('[data-case-filter]').onchange = () => renderCases(output, run);
      output.querySelectorAll('[data-export]').forEach(button => button.onclick = async () => {
        button.disabled = true;
        try {
          const response = await apiFetch(`/admin/nlp-evaluation/runs/${encodeURIComponent(run.id)}/export/${button.dataset.export}`);
          if (!response.ok) throw Error('Unable to export evaluation results.');
          download(`nlp-${run.id}-${button.dataset.export}.csv`, await response.text(), 'text/csv;charset=utf-8');
        } catch (error) { status(error.message); } finally { button.disabled = false; }
      });
    }
    async function loadRun(identifier) {
      const epoch = ++loadingRun;
      try { const run = await api('/runs/' + encodeURIComponent(identifier)); if (epoch === loadingRun && host.isConnected) { selectedRun = run.id; display(run); } }
      catch (error) { if (epoch === loadingRun) status(error.message); }
    }
    async function reload(datasetId) {
      const data = await api('');
      if (!host.isConnected) return;
      const datasets = host.querySelector('[data-dataset]');
      datasets.innerHTML = '<option value="">Select a dataset</option>' + data.datasets.map(row => `<option value="${esc(row.id)}">${esc(row.name)} · ${esc(row.version)} · ${row.case_count} cases</option>`).join('');
      datasets.value = datasetId || data.datasets[0]?.id || '';
      const history = host.querySelector('[data-history]');
      history.innerHTML = '<option value="">Select an evaluation run</option>' + data.runs.map(row => `<option value="${esc(row.id)}">${esc(row.dataset_name)} · ${esc(row.dataset_version)} · ${esc(new Date(row.created_at).toLocaleString())}</option>`).join('');
      history.value = data.runs.some(row => row.id === selectedRun) ? selectedRun : data.runs[0]?.id || '';
      host.querySelector('[data-labels]').innerHTML = `<p>Languages: ${esc(data.schema.languages.join(', '))}</p><p>Intents: ${esc(data.schema.intents.join(', '))}</p><p>Entity types: ${esc(data.schema.entity_types.join(', '))}</p>`;
      if (history.value) await loadRun(history.value); else display(null);
      status(busy ? 'An evaluation is running. Results will be saved when processing finishes.' : 'Ready. Dataset and run selectors show the latest 50 records.');
      lockControls();
    }
    host.querySelector('[data-example]').onclick = () => download('nlp-dataset-format-example.json', JSON.stringify({name: 'Format example — replace with reviewed benchmark', version: '1', cases: [
      {id: 'example_001', question: 'Hello!', expected_language: 'en', expected_intent: 'greeting', expected_entities: {money: [], law_references: []}},
      {id: 'example_002', question: 'My salary is PHP 15,000. How much is my 13th month pay?', expected_language: 'en', expected_intent: '13th_month', expected_entities: {money: ['PHP 15,000']}}
    ]}, null, 2));
    host.querySelector('[data-file]').onchange = async event => {
      pending = null; lockControls();
      const file = event.target.files[0];
      const preview = host.querySelector('[data-preview]'); preview.textContent = '';
      if (!file) return;
      try {
        if (file.size > 1000000) throw Error('Dataset must be at most 1 MB.');
        const value = JSON.parse((await file.text()).replace(/^\uFEFF/, ''));
        if (!value || !Array.isArray(value.cases) || value.cases.length < 1 || value.cases.length > 200) throw Error('Provide 1–200 cases. See the dataset format.');
        pending = value;
        preview.innerHTML = `<p>${esc(value.name || 'Name required')} · version ${esc(value.version || 'required')} · ${value.cases.length} cases. Preview of first 5:</p>${AdminUI.table(['ID', 'Question', 'Expected outputs'], value.cases.slice(0, 5).map(row => [esc(row?.id || ''), esc(row?.question || ''), `<pre>${esc(JSON.stringify({expected_language: row?.expected_language, expected_intent: row?.expected_intent, expected_entities: row?.expected_entities}, null, 2))}</pre>`]))}`;
        status('Preview loaded. Save validates all cases before storing the dataset.');
      } catch (error) { status(error.message); }
      lockControls();
    };
    host.querySelector('[data-upload]').onclick = async () => {
      if (busy || !pending) return;
      busy = true; lockControls(); status('Validating and saving dataset…');
      try {
        const value = await api('/datasets', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(pending)});
        pending = null; host.querySelector('[data-file]').value = ''; host.querySelector('[data-preview]').textContent = '';
        await reload(value.id); status('Dataset saved. Select Run NLP evaluation to measure it.');
      } catch (error) { status(error.message); } finally { finish(); }
    };
    host.querySelector('[data-run]').onclick = async () => {
      const identifier = host.querySelector('[data-dataset]').value;
      if (busy || !identifier) return;
      busy = true; lockControls(); status('Running existing NLP locally. Results will be saved; this may take a moment…');
      try {
        const run = await api('/datasets/' + encodeURIComponent(identifier) + '/runs', {method: 'POST'});
        selectedRun = run.id;
        await reload(identifier);
        status(`Evaluation saved: ${run.completed} completed, ${run.failed} failed.`);
      } catch (error) { status(error.message + ' Refresh run history to check for a saved result before retrying.'); }
      finally { finish(); }
    };
    host.querySelector('[data-dataset]').onchange = lockControls;
    host.querySelector('[data-history]').onchange = event => { if (event.target.value) loadRun(event.target.value); else { ++loadingRun; display(null); } };
    reload().catch(error => status(error.message));
  }
  return {mount};
})();
