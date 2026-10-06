/* Operational reports: no prompts, answers, source excerpts, or credentials. */
(() => {
  const U = AdminUI;
  const ms = value => Number.isFinite(value) ? `${(value / 1000).toFixed(2)} s` : 'Not recorded';
  const time = value => new Date(value).toLocaleString();
  const tones = {info: 'blue', warning: 'yellow', error: 'red'};
  let latest = null;
  function retrievalTrace(details) {
    const rows = [...(details.retrieval_candidate_trace || []),
                  ...(details.evidence_candidate_trace || [])];
    if (!rows.length) return '';
    return U.panel('Retrieved / selected / rejected evidence', U.table(
      ['Document ID', 'Region', 'Search method', 'Decision'], rows.map(row => [
        esc(row.doc_db_id || 'Unknown'), esc(row.region ? String(row.region).replace(/_/g, ' ').toUpperCase() : 'Not recorded'),
        esc((Array.isArray(row.retrieval_method) ? row.retrieval_method.join(', ') : row.retrieval_method) || ''),
        esc(String(row.decision || '').replace(/_/g, ' '))])));
  }
  const page = AdminPages.register('backend-diagnostics', 'Backend Diagnostics',
    'Find retrieval gaps, model errors and answer-check failures without opening the server terminal.',
    async (_search, section) => {
      const days = section.querySelector('[name="days"]').value;
      const level = section.querySelector('[name="level"]').value;
      const params = new URLSearchParams({days, limit: '50', offset: section.dataset.offset || '0'});
      if (level) params.set('level', level);
      const data = await AdminPages.api('/admin/backend-diagnostics?' + params);
      latest = data;
      const summary = data.summary || {};
      const outcomes = Object.entries(summary.outcomes || {});
      const body = U.cards([
        ['Chat requests', summary.requests || 0],
        ['Average total time', ms(summary.average_ms)],
        ['Warnings / errors', (summary.issues || []).reduce((sum, row) => sum + row.count, 0), 'yellow'],
        ['Storage dropped events', data.storage?.dropped_events || 0, data.storage?.dropped_events ? 'red' : 'blue'],
      ]) + `<p class="diagnostics-note">${esc(data.note)} Retention: ${Number(data.retention_days)} days.
        ${data.storage?.enabled === false ? 'Diagnostics are disabled in backend configuration.' : ''}</p>` +
        U.panel('Response outcomes', U.table(['Outcome', 'Requests'], outcomes.map(([state, count]) => [esc(state), Number(count)]))) +
        U.panel('Recent backend events', U.table(['Time', 'Level', 'Step', 'Observation', 'Request', 'Details'],
          (data.items || []).map(row => [esc(time(row.created_at)), U.badge(row.level, tones[row.level]),
            esc(row.component), `${esc(row.summary)}${row.details.reason ? `<br><small>${esc(row.details.reason)}</small>` : ''}
              ${row.details.model ? `<br><small>${esc(row.details.model)}</small>` : ''}`,
            row.request_id && row.request_id !== 'unknown' ? `<button class="btn btn-sm" data-request="${esc(row.request_id)}">${esc(row.request_id.slice(0, 8))}</button>` : 'Background task',
            `<button class="btn btn-sm" data-event="${Number(row.id)}">Inspect</button>`]))) +
        '<div class="workbench-pager diagnostics-pager"></div>';
      return {html: body, mount(result) {
        U.pager(result.querySelector('.diagnostics-pager'), data.offset, data.total, offset => {
          section.dataset.offset = String(offset); AdminPages.open('backend-diagnostics');
        }, 50);
        result.querySelectorAll('[data-event]').forEach(button => button.onclick = () => {
          const event = data.items.find(row => row.id === Number(button.dataset.event));
          U.drawer(event.summary, `<p>${esc(time(event.created_at))} · ${esc(event.component)}</p>
            <p>Request ID: ${esc(event.request_id)}</p>${retrievalTrace(event.details)}<pre class="diagnostics-json">${esc(JSON.stringify(event.details, null, 2))}</pre>`);
        });
        result.querySelectorAll('[data-request]').forEach(button => button.onclick = async () => {
          U.drawer('Request timeline', '<div class="loading">Loading...</div>');
          try {
            const timeline = await AdminPages.api('/admin/backend-diagnostics?' + new URLSearchParams({days, limit: '200', request_id: button.dataset.request}));
            U.drawer('Request timeline', `<p>Request ${esc(button.dataset.request)}</p>` + U.table(['Time', 'Step', 'Observation', 'Metrics'],
              timeline.items.slice().reverse().map(row => [esc(time(row.created_at)), esc(row.component), esc(row.summary),
                `<pre class="diagnostics-json">${esc(JSON.stringify(row.details, null, 2))}</pre>`])));
          } catch (error) {
            U.drawer('Request timeline', `<p class="err-msg">${esc(error.message)}</p>`);
          }
        });
      }};
    });
  const form = page.querySelector('form');
  form.querySelector('label').remove();
  form.querySelector('[type="reset"]').remove();
  form.insertAdjacentHTML('afterbegin', `<label>Period<select name="days"><option value="1">Last 24 hours</option>
    <option value="7" selected>Last 7 days</option><option value="30">Last 30 days</option></select></label>
    <label>Level<select name="level"><option value="">All events</option><option value="warning">Warnings</option>
    <option value="error">Errors</option><option value="info">Information / outcomes</option></select></label>`);
  form.querySelector('button').textContent = 'Refresh report';
  const download = document.createElement('button');
  download.type = 'button'; download.className = 'btn'; download.textContent = 'Download this page (JSON)';
  download.onclick = () => {
    if (!latest) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(latest, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'laborlens-backend-diagnostics.json';
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  form.append(download);
})();
