/* -- Performance -- */
function asBool(value) {
  return value === true || value === 1 || value === '1' || String(value).toLowerCase() === 'true';
}

function performanceResultBadge(row) {
  const status = String(row.result_status || '').toLowerCase();
  if (status === 'kb_found') return '<span class="badge badge-green">SOURCE FOUND</span>';
  if (status === 'clarified') return '<span class="badge badge-amber">ASKED FOR DETAILS</span>';
  if (status === 'handled') return '<span class="badge badge-blue">ANSWERED</span>';
  if (status === 'not_found' || !asBool(row.resolved)) return '<span class="badge badge-red">NOT FOUND</span>';
  return '<span class="badge badge-green">RESOLVED</span>';
}

async function loadPerformance() {
  try {
    const r=await apiFetch('/admin/query-performance');
    const qs=await r.json();
    if(!r.ok) throw new Error(qs.detail||'Failed to load performance logs.');
    const dr=await apiFetch('/admin/dashboard');
    const dd=dr.ok?await dr.json():{};
    const kr=await apiFetch('/kb/stats');
    const kd=kr.ok?await kr.json():{};
    const t=dd.totals||{};
    const kb=kbSummary(kd);
    const all=Array.isArray(qs)?qs:[];
    const periodDays=Number(adminReportPeriod()),cutoff=Date.now()-periodDays*86400000;
    const visibleRecords=all.filter(query=>!query.created_at||new Date(query.created_at).getTime()>=cutoff);
    const buckets={fast:0,medium:0,slow:0};
    visibleRecords.forEach(q=>{const ms=num(q.response_time_ms);if(ms<500)buckets.fast++;else if(ms<2000)buckets.medium++;else buckets.slow++;});
    const total=visibleRecords.length||1;
    const avgMs=visibleRecords.length?Math.round(visibleRecords.reduce((s,q)=>s+num(q.response_time_ms),0)/visibleRecords.length):0;
    document.getElementById('perf-stats').innerHTML=`
      <div class="stat-card green"><span class="stat-icon">Speed</span><div class="stat-lbl">Average Answer Time</div><div class="stat-val">${visibleRecords.length?avgMs:"—"}<span style="font-size:16px;color:var(--muted);">${visibleRecords.length?" ms":""}</span></div><div class="stat-sub">Last ${periodDays} days</div></div>
      <div class="stat-card blue"><span class="stat-icon">Total</span><div class="stat-lbl">Total Processed</div><div class="stat-val">${t.messages??all.length}</div><div class="stat-sub">All questions</div></div>
      <div class="stat-card ${kb.chunks>0?'gold':'red'}"><span class="stat-icon">Documents</span><div class="stat-lbl">Searchable Sections</div><div class="stat-val">${kb.chunks.toLocaleString()}</div><div class="stat-sub">${kb.indexedDocs} documents ready</div></div>
      <div class="stat-card ${buckets.slow>5?'red':'green'}"><span class="stat-icon">Slow</span><div class="stat-lbl">Slow (2s or more)</div><div class="stat-val">${buckets.slow}</div><div class="stat-sub">Questions</div></div>`;
    document.getElementById('rt-buckets').innerHTML=`
      <div style="margin-bottom:16px;"><div class="flex-between" style="margin-bottom:6px;"><span style="font-size:13px;font-weight:600;">Fast (&lt;500ms)</span><span style="font-size:13px;color:var(--green);font-weight:700;">${buckets.fast}</span></div><div class="progress-bar"><div class="progress-fill green" style="width:${Math.round((buckets.fast/total)*100)}%"></div></div></div>
      <div style="margin-bottom:16px;"><div class="flex-between" style="margin-bottom:6px;"><span style="font-size:13px;font-weight:600;">Medium (500ms to under 2s)</span><span style="font-size:13px;color:var(--amber);font-weight:700;">${buckets.medium}</span></div><div class="progress-bar"><div class="progress-fill gold" style="width:${Math.round((buckets.medium/total)*100)}%"></div></div></div>
      <div><div class="flex-between" style="margin-bottom:6px;"><span style="font-size:13px;font-weight:600;">Slow (2s or more)</span><span style="font-size:13px;color:var(--red);font-weight:700;">${buckets.slow}</span></div><div class="progress-bar"><div class="progress-fill red" style="width:${Math.round((buckets.slow/total)*100)}%"></div></div></div>`;
    await window.AdminWorkbench?.loadHealth('health-items');
    document.getElementById('perf-log-tbody').innerHTML=visibleRecords.length
      ?visibleRecords.slice(0,adminPageSize()).map((q,i)=>{const ms=num(q.response_time_ms);const cls=ms<500?'badge-green':ms<2000?'badge-amber':'badge-red';const query=String(q.query_text||'');return`<tr><td class="td-mono">${i+1}</td><td style="max-width:300px;font-size:13px;">${esc(query.substring(0,70))}${query.length>70?'...':''}<span class="status-note">${esc(q.anonymous_user||'Guest')} - ${esc((q.intent||'general').replace(/_/g,' '))}</span></td><td><span class="badge ${cls}">${ms}ms</span></td><td>${performanceResultBadge(q)}</td><td class="td-mono">${dateLabel(q.created_at)}</td></tr>`;}).join('')
      :'<tr><td colspan="5" class="empty">No answer times yet.</td></tr>';
  } catch(e){
    document.getElementById('perf-stats').innerHTML=`<div class="err-msg" style="grid-column:1/-1;">${esc(e.message)}</div>`;
    document.getElementById('rt-buckets').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;
    document.getElementById('health-items').innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;
    document.getElementById('perf-log-tbody').innerHTML=`<tr><td colspan="5"><div class="err-msg">${esc(e.message)}</div></td></tr>`;
  }
}
