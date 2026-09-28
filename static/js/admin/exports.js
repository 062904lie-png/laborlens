/* -- Analytics -- */
function csvCell(value) {
  const text=String(value??'').replace(/\r?\n/g,' ').replace(/"/g,'""');
  return `"${text}"`;
}

async function exportChatLogs() {
  try {
    const r=await apiFetch('/admin/chat-logs');
    const rows=await r.json();
    if(!r.ok) throw new Error(rows.detail||'Failed to export logs.');
    const headers=['id','session_id','anonymous_user','role','content','language','intent','confidence_level','confidence_reason','feedback_rating','feedback_comment','created_at'];
    const csv=[headers.join(',')].concat((rows||[]).map(row=>headers.map(h=>csvCell(row[h])).join(','))).join('\n');
    const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;
    a.download=`laborlens-chat-logs-${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast('Chat logs exported.');
  } catch(e){toast(e.message,true);}
}

async function exportAnalyticsReport() {
  try {
    const [dashboardRes, performanceRes, feedbackRes] = await Promise.all([
      apiFetch('/admin/dashboard'),
      apiFetch('/admin/query-performance'),
      apiFetch('/admin/feedback')
    ]);
    const dashboard = await dashboardRes.json();
    const performance = await performanceRes.json();
    const feedback = await feedbackRes.json();
    if(!dashboardRes.ok) throw new Error(dashboard.detail || 'Failed to export dashboard report.');
    if(!performanceRes.ok) throw new Error(performance.detail || 'Failed to export performance report.');
    if(!feedbackRes.ok) throw new Error(feedback.detail || 'Failed to export feedback report.');

    const t = dashboard.totals || {};
    const r = dashboard.resolution_summary || {};
    const f = dashboard.feedback_summary || {};
    const lines = [];
    lines.push('LaborLens Analytics Report');
    lines.push(`Generated,${csvCell(new Date().toLocaleString())}`);
    lines.push('');
    lines.push('Summary');
    lines.push('Metric,Value');
    [
      ['Registered users', t.users || 0],
      ['Sessions', t.sessions || 0],
      ['Total queries', r.total_queries ?? t.messages ?? 0],
      ['Indexed documents', t.docs || 0],
      ['Resolved queries', r.resolved_queries || 0],
      ['Unresolved queries', r.unresolved_queries ?? dashboard.unresolved ?? 0],
      ['Resolution rate', `${pct(r.resolved_queries, r.total_queries)}%`],
      ['Average response time', `${num(r.avg_response_ms)} ms`],
      ['Helpful feedback', f.helpful || 0],
      ['Not helpful feedback', f.not_helpful || 0],
      ['Satisfaction rate', `${pct(f.helpful, f.total)}%`]
    ].forEach(row => lines.push(row.map(csvCell).join(',')));

    lines.push('');
    lines.push('Top Questions');
    lines.push('Question,Intent,Times Asked,Last Asked');
    (dashboard.top_questions || []).forEach(row => {
      lines.push([row.question, row.intent, row.count, row.last_asked].map(csvCell).join(','));
    });

    lines.push('');
    lines.push('Intent Breakdown');
    lines.push('Intent,Count');
    (dashboard.intent_distribution || []).forEach(row => {
      lines.push([row.intent, row.count].map(csvCell).join(','));
    });

    lines.push('');
    lines.push('Language Distribution');
    lines.push('Language,Count');
    (dashboard.language_distribution || []).forEach(row => {
      lines.push([languageLabel(row.language), row.count].map(csvCell).join(','));
    });

    lines.push('');
    lines.push('Recent Performance');
    lines.push('Query,User,Intent,Response Time,Result,Date');
    (performance || []).slice(0, 100).forEach(row => {
      lines.push([
        row.query_text,
        row.anonymous_user || 'Guest',
        row.intent,
        `${row.response_time_ms || 0} ms`,
        row.result_status,
        row.created_at
      ].map(csvCell).join(','));
    });

    lines.push('');
    lines.push('Feedback Review');
    lines.push('User,Rating,Intent,Source Support,Answer Preview,Date');
    (feedback || []).forEach(row => {
      lines.push([
        row.anonymous_user || 'Guest',
        row.rating,
        row.intent,
        row.confidence_level,
        String(row.assistant_answer || '').replace(/\s+/g, ' ').slice(0, 180),
        row.created_at
      ].map(csvCell).join(','));
    });

    const blob = new Blob([lines.join('\n')], {type:'text/csv;charset=utf-8;'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `laborlens-analytics-report-${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast('Analytics report exported.');
  } catch(e) {
    toast(e.message, true);
  }
}

async function downloadSystemBackup() {
  try{
    const r=await apiFetch('/admin/system-backup');
    const data=await r.json();
    if(!r.ok) throw new Error(data.detail||'Failed to create system backup.');
    const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8;'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;
    a.download=`laborlens-system-backup-${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast('System backup exported.');
  }catch(e){toast(e.message,true);}
}
