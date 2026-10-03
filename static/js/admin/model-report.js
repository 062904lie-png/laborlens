/* Read-only rendering of the existing admin health response. */
(function(root){
  const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=value=>value==null||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString();
  const seconds=value=>value==null||!Number.isFinite(Number(value))?'Not recorded':`${(Number(value)/1000).toFixed(2)} s`;
  function renderLda(data){
    const status=data?.status||'Unavailable';
    const ready=data?.ready===true&&['Ready','Matching chunks','Matching incomplete'].includes(status);
    const tone=status==='Ready'?'good':['Failed','Needs documents','Unavailable','Matching incomplete'].includes(status)?'warning':'neutral';
    const values=[['Topics learned',ready?number(data.topic_count):'—'],['Training chunks',ready?number(data.trained_chunk_count):'—'],['Topic selection',data?.topic_selection||'—'],['Search hints',data?(data.search_hints_enabled?(ready?'On':'Waiting'):'Off'):'—']];
    const coverage=data?.assignment_checked_chunks!=null?`<p class="performance-model-detail">Chunks checked: <strong>${number(data.assignment_checked_chunks)} / ${number(data.assignment_total_chunks)}</strong> · With topic matches: ${number(data.assignment_classified_chunks)} · No clear match: ${number(data.assignment_unclassified_chunks)}</p>`:'';
    return `<section class="performance-model-group"><header><h3>Topic Learning (LDA)</h3><p>Helps find related document passages. It does not write or check answers.</p></header><article class="performance-model-card performance-lda-card"><div class="performance-model-heading"><span>KNOWLEDGE BASE</span><span class="performance-model-status ${tone}" aria-live="polite">${escape(status)}</span></div><h4>Document topics</h4><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>${coverage}<p class="performance-model-detail">${escape(data?.detail||'Topic-learning status is unavailable. Refresh the page after restarting the backend.')}</p><p class="performance-model-detail">Topics are learned from a document sample; all chunks are then checked. A chunk can match several topics, with no top-10 cutoff. Unclear matches never block search. Status is for this backend worker and resets after restart.</p></article></section>`;
  }
  function render(health){
    if(!health)return '<section class="performance-model-group"><h2>AI Models</h2><p>Model information is unavailable. Try Refresh.</p></section>';
    const rows=Array.isArray(health.providers)?health.providers:[];
    const groups=[['generation','Answer Generation','Creates a draft; a successful draft still needs answer checking.'],['reranking','Source Ranking','Orders retrieved passages before the answer is written.'],['verification','Answer Checking','Checks draft statements against the sources. A completed check may approve or reject the draft.']];
    return `<section class="performance-models"><h2>AI Models</h2><p class="performance-note">${escape(health.note||'Observations since backend restart; not a live availability check.')}</p>${renderLda(health.lda)}${groups.map(([role,title,note])=>{
      const models=rows.filter(row=>(row.role||'generation')===role);
      return `<section class="performance-model-group"><header><h3>${title}</h3><p>${note}</p></header><div class="performance-model-grid">${models.length?models.map(row=>{
        const status=row.status||'Not used since restart';
        const tone=/fail|missing|timed out|limit reached/i.test(status)?'warning':/success|active|in use/i.test(status)?'good':'neutral';
        const metrics=row.metrics_available!==false;
        const values=[['Attempts',number(row.requests??0)],[role==='verification'?'Completed checks':role==='generation'?'Drafts created':'Successful',number(row.successes??0)],['Failed',number(row.failures??0)],['Average time',seconds(row.average_latency_ms)]];
        if(metrics&&row.local_tpm_limit!=null)values.push(['Estimated tokens · last minute',number(row.estimated_tokens_last_minute)],['Local token limit · shared',number(row.local_tpm_limit)]);
        const issues={quotation_mismatch:'Source quotation did not match',user_fact_quotation_mismatch:'User fact quotation did not match',invalid_source_id:'Invalid source reference',invalid_claim_id:'Invalid statement reference',invalid_claim_list:'Invalid statement list',missing_supporting_evidence:'Supporting evidence missing',invalid_ranking_order:'Invalid ranking order',invalid_json:'Unreadable structured response',output_truncated:'Response was cut off',empty_output:'Model returned no visible response',missing_output:'Model response missing',provider_rejected_request:'Provider rejected the request',authentication_failed:'Authentication failed',permission_denied:'Model access denied',model_not_found:'Model not found',request_too_large:'Request too large'};
        issues.invalid_fragment_id='Invalid source fragment reference';
        issues.invalid_claim_text='Statement could not be matched to the draft';
        issues.invalid_question_points='Requested answer points were missing or invalid';
        issues.invalid_question_point='Requested answer point was invalid';
        issues.invalid_evidence_list='Source evidence list was invalid';
        issues.invalid_verdict='Statement check result was invalid';
        issues.legal_rule_as_user_fact='Legal claim was incorrectly treated as a user fact';
        const issue=issues[row.last_error_detail]||'';
        return `<article class="performance-model-card"><div class="performance-model-heading"><span>${escape(String(row.provider||'').toUpperCase())}</span><span class="performance-model-status ${tone}">${escape(status)}</span></div><h4>${escape(row.model||'Unknown model')}</h4><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>${metrics?`<p class="performance-model-detail">Timeouts: ${number(row.timeouts??0)} · Request-limit errors: ${number(row.rate_limits??0)}</p>`:''}${issue&&row.status!=='Observed successes'?`<p class="performance-model-detail">Last issue: ${escape(issue)}${row.last_http_status?` (HTTP ${number(row.last_http_status)})`:''}</p>`:''}${row.local_rpm_limit!=null?`<p class="performance-model-detail">Calls in last minute: <strong>${number(row.calls_last_minute)} / ${number(row.local_rpm_limit)}</strong> local limit, shared across roles</p>`:''}${row.retry_in_seconds>0?`<p class="performance-model-detail">Retry available in about ${number(row.retry_in_seconds)} seconds. Refresh to update.</p>`:''}${!metrics?'<p class="performance-model-detail">Configured does not mean a live connection has been tested.</p>':''}</article>`;
      }).join(''):'<p>No models configured for this role.</p>'}</div></section>`;
    }).join('')}</section>`;
  }
  root.AdminModelReport={render};
})(globalThis);
