/* Read-only rendering of the existing admin health response. */
(function(root){
  const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=value=>value==null||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString();
  const seconds=value=>value==null||!Number.isFinite(Number(value))?'Not recorded':`${(Number(value)/1000).toFixed(2)} s`;
  function renderScheduler(data){
    if(!data)return '<section class="performance-scheduler performance-unavailable"><h3>Model Request Pool</h3><p>Scheduling details unavailable. Restart the updated backend, then refresh performance.</p></section>';
    if(data.enabled!==true)return '<section class="performance-scheduler"><h3>Model Request Pool</h3><p>Model pool scheduling is disabled. Requests use the configured provider order.</p></section>';
    if(data.concurrency_uncapped===true){
      const values=[['Primary models',number(data.primary_model_count)],['Backup models',number(data.backup_model_count)],['Active requests',number(data.active_requests)],['App concurrency cap','No local cap']];
      return `<section class="performance-scheduler" aria-label="Generation model capacity"><header><h3>Model Request Pool</h3><span class="performance-model-status">Snapshot · this backend worker</span></header><p>Primary models share requests: idle first, then least busy. Occupied models can handle additional requests immediately.</p><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl><p>Gemini backups are used only after the OpenAI primary models fail or are unavailable—not just because they are busy.</p><p>No app-local generation concurrency cap or capacity queue. Provider limits and existing request deadlines still apply. Refresh performance to update the counts.</p></section>`;
    }
    const values=[['Models in pool',number(data.model_count)],['Active requests',number(data.active_requests)],['Configured slots',number(data.configured_slots)],['Waiting for a slot',number(data.waiting_requests)]];
    return `<section class="performance-scheduler" aria-label="Generation model capacity"><header><h3>Model Request Pool</h3><span class="performance-model-status">Snapshot · this backend worker</span></header><p>Use an idle model first. If all are occupied, use the least-busy model. Configured priority breaks ties.</p><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl><p>Up to <strong>${number(data.concurrency_per_model)}</strong> active requests per model · wait up to <strong>${number(data.queue_wait_seconds)} seconds</strong> for a slot · maximum <strong>${number(data.queue_max_waiters)}</strong> waiting requests.</p><p>Refresh performance to update these counts. Configured slots are maximum capacity, not a guarantee that every model is usable. Provider limits and existing request deadlines still apply.</p></section>`;
  }
  function renderLda(data){
    const status=data?.status||'Unavailable';
    const ready=data?.ready===true&&['Ready','Matching chunks','Matching incomplete'].includes(status);
    const tone=status==='Ready'?'good':['Failed','Needs documents','Unavailable','Matching incomplete'].includes(status)?'warning':'neutral';
    const values=[['Topics learned',ready?number(data.topic_count):'—'],['Training chunks',ready?number(data.trained_chunk_count):'—']];
    const coverage=data?.assignment_checked_chunks!=null?`<p class="performance-model-detail">Chunks processed: <strong>${number(data.assignment_checked_chunks)} / ${number(data.assignment_total_chunks)}</strong></p>`:'';
    return `<section class="performance-model-group"><header><h3>Background Topic-Model Training</h3><p>Shows training progress only. The current chat retrieval path does not use LDA search hints.</p></header><article class="performance-model-card performance-lda-card"><div class="performance-model-heading"><span>KNOWLEDGE BASE</span><span class="performance-model-status ${tone}" aria-live="polite">${escape(status)}</span></div><h4>Document topics</h4><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>${coverage}<p class="performance-model-detail">${escape(data?.detail||'Topic-learning status is unavailable. Refresh the page after restarting the backend.')}</p><p class="performance-model-detail">Topics are learned from a document sample; all chunks are then checked. A chunk can match several topics. Status is for this backend worker and resets after restart.</p></article></section>`;
  }
  function render(health){
    if(!health)return '<section class="performance-model-group"><h2>AI Models</h2><p>Model information is unavailable. Try Refresh.</p></section>';
    const rows=Array.isArray(health.providers)?health.providers:[];
    const groups=[['generation','Answer Generation','Primary models share the answer workload.','primary'],['generation','Backup Answer Generation','Used only after primary OpenAI models fail or are unavailable.','backup'],['reranking','Recovery Conflict Ranking','Used only when recovery finds conflicting evidence.']];
    return `<section class="performance-models"><h2>AI Models</h2><p class="performance-note">${escape(health.note||'Observations since backend restart; not a live availability check.')}</p>${renderScheduler(health.generation_scheduler)}${renderLda(health.lda)}${groups.map(([role,title,note,tier])=>{
      const models=rows.filter(row=>(row.role||'generation')===role&&(!tier||(row.generation_tier||'primary')===tier))
        .sort((a,b)=>Number(a.priority||999)-Number(b.priority||999));
      const orderNote=role==='generation'?(tier==='backup'?'Fallback-only · not used to balance a busy primary pool.':'Idle models first · least-busy models next · configured priority breaks ties.'):'Configured priority · first row is tried first.';
      return `<section class="performance-model-group"><header><h3>${title}</h3><p>${note}</p><p class="performance-model-order-note">${orderNote}</p></header>${models.length?`<ol class="performance-model-list" aria-label="${title} model priority order">${models.map((row,index)=>{
        const priority=Number.isFinite(Number(row.priority))&&Number(row.priority)>0?Math.floor(Number(row.priority)):index+1;
        const status=row.status||'Not used since restart';
        const tone=/fail|missing|timed out|limit reached/i.test(status)?'warning':/success|active|in use/i.test(status)?'good':'neutral';
        const metrics=row.metrics_available!==false;
        const values=[['Attempts',number(row.requests??0)],[role==='generation'?'Drafts created':'Successful',number(row.successes??0)],['Failed',number(row.failures??0)],['Average time',seconds(row.average_latency_ms)]];
        if(role==='generation'&&row.concurrency_limit!=null)values.push(['Active requests',`${number(row.active_requests)} / ${number(row.concurrency_limit)}`]);
        else if(role==='generation'&&row.concurrency_uncapped===true)values.push(['Active requests',number(row.active_requests)],['App concurrency cap','No local cap']);
        if(metrics&&row.local_tpm_limit!=null)values.push(['Estimated tokens · last minute',number(row.estimated_tokens_last_minute)],['Local token limit · shared',number(row.local_tpm_limit)]);
        const issues={invalid_ranking_order:'Invalid ranking order',invalid_json:'Unreadable structured response',output_truncated:'Response was cut off',empty_output:'Model returned no visible response',missing_output:'Model response missing',provider_rejected_request:'Provider rejected the request',authentication_failed:'Authentication failed',permission_denied:'Model access denied',model_not_found:'Model not found',request_too_large:'Request too large'};
        const issue=issues[row.last_error_detail]||'';
        const details=[];
        if(role==='generation'&&row.available_slots!=null)details.push(`Available slots ${number(row.available_slots)} · this backend worker`);
        if(metrics)details.push(`Timeouts ${number(row.timeouts??0)} · limit errors ${number(row.rate_limits??0)}`);
        if(row.local_rpm_limit!=null)details.push(`Calls ${number(row.calls_last_minute)} / ${number(row.local_rpm_limit)} per minute`);
        else if(row.calls_last_minute!=null)details.push(`Calls ${number(row.calls_last_minute)} per minute · no local cap`);
        if(row.retry_in_seconds>0)details.push(`Retry in about ${number(row.retry_in_seconds)} seconds`);
        if(issue&&row.status!=='Observed successes')details.push(`Last issue: ${issue}${row.last_http_status?` (HTTP ${number(row.last_http_status)})`:''}`);
        if(!metrics)details.push('No live metric recorded');
        return `<li class="performance-model-row" value="${priority}"><div class="performance-model-entry"><div class="performance-model-line"><span class="performance-model-provider">${escape(String(row.provider||'').toUpperCase())}</span><h4>${escape(row.model||'Unknown model')}</h4><span class="performance-model-status ${tone}" aria-live="polite">${escape(status)}</span></div><dl>${values.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>${details.length?`<p class="performance-model-detail">${details.map(escape).join(' · ')}</p>`:''}</div></li>`;
      }).join('')}</ol>`:'<p class="performance-model-empty">No models configured for this role.</p>'}</section>`;
    }).join('')}</section>`;
  }
  root.AdminModelReport={render};
})(globalThis);
